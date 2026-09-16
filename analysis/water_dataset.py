"""
Turn recorded water-shake trials into per-trial TENSORS for a learned classifier.

Reads the bags ``collect_water_trials.sh`` writes -- ``<root>/<level>/trial_<n>`` -- and emits,
per trial, the raw-ish inputs a CNN encoder wants rather than hand-picked statistics:

* ``vision``   ``(K, 3, IMG, IMG)`` uint8  -- K frames sampled evenly across the shake
* ``tactile``  ``(K, 3, IMG, IMG)`` uint8  -- same, from the finger camera
* ``audio``    ``(1, N_MELS, MEL_T)`` float16 -- one log-mel spectrogram of the whole shake

Frames are SAMPLED EVENLY rather than taken from a window, because the shake is a fixed 5-cycle
motion: sampling across it means frame k of every trial is at the same point in the same motion,
which is the invariance the classifier gets to exploit instead of having to learn alignment.

Decoding once, offline, also keeps the four model variants honest -- they see byte-identical
inputs and differ only in which encoders are attached.

Usage (on polyumi-server, with ROS sourced):
    python3 water_dataset.py --root /data/eval_results_9_6/water --out water_tensors.npz
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

#: Fallback only. Classes are normally DISCOVERED from the directories under --root, so the same
#: pipeline serves water levels, materials, or anything else collected the same way.
DEFAULT_LEVELS = ('empty', 'some', 'half')

AUDIO_TOPIC = '/pi/audio/raw'
TACTILE_TOPIC = '/pi/camera/image/compressed'
VISION_TOPIC = '/gopro/image_raw/compressed'

N_MELS = 64
N_FFT = 512
HOP = 160  # 10 ms at 16 kHz
MEL_T = 256  # every trial's spectrogram is resampled to this many frames
# The finger camera's native size; the GoPro is scaled down to match, so both cameras reach the
# encoders at one resolution.
IMG = 224
K_FRAMES = 8  # frames kept per trial per camera, evenly spaced across the window
TAIL_S = 0.0  # seconds of the END of each bag to keep; 0 keeps the WHOLE bag. See --tail-s.
# Low edge of the mel filterbank, in Hz. 20 keeps everything the mic hears; see mel_filterbank
# for why that buries the signal under shake rumble. Override with --f-min.
F_MIN = 20.0


def mel_filterbank(n_fft: int, n_mels: int, sr: int, f_min: float = F_MIN) -> np.ndarray:
    """
    Triangular mel filterbank, so this needs no torchaudio/librosa in the ROS env.

    `f_min` matters far more than it looks. The shake puts most of the recorded energy into arm
    rumble below a couple of hundred hertz -- measured on the rv4 set, 76% of the power inside a
    20 Hz floor sits in 20-200 Hz, a band that classifies the three materials at barely above
    chance. The bands that DO separate them (500 Hz-1 kHz and 2-4 kHz) carry 1-2% of the power
    each, so with the floor at 20 Hz they are a rounding error in the spectrogram and the encoder
    largely ignores them. Raising the floor spends every mel bin on the part of the spectrum that
    carries the label.
    """

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    pts = mel_to_hz(np.linspace(hz_to_mel(f_min), hz_to_mel(sr / 2), n_mels + 2))
    bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, mid, hi = bins[m], bins[m + 1], bins[m + 2]
        if mid > lo:
            fb[m, lo:mid] = np.linspace(0, 1, mid - lo, endpoint=False)
        if hi > mid:
            fb[m, mid:hi] = np.linspace(1, 0, hi - mid, endpoint=False)
    return fb


def log_mel(samples: np.ndarray, sr: int, fb: np.ndarray) -> np.ndarray:
    """Log-mel spectrogram, shape (n_mels, frames)."""
    if len(samples) < N_FFT:
        return np.zeros((fb.shape[0], 1))
    win = np.hanning(N_FFT)
    n_frames = 1 + (len(samples) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = samples[idx] * win
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    return np.log(fb @ power.T + 1e-10)


def _sample_frames(frames: list[np.ndarray], k: int = K_FRAMES) -> np.ndarray:
    """Evenly sample `k` frames as (k, 3, IMG, IMG) uint8, padding by repeat if the trial is short."""
    if not frames:
        return np.zeros((k, 3, IMG, IMG), dtype=np.uint8)
    idx = np.linspace(0, len(frames) - 1, k).round().astype(int)
    picked = np.stack([frames[i] for i in idx])  # (k, IMG, IMG, 3)
    return np.ascontiguousarray(picked.transpose(0, 3, 1, 2))


def _resample_mel(lm: np.ndarray, t: int = MEL_T) -> np.ndarray:
    """Stretch or squeeze a log-mel to a fixed frame count so every trial is one tensor shape."""
    if lm.shape[1] == t:
        return lm
    src = np.linspace(0.0, 1.0, lm.shape[1])
    dst = np.linspace(0.0, 1.0, t)
    return np.stack([np.interp(dst, src, row) for row in lm])


def read_trial(bag: pathlib.Path, tail_s: float = TAIL_S, f_min: float = F_MIN) -> dict[str, np.ndarray]:
    """
    Extract the three modality tensors from one bag; `tail_s` > 0 keeps only its last seconds.

    The default keeps the whole bag. Cropping to a fixed tail is the one thing that reliably
    destroys this dataset: the recorder subscribes, the arm levels and settles, shakes, and then
    holds still while the recording runs on. The shake lands in the MIDDLE, so "the last N seconds"
    is post-shake silence, not the experiment. Measured on the `_pr` set, the shake occupies
    seconds 6.5-10 of a 17.6 s bag, and the 1-2 kHz band that separates the materials carries
    19-22% of power there against ~0.1% in the final six seconds.

    Nothing errors when the window is wrong -- the tensors have the right shape and the only
    symptom is a classifier stuck near chance -- so prefer the whole bag and let the encoder's
    global pool find the active stretch. Pass `tail_s` only after checking, on these recordings,
    that the window you are asking for contains the motion.
    """
    import cv2
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id='mcap'),
        rosbag2_py.ConverterOptions('', ''),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}

    pcm: list[np.ndarray] = []
    pcm_t: list[int] = []
    sr = 16000
    tac: list[np.ndarray] = []
    tac_t: list[int] = []
    vis: list[np.ndarray] = []
    vis_t: list[int] = []
    t_last = 0

    while reader.has_next():
        topic, data, stamp = reader.read_next()
        t_last = max(t_last, stamp)
        if topic == AUDIO_TOPIC:
            m = deserialize_message(data, get_message(types[topic]))
            sr = int(m.sample_rate)
            s = np.frombuffer(bytes(m.data), dtype='<i2').astype(np.float64) / 32768.0
            if int(m.number_of_channels) > 1:  # interleaved -> take one channel
                s = s[:: int(m.number_of_channels)]
            pcm.append(s)
            pcm_t.append(stamp)
        elif topic in (TACTILE_TOPIC, VISION_TOPIC):
            m = deserialize_message(data, get_message(types[topic]))
            img = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            small = cv2.resize(img, (IMG, IMG), interpolation=cv2.INTER_AREA)
            if topic == TACTILE_TOPIC:
                tac.append(small)
                tac_t.append(stamp)
            else:
                vis.append(small)
                vis_t.append(stamp)

    # Trim to the tail, if one was asked for. Done on message timestamps rather than by counting
    # messages, because the three streams run at different rates and a fixed count would cover
    # different spans of time.
    if tail_s > 0:
        cutoff = t_last - int(tail_s * 1e9)
        pcm = [x for x, t in zip(pcm, pcm_t) if t >= cutoff] or pcm
        tac = [x for x, t in zip(tac, tac_t) if t >= cutoff] or tac
        vis = [x for x, t in zip(vis, vis_t) if t >= cutoff] or vis

    fb = mel_filterbank(N_FFT, N_MELS, sr, f_min)
    if pcm:
        lm = _resample_mel(log_mel(np.concatenate(pcm), sr, fb))
    else:
        lm = np.zeros((N_MELS, MEL_T))

    return {
        'audio': lm[None].astype(np.float16),
        'tactile': _sample_frames(tac),
        'vision': _sample_frames(vis),
        'n_audio': np.array([len(pcm)]),
        'n_tactile': np.array([len(tac)]),
        'n_vision': np.array([len(vis)]),
    }


def main() -> int:
    """Walk the trial tree and write one npz with every modality block and the labels."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument(
        '--levels',
        nargs='+',
        default=None,
        help='class labels, in order; default is every sub-directory of --root that holds trials, sorted',
    )
    ap.add_argument(
        '--tail-s',
        type=float,
        default=TAIL_S,
        help='seconds at the END of each bag to use; 0 (the default) uses the whole '
        'bag. The shake sits in the middle of these recordings, so a tail crop '
        'silently feeds the model post-shake silence -- check before setting it',
    )
    ap.add_argument(
        '--val-from',
        type=int,
        default=21,
        help='first trial number of the validation split',
    )
    ap.add_argument(
        '--f-min',
        type=float,
        default=F_MIN,
        help='low edge of the mel filterbank in Hz (default 20). Raising it to ~300 drops the '
        'arm-rumble band, which carries most of the power and almost none of the label',
    )
    ap.add_argument(
        '--test-from',
        type=int,
        default=31,
        help='first trial number of the TEST split, which must be higher than --val-from. '
        'Trials below --val-from train, those between the two validate, those at or above '
        'this test. Set it above the highest trial number to collect no test split at all.',
    )
    args = ap.parse_args()

    levels = args.levels
    if levels is None:
        levels = sorted(d.name for d in args.root.iterdir() if d.is_dir() and any(d.glob('trial_*')))
    if not levels:
        print(f'no class directories with trials under {args.root}', file=sys.stderr)
        return 1
    print(f'classes: {list(levels)}')

    rows: dict[str, list] = {'audio': [], 'tactile': [], 'vision': []}
    labels, names, counts, splits = [], [], [], []
    for li, level in enumerate(levels):
        for bag in sorted((args.root / level).glob('trial_*')):
            if not bag.is_dir():
                continue
            f = read_trial(bag, tail_s=args.tail_s, f_min=args.f_min)
            for k in rows:
                rows[k].append(f[k])
            labels.append(li)
            names.append(f'{level}/{bag.name}')
            # Split by trial NUMBER, not at random, and into three parts. Collection is
            # interleaved in blocks, so consecutive trial numbers within a class come from one
            # block -- which makes each split a DIFFERENT block from the others. That is the
            # point: a model that has merely learnt a per-block nuisance (the finger camera's
            # level and colour balance settle slightly differently each time the rig is
            # disturbed) cannot carry it across the boundary, whereas a random split would hand
            # it trials from the same block it trained on and score it far too well. It also
            # needs no seed to reproduce.
            n_trial = int(bag.name.split('_')[-1])
            if n_trial >= args.test_from:
                splits.append('test')
            elif n_trial >= args.val_from:
                splits.append('val')
            else:
                splits.append('train')
            counts.append((int(f['n_audio'][0]), int(f['n_tactile'][0]), int(f['n_vision'][0])))
            print(
                f'  {level:<6} {bag.name:<12} audio={counts[-1][0]:>4} '
                f'tactile={counts[-1][1]:>4} vision={counts[-1][2]:>4}',
                flush=True,
            )

    if not labels:
        print(f'no trials under {args.root}', file=sys.stderr)
        return 1

    out = {k: np.asarray(v) for k, v in rows.items()}
    out['label'] = np.asarray(labels)
    out['name'] = np.asarray(names)
    out['levels'] = np.asarray(levels)
    out['split'] = np.asarray(splits)
    np.savez(args.out, **out)
    counts_arr = np.asarray(counts)
    print(
        f'\nwrote {args.out}: {len(labels)} trials  '
        f'{ {levels[i]: int((out["label"] == i).sum()) for i in range(len(levels))} }'
    )
    print('  tensor shapes: ' + ', '.join(f'{k}{out[k].shape[1:]}' for k in rows))
    n_tr = sum(1 for sp in splits if sp == 'train')
    n_va = splits.count('val')
    n_te = splits.count('test')
    print(
        f'  split: {n_tr} train / {n_va} val / {n_te} test  '
        f'(val = trial >= {args.val_from}, test = trial >= {args.test_from})'
    )
    if n_te == 0:
        print('  note: no test trials -- --test-from is above every trial number present')
    print(
        f'  msgs/trial (min): audio {counts_arr[:, 0].min()} '
        f'tactile {counts_arr[:, 1].min()} vision {counts_arr[:, 2].min()}'
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
