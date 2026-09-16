"""
The contact-mic preprocessing contract, inference side.

This MUST stay in lock-step with what ``pingest export --type polyumi`` writes as ``data/mic_0``:
the policy only compares like with like, so the waveform baked into training and the waveform this
node feeds the policy have to be the same numbers. As with
:mod:`polyumi_ros2.camera_preproc`, the ingest and ROS packages cannot share a Python import, so
the contract is mirrored here on purpose. See ``docs/data-format.md`` and the Vista fork's
``docs/SENSOR_PROCESSING.md``.

Three properties define a ``mic_0`` row, and all three are easy to get subtly wrong:

* **Source** — the *piezo* contact mic, which the Pi records as channel 0 of a stereo stream.
  Channel 1 is the air mic and is not what any policy was trained on.
* **Scale** — float32 in ``[-1, 1)``. ``pzarr`` builds it from the WAV as
  ``int16 / 32768``, so the same divisor is applied here rather than shipping raw counts.
* **Causality** — a row is the audio *ending at* the observation instant. Never any sample from
  after it: at inference that audio does not exist yet, and a policy trained on look-ahead cannot
  be served. ``config/contact_audio.yaml`` records this as ``block_alignment: causal`` and the
  exported buffer carries it in ``meta.attrs['mic_0_block_alignment']``.

**On the row width.** The exporter builds a row from ``stride`` per-frame blocks of
``samples_per_gopro_frame`` samples each (2 x 268 = 536 at the shipped settings), anchored by
``searchsorted`` on each GoPro frame's own timestamp. Those two blocks abut: at 59.94 fps they are
~267 samples apart and 268 wide, so together they span the ~33.5 ms ending at the frame the step
was taken from. This module therefore takes a row as one contiguous span of
:data:`MIC0_SAMPLES_PER_ROW` samples ending at the observation instant, which reproduces the
exporter to within the ~1-sample seam its deliberate block overlap creates. Doing it this way
makes the contract independent of the *camera* rate, which differs between recording (a 59.94 fps
GoPro) and inference (the Elgato capture) — anchoring to frame indices here would silently encode
the wrong span.
"""

import numpy as np

#: Sample rate the Pi records the finger mics at, asserted rather than trusted — see
#: ``ingest/config/contact_audio.yaml``.
MIC0_SAMPLE_RATE_HZ = 16000

#: Samples in one ``mic_0`` row: ``stride * samples_per_gopro_frame`` = 2 * 268 at the shipped
#: settings, i.e. ~33.5 ms. This is the width the exported buffer records as
#: ``mic_0_samples_per_step``, and the policy's ``shape_meta`` repeats it as ``mic_0: [536]``.
MIC0_SAMPLES_PER_ROW = 536

#: Which channel of the Pi's stereo finger audio is the contact mic. ``pzarr`` splits the WAV as
#: ``finger_piezo = audio[:, 0]`` / ``finger_air = audio[:, 1]``; ``mic_0`` is built from the
#: piezo alone.
MIC0_PIEZO_CHANNEL = 0

#: Divisor taking int16 PCM to ``[-1, 1)``. ``-np.iinfo(np.int16).min``, matching
#: ``polyumi_ingest.pzarr.store._read_wav``.
MIC0_INT16_SCALE = 32768.0

#: The wire/message format the Pi bridge publishes. Anything else means the audio pipeline
#: changed underneath this contract.
MIC0_EXPECTED_FORMAT = 'pcm-s16'


def decode_piezo(data: bytes, number_of_channels: int) -> np.ndarray:
    """
    Decode one ``foxglove_msgs/RawAudio`` block to the piezo channel as float32 in ``[-1, 1)``.

    The message carries interleaved little-endian ``pcm-s16``; this de-interleaves, keeps
    :data:`MIC0_PIEZO_CHANNEL`, and applies the same scaling the pzarr builder applies to the WAV.

    :param data: the message's ``data`` field — interleaved int16 little-endian bytes.
    :param number_of_channels: the message's ``number_of_channels``.
    :returns: 1-D float32, one element per frame of the block.
    :raises ValueError: if the channel count cannot hold the piezo channel, or the byte count is
        not a whole number of interleaved frames. Both mean the stream is not what this contract
        describes, and a silently mis-strided buffer sounds like plausible audio.
    """
    if number_of_channels < 2:
        raise ValueError(
            f'contact-mic audio must be at least stereo (L=piezo, R=air); the message declares '
            f'{number_of_channels} channel(s). The pzarr builder rejects the same stream for the '
            f'same reason — a mono capture is not this contract, and reading channel 0 out of it '
            f'would silently produce plausible audio from an unknown microphone.'
        )
    flat = np.frombuffer(data, dtype='<i2')
    if flat.size % number_of_channels:
        raise ValueError(
            f'audio block holds {flat.size} int16 sample(s), not a whole number of {number_of_channels}-channel frames.'
        )
    piezo = flat.reshape(-1, number_of_channels)[:, MIC0_PIEZO_CHANNEL]
    return piezo.astype(np.float32) / MIC0_INT16_SCALE


def mic0_rows(piezo: np.ndarray, end_index: int, n_rows: int) -> np.ndarray:
    """
    Cut ``n_rows`` contiguous ``mic_0`` rows ending at ``end_index``.

    The rows tile a single gapless span of ``n_rows * MIC0_SAMPLES_PER_ROW`` samples that *ends*
    at ``end_index`` (exclusive), oldest row first — the order the sampler produced at training,
    and the order the Vista frontend flattens back into a waveform before the mel.

    History shorter than the window is zero-padded at the **front**, matching the exporter, which
    fills a row's out-of-episode blocks with silence rather than repeating the nearest audio.
    That keeps a rollout's first observations honest about having no past instead of inventing a
    stationary one.

    :param piezo: 1-D float32 buffer of piezo samples in ``[-1, 1)``, oldest first.
    :param end_index: index one past the last sample belonging to the observation instant.
    :param n_rows: rows to return — the policy's ``audio_obs_horizon``.
    :returns: ``(n_rows, MIC0_SAMPLES_PER_ROW)`` float32.
    :raises ValueError: if ``n_rows`` is not positive or ``end_index`` is out of range.
    """
    if n_rows <= 0:
        raise ValueError(f'n_rows must be positive, got {n_rows}')
    if not 0 <= end_index <= piezo.shape[0]:
        raise ValueError(f'end_index {end_index} outside buffer of {piezo.shape[0]} sample(s)')

    span = n_rows * MIC0_SAMPLES_PER_ROW
    start = end_index - span
    if start >= 0:
        window = piezo[start:end_index]
    else:
        window = np.concatenate((np.zeros(-start, dtype=np.float32), piezo[0:end_index]))
    return np.ascontiguousarray(window, dtype=np.float32).reshape(n_rows, MIC0_SAMPLES_PER_ROW)
