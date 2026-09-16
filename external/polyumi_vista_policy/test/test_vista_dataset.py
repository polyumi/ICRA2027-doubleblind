"""VistaDataset tests (require dataset on disk)."""

import os

import numpy as np

from vista.data.sampler import VistaSequenceSampler


def test_audio_zero_pad_at_episode_start():
    """Audio rows before episode start are zero, not repeated."""
    shape_meta = {
        "obs": {
            "mic_0": {
                "shape": [536],
                "horizon": 4,
                "latency_steps": 0,
                "down_sample_steps": 1,
                "type": "audio",
            }
        },
        "action": {"shape": [10], "horizon": 16, "latency_steps": 0, "down_sample_steps": 3},
    }
    replay = {
        "mic_0": np.ones((10, 536), dtype=np.float32),
        "robot0_gripper_width": np.ones((10, 1), dtype=np.float32),
        "action": np.zeros((10, 10), dtype=np.float32),
    }

    class _Buf:
        episode_ends = np.array([10], dtype=np.int64)

        def __getitem__(self, key):
            return replay[key]

        def __contains__(self, key):
            return key in replay

        def keys(self):
            return replay.keys()

    sampler = VistaSequenceSampler(
        shape_meta=shape_meta,
        replay_buffer=_Buf(),
        rgb_keys=[],
        lowdim_keys=[],
        audio_keys=["mic_0"],
        key_horizon={"mic_0": 4, "action": 16},
        key_latency_steps={"mic_0": 0, "action": 0},
        key_down_sample_steps={"mic_0": 1, "action": 3},
        action_padding=True,
    )
    seq = sampler.sample_sequence(0)
    assert seq["mic_0"].shape == (4, 536)
    assert np.all(seq["mic_0"][:3] == 0)
    assert np.all(seq["mic_0"][3] == 1)


def test_audio_contiguous_horizon_10():
    """SHF/Sparsh/Qformer: 10 contiguous mic rows ending at current_idx (ds=1)."""
    shape_meta = {
        "obs": {
            "mic_0": {
                "shape": [536],
                "horizon": 10,
                "latency_steps": 0,
                "down_sample_steps": 1,
                "type": "audio",
            }
        },
        "action": {"shape": [10], "horizon": 16, "latency_steps": 0, "down_sample_steps": 3},
    }
    replay = {
        "mic_0": np.arange(40, dtype=np.float32).reshape(40, 1).repeat(536, axis=1),
        "robot0_gripper_width": np.ones((40, 1), dtype=np.float32),
        "action": np.zeros((40, 10), dtype=np.float32),
    }

    class _Buf:
        episode_ends = np.array([40], dtype=np.int64)

        def __getitem__(self, key):
            return replay[key]

        def __contains__(self, key):
            return key in replay

        def keys(self):
            return replay.keys()

    sampler = VistaSequenceSampler(
        shape_meta=shape_meta,
        replay_buffer=_Buf(),
        rgb_keys=[],
        lowdim_keys=[],
        audio_keys=["mic_0"],
        key_horizon={"mic_0": 10, "action": 16},
        key_latency_steps={"mic_0": 0, "action": 0},
        key_down_sample_steps={"mic_0": 1, "action": 3},
        action_padding=True,
    )
    seq = sampler.sample_sequence(20)
    assert seq["mic_0"].shape == (10, 536)
    assert seq["mic_0"][0, 0] == 11.0
    assert seq["mic_0"][-1, 0] == 20.0


def test_audio_contiguous_horizon_17():
    """Sampler still supports longer contiguous audio windows (legacy / ablations)."""
    shape_meta = {
        "obs": {
            "mic_0": {
                "shape": [536],
                "horizon": 17,
                "latency_steps": 0,
                "down_sample_steps": 1,
                "type": "audio",
            }
        },
        "action": {"shape": [10], "horizon": 16, "latency_steps": 0, "down_sample_steps": 3},
    }
    replay = {
        "mic_0": np.arange(40, dtype=np.float32).reshape(40, 1).repeat(536, axis=1),
        "robot0_gripper_width": np.ones((40, 1), dtype=np.float32),
        "action": np.zeros((40, 10), dtype=np.float32),
    }

    class _Buf:
        episode_ends = np.array([40], dtype=np.int64)

        def __getitem__(self, key):
            return replay[key]

        def __contains__(self, key):
            return key in replay

        def keys(self):
            return replay.keys()

    sampler = VistaSequenceSampler(
        shape_meta=shape_meta,
        replay_buffer=_Buf(),
        rgb_keys=[],
        lowdim_keys=[],
        audio_keys=["mic_0"],
        key_horizon={"mic_0": 17, "action": 16},
        key_latency_steps={"mic_0": 0, "action": 0},
        key_down_sample_steps={"mic_0": 1, "action": 3},
        action_padding=True,
    )
    seq = sampler.sample_sequence(20)
    assert seq["mic_0"].shape == (17, 536)
    assert seq["mic_0"][0, 0] == 4.0
    assert seq["mic_0"][-1, 0] == 20.0


def test_audio_downsampled_aligned_with_n_obs():
    """Legacy strided audio path (ds>1) still works if configured."""
    shape_meta = {
        "obs": {
            "mic_0": {
                "shape": [536],
                "horizon": 2,
                "latency_steps": 0,
                "down_sample_steps": 3,
                "type": "audio",
            }
        },
        "action": {"shape": [10], "horizon": 16, "latency_steps": 0, "down_sample_steps": 3},
    }
    replay = {
        "mic_0": np.arange(20, dtype=np.float32).reshape(20, 1).repeat(536, axis=1),
        "robot0_gripper_width": np.ones((20, 1), dtype=np.float32),
        "action": np.zeros((20, 10), dtype=np.float32),
    }

    class _Buf:
        episode_ends = np.array([20], dtype=np.int64)

        def __getitem__(self, key):
            return replay[key]

        def __contains__(self, key):
            return key in replay

        def keys(self):
            return replay.keys()

    sampler = VistaSequenceSampler(
        shape_meta=shape_meta,
        replay_buffer=_Buf(),
        rgb_keys=[],
        lowdim_keys=[],
        audio_keys=["mic_0"],
        key_horizon={"mic_0": 2, "action": 16},
        key_latency_steps={"mic_0": 0, "action": 0},
        key_down_sample_steps={"mic_0": 3, "action": 3},
        action_padding=True,
    )
    # current_idx=6 → samples at 3 and 6
    seq = sampler.sample_sequence(6)
    assert seq["mic_0"].shape == (2, 536)
    assert seq["mic_0"][0, 0] == 3.0
    assert seq["mic_0"][1, 0] == 6.0


def test_vista_dataset_loads_local_zarr():
    path = "/Users/krohn/PolyUmiALL/datasets/single_episode_noisy_vista"
    if not os.path.isdir(path) and not os.path.isfile(path):
        print("skip: local vista zarr not present")
        return
    # Prefer zip if directory was listed; VistaDataset expects ZipStore path or we
    # skip when only an unzipped folder exists without zip companion.
    zip_path = path if path.endswith(".zip") else path + ".zarr.zip"
    if os.path.isdir(path) and not os.path.isfile(zip_path):
        # Try loading via directory through a temp workaround — skip if zip missing.
        print("skip: local vista export is a directory, not .zarr.zip")
        return
    from vista.data.vista_dataset import VistaDataset

    ds = VistaDataset(
        shape_meta={
            "obs": {
                "camera0_rgb": {
                    "shape": [3, 224, 224],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "rgb",
                },
                "finger_rgb": {
                    "shape": [3, 224, 224],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "rgb",
                },
                "mic_0": {
                    "shape": [536],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "audio",
                },
                "robot0_eef_pos": {
                    "shape": [3],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "low_dim",
                },
                "robot0_eef_rot_axis_angle": {
                    "raw_shape": [3],
                    "shape": [6],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "low_dim",
                    "rotation_rep": "rotation_6d",
                },
                "robot0_gripper_width": {
                    "shape": [1],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "low_dim",
                },
                "robot0_eef_rot_axis_angle_wrt_start": {
                    "raw_shape": [3],
                    "shape": [6],
                    "horizon": 2,
                    "latency_steps": 0,
                    "down_sample_steps": 3,
                    "type": "low_dim",
                },
            },
            "action": {
                "shape": [10],
                "horizon": 16,
                "latency_steps": 0,
                "down_sample_steps": 3,
                "rotation_rep": "rotation_6d",
            },
        },
        dataset_path=zip_path if os.path.isfile(zip_path) else path,
        val_ratio=0.0,
    )
    sample = ds[0]
    assert sample["action"].shape == (16, 10)
    assert sample["obs"]["camera0_rgb"].shape == (2, 3, 224, 224)
    assert sample["obs"]["finger_rgb"].shape == (2, 3, 224, 224)
    assert sample["obs"]["mic_0"].shape == (2, 536)
