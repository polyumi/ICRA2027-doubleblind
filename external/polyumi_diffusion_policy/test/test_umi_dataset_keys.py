"""
Unit test for get_load_keys — which store keys UmiDataset copies into memory.

One .zarr.zip (a `pingest export --type polyumi` one) feeds both the visuomotor config and a
multimodal one; the extra streams must cost nothing for the config that ignores them.
Run inside the container's umi env:
    python -m pytest test/test_umi_dataset_keys.py -q
"""

# ruff: noqa: D103  - test functions are self-describing via names + inline comments

import numpy as np
import zarr

from diffusion_policy.dataset.umi_dataset import get_load_keys

# What a --type polyumi export puts in data/, mic_0 and finger_rgb included.
STORE_KEYS = [
    'camera0_rgb', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width',
    'robot0_demo_start_pose', 'robot0_demo_end_pose', 'mic_0', 'finger_rgb',
]

VISUOMOTOR_OBS = [
    'camera0_rgb', 'robot0_eef_pos', 'robot0_eef_rot_axis_angle', 'robot0_gripper_width',
    'robot0_eef_rot_axis_angle_wrt_start',  # derived at load time, never in the store
]


def write_store(path):
    with zarr.ZipStore(str(path), mode='w') as store:
        root = zarr.group(store=store)
        root.create_group('meta').array('episode_ends', np.array([4], dtype=np.int64))
        data = root.create_group('data')
        for key in STORE_KEYS:
            data.array(key, np.zeros((4, 2), dtype=np.float32))


def test_extra_modalities_are_not_loaded(tmp_path):
    path = tmp_path / 'ds.zarr.zip'
    write_store(path)

    shape_meta = {'obs': {key: {} for key in VISUOMOTOR_OBS}}
    keys = get_load_keys(shape_meta, str(path))

    # the demo poses stay (UmiDataset derives _wrt_start from them), the wrt key never existed,
    # and the two PolyUMI streams this config does not sample are dropped.
    assert set(keys) == set(STORE_KEYS) - {'mic_0', 'finger_rgb'}


def test_multimodal_config_loads_them(tmp_path):
    path = tmp_path / 'ds.zarr.zip'
    write_store(path)

    shape_meta = {'obs': {key: {} for key in VISUOMOTOR_OBS + ['mic_0', 'finger_rgb']}}
    keys = get_load_keys(shape_meta, str(path))

    assert set(keys) == set(STORE_KEYS)


if __name__ == '__main__':
    import pathlib
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_extra_modalities_are_not_loaded(pathlib.Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_multimodal_config_loads_them(pathlib.Path(d))
    print('ok')
