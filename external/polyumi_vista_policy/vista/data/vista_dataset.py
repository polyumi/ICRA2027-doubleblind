"""Vista dataset with audio and finger_rgb support."""

import copy
import os
import pathlib
import shutil
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch
import zarr
from filelock import FileLock
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.normalize_util import (
    array_to_stats,
    concatenate_normalizer,
    get_identity_normalizer_from_stat,
    get_image_identity_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pose_repr_util import convert_pose_mat_rep
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.dataset.base_dataset import BaseDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer
from umi.common.pose_util import mat_to_pose10d, pose_to_mat

from vista.data.sampler import VistaSequenceSampler, get_val_mask

register_codecs()

FINGER_RGB_SIZE = 224


def _center_crop_square_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Center-crop to square then resize to ``size`` x ``size``."""
    import cv2

    h, w = img.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    cropped = img[y0 : y0 + side, x0 : x0 + side]
    return cv2.resize(cropped, (size, size), interpolation=cv2.INTER_LINEAR)


class VistaDataset(BaseDataset):
    """UMI-style dataset with VistaSequenceSampler and multimodal obs."""

    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        cache_dir: Optional[str] = None,
        pose_repr: Optional[dict] = None,
        action_padding: bool = False,
        temporally_independent_normalization: bool = False,
        repeat_frame_prob: float = 0.0,
        seed: int = 42,
        val_ratio: float = 0.0,
        max_duration: Optional[float] = None,
    ):
        pose_repr = pose_repr or {}
        self.pose_repr = pose_repr
        self.obs_pose_repr = self.pose_repr.get("obs_pose_repr", "rel")
        self.action_pose_repr = self.pose_repr.get("action_pose_repr", "rel")

        if cache_dir is None:
            with zarr.ZipStore(dataset_path, mode="r") as zip_store:
                replay_buffer = ReplayBuffer.copy_from_store(
                    src_store=zip_store,
                    store=zarr.MemoryStore(),
                )
        else:
            mod_time = os.path.getmtime(dataset_path)
            stamp = datetime.fromtimestamp(mod_time).isoformat()
            stem_name = os.path.basename(dataset_path).split(".")[0]
            cache_name = "_".join([stem_name, stamp])
            cache_dir = pathlib.Path(os.path.expanduser(cache_dir))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir.joinpath(cache_name + ".zarr.mdb")
            lock_path = cache_dir.joinpath(cache_name + ".lock")

            print("Acquiring lock on cache.")
            with FileLock(lock_path):
                if not cache_path.exists():
                    try:
                        with zarr.LMDBStore(
                            str(cache_path),
                            writemap=True,
                            metasync=False,
                            sync=False,
                            map_async=True,
                            lock=False,
                        ) as lmdb_store:
                            with zarr.ZipStore(dataset_path, mode="r") as zip_store:
                                print(f"Copying data to {str(cache_path)}")
                                ReplayBuffer.copy_from_store(
                                    src_store=zip_store,
                                    store=lmdb_store,
                                )
                        print("Cache written to disk!")
                    except Exception as exc:
                        shutil.rmtree(cache_path)
                        raise exc

            store = zarr.LMDBStore(str(cache_path), readonly=True, lock=False)
            replay_buffer = ReplayBuffer.create_from_group(group=zarr.group(store))

        self.num_robot = 0
        rgb_keys = []
        lowdim_keys = []
        audio_keys = []
        key_horizon = {}
        key_down_sample_steps = {}
        key_latency_steps = {}
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            obs_type = attr.get("type", "low_dim")
            if obs_type == "rgb":
                rgb_keys.append(key)
            elif obs_type == "audio":
                audio_keys.append(key)
            elif obs_type == "low_dim":
                lowdim_keys.append(key)

            if key.endswith("eef_pos"):
                self.num_robot += 1

            key_horizon[key] = shape_meta["obs"][key]["horizon"]
            key_latency_steps[key] = shape_meta["obs"][key]["latency_steps"]
            key_down_sample_steps[key] = shape_meta["obs"][key]["down_sample_steps"]

        key_horizon["action"] = shape_meta["action"]["horizon"]
        key_latency_steps["action"] = shape_meta["action"]["latency_steps"]
        key_down_sample_steps["action"] = shape_meta["action"]["down_sample_steps"]

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed,
        )
        train_mask = ~val_mask

        self.sampler_lowdim_keys = []
        for key in lowdim_keys:
            if "wrt" not in key:
                self.sampler_lowdim_keys.append(key)

        for key in replay_buffer.keys():
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                self.sampler_lowdim_keys.append(key)
                query_key = key.split("_")[0] + "_eef_pos"
                key_horizon[key] = shape_meta["obs"][query_key]["horizon"]
                key_latency_steps[key] = shape_meta["obs"][query_key]["latency_steps"]
                key_down_sample_steps[key] = shape_meta["obs"][query_key]["down_sample_steps"]

        sampler = VistaSequenceSampler(
            shape_meta=shape_meta,
            replay_buffer=replay_buffer,
            rgb_keys=rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            audio_keys=audio_keys,
            key_horizon=key_horizon,
            key_latency_steps=key_latency_steps,
            key_down_sample_steps=key_down_sample_steps,
            episode_mask=train_mask,
            action_padding=action_padding,
            repeat_frame_prob=repeat_frame_prob,
            max_duration=max_duration,
        )
        self.shape_meta = shape_meta
        self.replay_buffer = replay_buffer
        self.rgb_keys = rgb_keys
        self.audio_keys = audio_keys
        self.lowdim_keys = lowdim_keys
        self.key_horizon = key_horizon
        self.key_latency_steps = key_latency_steps
        self.key_down_sample_steps = key_down_sample_steps
        self.val_mask = val_mask
        self.action_padding = action_padding
        self.repeat_frame_prob = repeat_frame_prob
        self.max_duration = max_duration
        self.sampler = sampler
        self.temporally_independent_normalization = temporally_independent_normalization
        self.threadpool_limits_is_applied = False

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = VistaSequenceSampler(
            shape_meta=self.shape_meta,
            replay_buffer=self.replay_buffer,
            rgb_keys=self.rgb_keys,
            lowdim_keys=self.sampler_lowdim_keys,
            audio_keys=self.audio_keys,
            key_horizon=self.key_horizon,
            key_latency_steps=self.key_latency_steps,
            key_down_sample_steps=self.key_down_sample_steps,
            episode_mask=self.val_mask,
            action_padding=self.action_padding,
            repeat_frame_prob=self.repeat_frame_prob,
            max_duration=self.max_duration,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()
        data_cache = {key: [] for key in self.lowdim_keys + ["action"]}
        self.sampler.ignore_rgb(True)
        dataloader = torch.utils.data.DataLoader(
            dataset=self,
            batch_size=64,
            num_workers=32,
        )
        for batch in tqdm(dataloader, desc="iterating dataset to get normalization"):
            for key in self.lowdim_keys:
                data_cache[key].append(copy.deepcopy(batch["obs"][key]))
            data_cache["action"].append(copy.deepcopy(batch["action"]))
        self.sampler.ignore_rgb(False)

        for key in data_cache.keys():
            data_cache[key] = np.concatenate(data_cache[key])
            assert data_cache[key].shape[0] == len(self.sampler)
            assert len(data_cache[key].shape) == 3
            b, t, d = data_cache[key].shape
            if not self.temporally_independent_normalization:
                data_cache[key] = data_cache[key].reshape(b * t, d)

        assert data_cache["action"].shape[-1] % self.num_robot == 0
        dim_a = data_cache["action"].shape[-1] // self.num_robot
        action_normalizers = []
        for i in range(self.num_robot):
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(data_cache["action"][..., i * dim_a : i * dim_a + 3])
                )
            )
            action_normalizers.append(
                get_identity_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][..., i * dim_a + 3 : (i + 1) * dim_a - 1]
                    )
                )
            )
            action_normalizers.append(
                get_range_normalizer_from_stat(
                    array_to_stats(
                        data_cache["action"][..., (i + 1) * dim_a - 1 : (i + 1) * dim_a]
                    )
                )
            )
        normalizer["action"] = concatenate_normalizer(action_normalizers)

        for key in self.lowdim_keys:
            stat = array_to_stats(data_cache[key])
            if key.endswith("pos") or "pos_wrt" in key:
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("pos_abs"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            elif key.endswith("rot_axis_angle") or "rot_axis_angle_wrt" in key:
                this_normalizer = get_identity_normalizer_from_stat(stat)
            elif key.endswith("gripper_width"):
                this_normalizer = get_range_normalizer_from_stat(stat)
            else:
                raise RuntimeError(f"unsupported lowdim key: {key}")
            normalizer[key] = this_normalizer

        for key in self.rgb_keys:
            normalizer[key] = get_image_identity_normalizer()
        for key in self.audio_keys:
            normalizer[key] = get_identity_normalizer_from_stat(
                array_to_stats(np.zeros((1, 1), dtype=np.float32))
            )
        return normalizer

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.threadpool_limits_is_applied:
            threadpool_limits(1)
            self.threadpool_limits_is_applied = True
        data = self.sampler.sample_sequence(idx)

        obs_dict = {}
        for key in self.rgb_keys:
            if key not in data:
                continue
            arr = data[key].astype(np.float32)
            if key == "finger_rgb":
                resized = []
                for t in range(arr.shape[0]):
                    resized.append(_center_crop_square_resize(arr[t], FINGER_RGB_SIZE))
                arr = np.stack(resized, axis=0)
            obs_dict[key] = np.moveaxis(arr, -1, 1) / 255.0
            del data[key]

        for key in self.audio_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        for key in self.sampler_lowdim_keys:
            obs_dict[key] = data[key].astype(np.float32)
            del data[key]

        for robot_id in range(self.num_robot):
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            for other_robot_id in range(self.num_robot):
                if robot_id == other_robot_id:
                    continue
                if f"robot{robot_id}_eef_pos_wrt{other_robot_id}" not in self.lowdim_keys:
                    continue
                other_pose_mat = pose_to_mat(
                    np.concatenate(
                        [
                            obs_dict[f"robot{other_robot_id}_eef_pos"],
                            obs_dict[f"robot{other_robot_id}_eef_rot_axis_angle"],
                        ],
                        axis=-1,
                    )
                )
                rel_obs_pose_mat = convert_pose_mat_rep(
                    pose_mat,
                    base_pose_mat=other_pose_mat[-1],
                    pose_rep="relative",
                    backward=False,
                )
                rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
                obs_dict[f"robot{robot_id}_eef_pos_wrt{other_robot_id}"] = rel_obs_pose[:, :3]
                obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt{other_robot_id}"] = (
                    rel_obs_pose[:, 3:]
                )

        for robot_id in range(self.num_robot):
            if (f"robot{robot_id}_eef_pos_wrt_start" not in self.shape_meta["obs"]) and (
                f"robot{robot_id}_eef_rot_axis_angle_wrt_start" not in self.shape_meta["obs"]
            ):
                continue
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            start_pose = obs_dict[f"robot{robot_id}_demo_start_pose"][0]
            start_pose += np.random.normal(
                scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
                size=start_pose.shape,
            )
            start_pose_mat = pose_to_mat(start_pose)
            rel_obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=start_pose_mat,
                pose_rep="relative",
                backward=False,
            )
            rel_obs_pose = mat_to_pose10d(rel_obs_pose_mat)
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle_wrt_start"] = rel_obs_pose[:, 3:]

        del_keys = []
        for key in obs_dict:
            if key.endswith("_demo_start_pose") or key.endswith("_demo_end_pose"):
                del_keys.append(key)
        for key in del_keys:
            del obs_dict[key]

        actions = []
        for robot_id in range(self.num_robot):
            pose_mat = pose_to_mat(
                np.concatenate(
                    [
                        obs_dict[f"robot{robot_id}_eef_pos"],
                        obs_dict[f"robot{robot_id}_eef_rot_axis_angle"],
                    ],
                    axis=-1,
                )
            )
            action_mat = pose_to_mat(data["action"][..., 7 * robot_id : 7 * robot_id + 6])
            obs_pose_mat = convert_pose_mat_rep(
                pose_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False,
            )
            action_pose_mat = convert_pose_mat_rep(
                action_mat,
                base_pose_mat=pose_mat[-1],
                pose_rep=self.obs_pose_repr,
                backward=False,
            )
            obs_pose = mat_to_pose10d(obs_pose_mat)
            action_pose = mat_to_pose10d(action_pose_mat)
            action_gripper = data["action"][..., 7 * robot_id + 6 : 7 * robot_id + 7]
            actions.append(np.concatenate([action_pose, action_gripper], axis=-1))
            obs_dict[f"robot{robot_id}_eef_pos"] = obs_pose[:, :3]
            obs_dict[f"robot{robot_id}_eef_rot_axis_angle"] = obs_pose[:, 3:]

        data["action"] = np.concatenate(actions, axis=-1)
        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
