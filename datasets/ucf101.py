"""UCF Dataset class copied from video-classification/ repo."""

"""Defines dataset object for UCF101"""

import os
from os.path import join
from copy import deepcopy
from numpy.random.mtrand import shuffle
from tqdm import tqdm
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Callable
import torch
from torch import Tensor

from torchvision.io import read_video_timestamps, read_video
from torchvision.datasets.video_utils import VideoClips, _VideoTimestampsDataset
from torchvision.datasets.vision import VisionDataset

from os.path import join
import numpy as np


class ClipSampler:
    """Defines a clip sampler class that samples clip(s) from given videos (paths)."""
    # def __init__(self, video_paths: list, num_clips=1, clip_len=32, frame_interval=2, fps=60) -> None:
    def __init__(self, num_clips=1, clip_len=32, frame_interval=2, fps=60, strategy="random") -> None:
        self.num_clips = num_clips
        self.clip_len = clip_len
        self.frame_interval = frame_interval
        self.fps = fps

        assert strategy in ["random", "linspace"]
        self.strategy = strategy
    

    def _sample_clip_centers(self, n_clips, video_start, video_end, strategy):

        if strategy == "random":
            clip_centers = np.random.choice(range(video_start, video_end), n_clips, replace=False)
            clip_centers = np.sort(clip_centers)
        elif strategy == "linspace":
            clip_centers = np.linspace(video_start, video_end, n_clips, dtype=int)
        
        return clip_centers

    def _get_clip(self, center_index, video_start, video_end):
        start = center_index - (self.clip_len // 2) * self.frame_interval
        end = center_index + ((self.clip_len + 1) // 2) * self.frame_interval
        frame_inds = list(range(start, end, self.frame_interval))
        frame_inds = np.clip(frame_inds, video_start, video_end - 1)
        return frame_inds

    def __call__(self, video_info: dict):
        assert {"num_frames", "start_index", "end_index"}.issubset(set(video_info.keys()))
        
        num_frames = video_info["num_frames"]
        video_start = video_info["start_index"]
        video_end = video_info["end_index"]
        clip_centers = self._sample_clip_centers(self.num_clips, video_start, video_end, self.strategy)

        clips = [self._get_clip(x, video_start, video_end) for x in clip_centers]

        return clips


class FrameSampler:
    """Defines a frame sampler class that samples frames from given videos (paths)."""
    def __init__(self, num_frames=16, strategy="random") -> None:
        self.num_frames = num_frames
        self.strategy = strategy
    
    def __call__(self, video_info: dict):
        assert {"num_frames", "start_index", "end_index"}.issubset(set(video_info.keys()))
        
        num_frames = video_info["num_frames"]
        video_start = video_info["start_index"]
        video_end = video_info["end_index"]
        if self.strategy == "random":
            frame_inds = np.random.choice(range(video_start, video_end), self.num_frames, replace=False)
        elif self.strategy == "linspace":
            frame_inds = np.linspace(video_start, video_end - 1, self.num_frames, dtype=int)
        else:
            raise ValueError("Unknown strategy: {}".format(self.strategy))
        frame_inds = np.sort(frame_inds)

        return [frame_inds]


def _collate_fn(x):
    """
    Dummy collate function to be used with _VideoTimestampsDataset
    """
    return x


class UCF101(VisionDataset):
    """
    UCF101 Dataset adapted from torchvision.dataset.ucf101.UCF101().
    """

    def __init__(
        self,
        root: str,
        video_prefix: str,
        annotation_path: str,
        num_clips_per_video=1,
        clip_len=32,
        frame_interval=2,
        debug: bool = False,
        debug_samples: int = 10,
        train_mode: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 10,
        sampler="ClipSampler",
    ) -> None:
        super(UCF101, self).__init__(root)

        extensions = ('avi',)
        self.num_workers = num_workers
        self.train_mode = train_mode

        # get data (video) samples
        self.data_samples = self.load_samples(root, annotation_path, video_prefix, debug, debug_samples)

        # define clip sampler
        clip_sampling_strategy = "random" if train_mode else "linspace"
        # clip_sampling_strategy = "random"
        if sampler == "ClipSampler":
            self.clip_sampler = ClipSampler(
                num_clips=num_clips_per_video,
                clip_len=clip_len,
                frame_interval=frame_interval,
                strategy=clip_sampling_strategy,
            )
        elif sampler == "FrameSampler":
            self.clip_sampler = FrameSampler(
                num_frames=clip_len,
                strategy=clip_sampling_strategy,
            )
        else:
            raise ValueError(f"Unknown sampler: {sampler}")

        self.transform = transform
    
    def load_samples(self, root, annotation_path, video_prefix, debug, debug_samples):
        # load path and labels from split file
        df = pd.read_csv(join(root, annotation_path))
        df["video_path"] = df["path"].apply(lambda x: join(root, video_prefix, x))

        if debug:
            df = df.sample(debug_samples)

        # add frames and FPS
        video_paths = df["video_path"].values
        timestamp_dataset = _VideoTimestampsDataset(video_paths)
        dl = torch.utils.data.DataLoader(
            timestamp_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            collate_fn=_collate_fn,
            shuffle=False,
            drop_last=False,
        )
        video_frames = []
        video_fps = []
        video_num_frames = []

        for batch in tqdm(dl, desc="Loading video samples"):
            clips, fps = list(zip(*batch))
            # we need to specify dtype=torch.long because for empty list,
            # torch.as_tensor will use torch.float as default dtype. This
            # happens when decoding fails and no pts is returned in the list.
            # clips = [torch.as_tensor(c, dtype=torch.long) for c in clips]
            num_frames = list(map(len, clips))

            video_frames.extend(clips)
            video_fps.extend(fps)
            video_num_frames.extend(num_frames)

        df["frames"] = video_frames
        df["fps"] = video_fps
        df["num_frames"] = video_num_frames
        df["start_index"] = 0
        df["end_index"] = video_num_frames

        return df.to_dict('records')

    def __len__(self) -> int:
        return len(self.data_samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, int]:

        data_sample = deepcopy(self.data_samples[idx])

        clips_indices = self.clip_sampler(data_sample)

        # label = data_sample["label"]
        label = 1

        video, _, _ = read_video(data_sample["video_path"], pts_unit="sec")
        # if not label:
        #     # reverse the video
        #     video = torch.flip(video, dims=[0])

        # sample clips
        clips = torch.cat([video[x] for x in clips_indices], dim=0)

        if self.transform is not None:
            clips = self.transform(clips)

        instance = {
            "frames": clips,
            "label": label,
            # "clips_indices": clips_indices,
        }

        data_sample.update(
            {
                "frames": clips,
                "clips_indices": clips_indices,
            }
        )

        return instance


if __name__ == "__main__":

    dataset_args = dict(
        root="/ssd/pbagad/datasets/ucf101/",
        video_prefix="videos",
        annotation_path="versions/aot_classification_test_01_v3.0.csv",
        debug=True,
        num_workers=10,
    )
    dataset = UCF101(**dataset_args)

    sample_video = dataset.data_samples[0]

    # check getitem
    instance = dataset[0]
    assert instance["frames"].shape == torch.Size([32, 240, 320, 3])
    assert instance["label"] == 1

