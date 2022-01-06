"""Defines dataset object for UCF101"""

import os
from os.path import join
from copy import deepcopy
import numpy as np
from tqdm import tqdm
import pandas as pd
from typing import Any, Dict, List, Tuple, Optional, Callable
import torch
from torch import Tensor

from torchvision.io import read_video
from torchvision.datasets.video_utils import _VideoTimestampsDataset
from torchvision.datasets.vision import VisionDataset


def _collate_fn(x):
    """
    Dummy collate function to be used with _VideoTimestampsDataset
    """
    return x


def random_sample_clip_indices(num_frames_total, num_frames_to_sample, new_length=1):
    
    average_duration = (num_frames_total - new_length + 1) // num_frames_to_sample
    if average_duration > 0:
        offsets = np.multiply(list(range(num_frames_to_sample)), average_duration) + \
            np.random.randint(average_duration, size=num_frames_to_sample)
    else:
        tick = (num_frames_total - new_length + 1) / float(num_frames_to_sample)
        tick = 0 if tick < 0 else tick
        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_frames_to_sample)])
    
    # NOTE: no need to reorder offsets since we flip order of images during training
    # offsets = self._reorder_offsets(offsets, record)

    # return offsets + 1
    return offsets


def deterministic_sample_clip_indices(num_frames_total, num_frames_to_sample, new_length=1):
    tick = (num_frames_total - new_length + 1) / float(num_frames_to_sample)
    tick = 0 if tick < 0 else tick
    offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_frames_to_sample)])

    # return offsets + 1
    return offsets


class UCF101Debug(VisionDataset):
    """
    UCF101Debug Dataset adapted from torchvision.dataset.UCF101.UCF101().
    """

    def __init__(
        self,
        root: str,
        video_prefix: str,
        annotation_path: str,
        clip_len=32,
        debug: bool = False,
        debug_samples: int = 10,
        train_mode: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 10,
        retain_label=False,
    ) -> None:
        super(UCF101Debug, self).__init__(root)

        extensions = ('avi',)
        self.num_workers = num_workers
        self.train_mode = train_mode
        self.retain_label = retain_label
        self.clip_len = clip_len

        # get data (video) samples
        self.data_samples = self.load_samples(root, annotation_path, video_prefix, debug, debug_samples)
        
        # get class labels
        classes = open(join(root, "annotations/classInd.txt")).read().strip().split("\n")
        self.classes = {x.split(" ")[-1]: int(x.split(" ")[0])  - 1 for x in classes}
        
        # define clip sampler
        self.clip_sampler = random_sample_clip_indices if self.train_mode else deterministic_sample_clip_indices

        # define transforms
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

    def __getitem__(self, idx: int, return_indices=False) -> Tuple[Tensor, Tensor, int]:

        data_sample = deepcopy(self.data_samples[idx])

        # clips_indices = self.clip_sampler(data_sample)
        clip_indices = self.clip_sampler(
            num_frames_total=data_sample["num_frames"],
            num_frames_to_sample=self.clip_len,
        )

        # we only sample a single clip per video
        clips_indices = [clip_indices]

        # label = data_sample["label"]
        label = 1
        if self.retain_label:
            label = self.classes[data_sample["path"].split("/")[0]]

        video, _, _ = read_video(data_sample["video_path"], pts_unit="sec")

        # sample clips
        clips = torch.cat([video[x] for x in clips_indices], dim=0)

        if self.transform is not None:
            clips = self.transform(clips)

        instance = {
            # "frames": clips,
            "frames": clips,
            "label": label,
        }
        if return_indices:
            instance.update({"clips_indices": clips_indices})

        data_sample.update(
            {
                "frames": clips,
                "clips_indices": clips_indices,
            }
        )

        return instance


if __name__ == "__main__":
    
    args = dict(
        root="/ssd/pbagad/datasets/ucf101/",
        video_prefix="videos",
        clip_len=8,
        debug=True,
        debug_samples=100,
        annotation_path="versions/aot_classification_val_01_v1.0.csv",
    )

    # check with train_mode=True
    print("check with train_mode=True")
    np.random.seed(0)
    dataset = UCF101Debug(train_mode=True, **args)

    # check getitem
    sample_video = dataset.data_samples[0]
    instance = dataset.__getitem__(0, return_indices=True)
    assert instance["frames"].shape == torch.Size([args["clip_len"], 240, 320, 3])
    assert instance["label"] == 1
    print("Total number of frames: ", sample_video["num_frames"])
    print(instance["clips_indices"])

    instance = dataset.__getitem__(0, return_indices=True)
    print(instance["clips_indices"])

    instance = dataset.__getitem__(0, return_indices=True)
    print(instance["clips_indices"])

    # check with train_mode=False
    print("check with train_mode=False")
    np.random.seed(0)
    dataset = UCF101Debug(train_mode=False, **args)

    # check getitem
    sample_video = dataset.data_samples[0]
    instance = dataset.__getitem__(0, return_indices=True)
    assert instance["frames"].shape == torch.Size([args["clip_len"], 240, 320, 3])
    assert instance["label"] == 1
    print("Total number of frames: ", sample_video["num_frames"])
    print(instance["clips_indices"])

    instance = dataset.__getitem__(0, return_indices=True)
    print(instance["clips_indices"])

    instance = dataset.__getitem__(0, return_indices=True)
    print(instance["clips_indices"])
    
    # check with relevant transforms
    print("check with relevant transforms")
    from pytorchvideo.transforms import (
        Normalize,
        Permute,
        ShortSideScale,
    )
    from torchvision.transforms import (
        Compose,
        Lambda,
        CenterCrop,
    )
    # For ResNet3D (torchvision): https://pytorch.org/vision/stable/models.html#video-classification
    _MEAN = [0.43216, 0.394666, 0.37645]
    _STD = [0.22803, 0.22145, 0.216989]
    _CROP_SIZE = 112

    transform = Compose(
        [
            Lambda(lambda x: x / 255.0),
            Permute((3, 0, 1, 2)),
            Normalize(_MEAN, _STD),
            ShortSideScale(_CROP_SIZE),
            CenterCrop(_CROP_SIZE),
        ]
    )
    np.random.seed(0)
    dataset = UCF101Debug(train_mode=False, transform=transform, **args)
    sample_video = dataset.data_samples[0]
    instance = dataset.__getitem__(0, return_indices=True)
    assert instance["frames"].shape == torch.Size([3, args["clip_len"], 112, 112])
    assert instance["label"] == 1