# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets.video_db import VideoDataset
from utils.constants import UCF_ANNO_PATH as ANNO_PATH
from utils.constants import UCF_DATA_PATH as DATA_PATH

# DATA_PATH = '/ssdstore/fmthoker/ucf101/UCF-101'
# ANNO_PATH = '/ssdstore/fmthoker/ucf101/ucfTrainTestlist'
#DATA_PATH = '/home/fthoker/ucf101/UCF-101'
#ANNO_PATH = '/home/fthoker/ucf101/ucfTrainTestlist'


class UCF(VideoDataset):
    def __init__(self, subset,
                 video_clip_duration=0.5,
                 return_video=True,
                 video_fps=16.,
                 video_transform=None,
                 return_audio=False,
                 return_labels=False,
                 max_offsync_augm=0,
                 mode='clip',
                 clips_per_video=20,
                 frame_sampling="random",
                 ):

        assert return_audio is False
        self.name = 'UCF-101'
        self.root = DATA_PATH
        self.subset = subset

        classes_fn = f'{ANNO_PATH}/classInd.txt'
        self.classes = [l.strip().split()[1] for l in open(classes_fn)]

        filenames = [ln.strip().split()[0] for ln in open(f'{ANNO_PATH}/{subset}.txt')]
        labels = [fn.split('/')[0] for fn in filenames]
        labels = [self.classes.index(cls) for cls in labels]

        self.num_classes = len(self.classes)
        self.num_videos = len(filenames)

        super(UCF, self).__init__(
            return_video=return_video,
            video_root=DATA_PATH,
            video_clip_duration=video_clip_duration,
            video_fns=filenames,
            video_fps=video_fps,
            video_transform=video_transform,
            return_audio=False,
            return_labels=return_labels,
            labels=labels,
            max_offsync_augm=max_offsync_augm,
            mode=mode,
            clips_per_video=clips_per_video,
            frame_sampling=frame_sampling,
        )

if __name__ == '__main__':
    import numpy as np
    import torch
    from pytorchvideo.transforms import (
        Normalize,
        Permute,
        RandomShortSideScale,
        ShortSideScale,
    )
    from torchvision.transforms import (
        Compose,
        Lambda,
        RandomCrop,
        RandomHorizontalFlip,
        CenterCrop,
    )
    from utils.videotransforms import volume_transforms

    _MEAN = [0.43216, 0.394666, 0.37645]
    _STD = [0.22803, 0.22145, 0.216989]
    _CROP_SIZE = 112

    video_transform = Compose(
        [   
            volume_transforms.ClipToTensor(),
            Normalize(_MEAN, _STD),
            ShortSideScale(_CROP_SIZE),
            CenterCrop(_CROP_SIZE),
        ]
    )

    db = UCF(
        subset="trainlist01",
        return_video=True,
        video_clip_duration=1,
        video_fps=16,
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        clips_per_video=1,
        mode="frames",
        frame_sampling="random",
    )
    sample = db[0]
    print(sample['frames'].shape)


    print("::::::::: Testing: Frame sampling ::::::::::::")
    print("::: Test frame sampling: random")
    np.random.seed(0)
    db = UCF(
        subset="trainlist01",
        return_video=True,
        video_fps=8, # when mode is "frames", this acts as number of frames to be sampled
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode="frames",
        frame_sampling="random",
    )
    sample = db.__getitem__(0, return_frame_indices=True)
    assert sample["frames"].shape == torch.Size([3, 8, 112, 112])
    assert sample["label"] == 0
    print("Frame indices", sample["frame_indices"])

    print("::: Test frame sampling: linspace")
    np.random.seed(0)
    db = UCF(
        subset="trainlist01",
        return_video=True,
        video_fps=8, # when mode is "frames", this acts as number of frames to be sampled
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode="frames",
        frame_sampling="linspace",
    )
    sample = db.__getitem__(0, return_frame_indices=True)
    assert sample["frames"].shape == torch.Size([3, 8, 112, 112])
    assert sample["label"] == 0
    print("Frame indices", sample["frame_indices"])

    print("::: Test frame sampling: linspace_with_offset")
    np.random.seed(0)
    db = UCF(
        subset="trainlist01",
        return_video=True,
        video_fps=8, # when mode is "frames", this acts as number of frames to be sampled
        video_transform=video_transform,
        return_audio=False,
        return_labels=True,
        mode="frames",
        frame_sampling="linspace_with_offset",
    )
    sample = db.__getitem__(0, return_frame_indices=True)
    assert sample["frames"].shape == torch.Size([3, 8, 112, 112])
    assert sample["label"] == 0
    print("Frame indices", sample["frame_indices"])

    sample = db.__getitem__(10, return_frame_indices=True)
    print("Frame indices", sample["frame_indices"])

    sample = db.__getitem__(100, return_frame_indices=True)
    print("Frame indices", sample["frame_indices"])
