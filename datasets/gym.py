# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets.video_db import VideoDataset
import os

DATA_PATH = '/local-ssd/fmthoker/gym/subactions'
ANNO_PATH = '/local-ssd/fmthoker/gym/annotations'


class GYM99(VideoDataset):
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
                 ):

        assert return_audio is False
        self.name = 'GYM-99'
        self.root = DATA_PATH
        self.subset = subset

        filenames = []
        labels = []
        if 'train' in subset:

               for ln in open(f'{ANNO_PATH}/gym99_train.txt'):
                      file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                      if os.path.isfile(DATA_PATH+'/'+file_name):
                              #print(file_name,label)
                              filenames.append(file_name)
                              labels.append(label)

        else:
               for ln in open(f'{ANNO_PATH}/gym99_val.txt'):
                      file_name, label = ln.strip().split()[0][0:-3]+'avi',int(ln.strip().split()[1])
                      if os.path.isfile(DATA_PATH+'/'+file_name):
                              filenames.append(file_name)
                              labels.append(label)
        print(filenames[0:10],labels[0:10])

        #filenames = filenames[0:1000]
        #labels   =  labels[0:1000]
        self.num_videos = len(filenames)

        super(GYM99, self).__init__(
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
        )

