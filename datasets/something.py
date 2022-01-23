# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from datasets.video_db import VideoDataset
import os
from pathlib import Path
from typing import Dict
import json
import pandas as pd
import numpy as np

# DATA_PATH = '/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-videos_avi'
# ANNO_PATH = '/ssdstore/fmthoker/20bn-something-something-v2/something-something-v2-annotations/'

#DATA_PATH = '/home/fthoker/20bn-something-something-v2/something-something-v2-videos_avi'
#ANNO_PATH = '/home/fthoker/20bn-something-something-v2/something-something-v2-annotations/'

DATA_PATH = '/local-ssd/fmthoker/20bn-something-something-v2/something-something-v2-videos_avi'
ANNO_PATH = '/local-ssd/fmthoker/20bn-something-something-v2/something-something-v2-annotations/'

FINE_TO_COARSE_URLS = {
    10: "https://raw.githubusercontent.com/willprice/20bn-something-something-label-hierarchies/master/fine_to_10_classes.csv",
}

GRANULARITIES = {
    "coarse": 10,
    "fine": 174,
    "coarse_plus_fine": 40, # 10 coarse + 30 fine
}


def read_class_idx(annotation_dir: Path) -> Dict[str, str]:
    class_ind_path = annotation_dir+'/something-something-v2-labels.json'
    with open(class_ind_path) as f:
        class_dict = json.load(f)
    return class_dict


def filter_examples_based_on_granularity(filenames, labels, fine_to_coarse_mapping):
    indices = [idx for idx, l in enumerate(labels) if l in fine_to_coarse_mapping]
    labels = np.array(labels)[indices]
    filenames = np.array(filenames)[indices]
    labels = [fine_to_coarse_mapping[l] for l in labels]
    filenames = list(filenames)
    
    return filenames, labels


class SOMETHING(VideoDataset):
    def __init__(self, subset,
                 granularity: str = "fine",
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
        self.name = 'SOMETHING'
        self.root = DATA_PATH
        self.subset = subset
        self.class_idx_dict = read_class_idx(ANNO_PATH)

        assert granularity in set(GRANULARITIES.keys())
        self.granularity = granularity

        filenames = []
        labels = []
        if 'train' in subset:
               video_list_path = f'{ANNO_PATH}/something-something-v2-train.json' 
               with open(video_list_path) as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(self.class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                            filenames.append(video_name)
                            labels.append(class_index)

        else:
               video_list_path = f'{ANNO_PATH}/something-something-v2-validation.json' 
               with open(video_list_path) as f:
                   video_infos = json.load(f)
                   for video_info in video_infos:
                       video = int(video_info['id'])
                       video_name = f'{video}.avi'
                       class_name = video_info['template'].replace('[', '').replace(']', '')
                       class_index = int(self.class_idx_dict[class_name])
                       if os.path.isfile(DATA_PATH+'/'+video_name):
                             filenames.append(video_name)
                             labels.append(class_index)

        # print(filenames[0:10],labels[0:10])

        if self.granularity == 'coarse':
            self.num_classes = GRANULARITIES["coarse"]
            df = pd.read_csv(FINE_TO_COARSE_URLS[GRANULARITIES["coarse"]])
            fine_to_coarse = dict(df[["fine_grained_class_index", "coarse_class_index"]].values)

            filenames, labels = filter_examples_based_on_granularity(filenames, labels, fine_to_coarse)

        elif self.granularity == "coarse_plus_fine":
            self.num_classes = GRANULARITIES["coarse_plus_fine"]

            # first obtain coarse labels
            df = pd.read_csv(FINE_TO_COARSE_URLS[GRANULARITIES["coarse"]])
            fine_to_coarse = dict(df[["fine_grained_class_index", "coarse_class_index"]].values)

            # select certain fine labels to ve added to coarse labels
            URL = "https://raw.githubusercontent.com/willprice/20bn-something-something-label-hierarchies/master/40_class_subset.csv"
            df = pd.read_csv(URL)
            selected_fine_df = df[df.coarse_grained == False]
            fine_to_selected_fine = dict(selected_fine_df[["fine_grained_class_index", "class_index"]].values)

            given_coarse_df = df[df.coarse_grained == True]
            given_coarse_class_to_index = dict(given_coarse_df[["class_name", "class_index"]].values)

            URL = "https://raw.githubusercontent.com/willprice/20bn-something-something-label-hierarchies/master/10_class_subset.csv"
            original_coarse_df = pd.read_csv(URL)
            original_coarse_class_to_index = dict(original_coarse_df[["class_name", "class_index"]].values)
            coarse_index_original_to_given = {original_coarse_class_to_index[k]: given_coarse_class_to_index[k] for k in original_coarse_class_to_index}
            fine_to_coarse = {k: coarse_index_original_to_given[v] for k, v in fine_to_coarse.items()}

            fine_to_coarse_plus_fine = {**fine_to_coarse, **fine_to_selected_fine}
            filenames, labels = filter_examples_based_on_granularity(filenames, labels, fine_to_coarse_plus_fine)

        self.num_videos = len(filenames)
        self.num_classes = len(set(labels))

        print("")
        print(f"Using granularity: {self.granularity}")
        print("Number of classes:", self.num_classes)
        print("Number of videos:", self.num_videos)
        print("")

        super(SOMETHING, self).__init__(
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


if __name__ == "__main__":
    dataset = SOMETHING('test')
    dataset = SOMETHING('test', granularity='coarse')
    dataset = SOMETHING('test', granularity='coarse_plus_fine')
