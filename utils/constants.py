"""Defines constants"""
import socket


UCF_DATA_PATH = '/ssd/fmthoker/ucf101/UCF-101'
UCF_ANNO_PATH = '/ssd/fmthoker/ucf101/ucfTrainTestlist'

# UCF_DATA_PATH = '/local-ssd/fmthoker/ucf101/UCF-101'
# UCF_ANNO_PATH = '/local-ssd/fmthoker/ucf101/ucfTrainTestlist'

DATA_PATHS = {
    "diva": {
        "UCF": {
            "data": "/ssd/fmthoker/ucf101/UCF-101",
            "annotations": "/ssd/fmthoker/ucf101/ucfTrainTestlist",
        }
    },
    "fs4": {
        "UCF": {
            "data": "/local-ssd/fmthoker/ucf101/UCF-101",
            "annotations": "/local-ssd/fmthoker/ucf101/ucfTrainTestlist",
        }
    },
}


def get_data_paths(dataset_name):
    hostname = socket.gethostname()
    assert hostname in DATA_PATHS
    assert dataset_name in DATA_PATHS[hostname]
    return DATA_PATHS[hostname][dataset_name]
