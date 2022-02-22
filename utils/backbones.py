"""Helper functions for loading/working with backbones for various VSSL methods."""

import re
import os

import torch
from torchsummary import summary

import torchvision.models.video as video_models

from utils.misc import print_update


def _check_inputs(backbone, init_method, ckpt_path):
    """Checks inputs for load_backbone()."""
    assert backbone in video_models.__dict__, f"{backbone} is not a valid backbone."
    assert init_method in [
        "scratch",
        "supervised",
        "CTP",
        "GDT",
        "RSPNet",
    ]
    
    if init_method in ["scratch", "supervised"]:
        assert ckpt_path is None, f"{init_method} cannot be initialized from a checkpoint."\
            f"Pass ckpt_path=None while using init_method={init_method}."
    else:
        assert ckpt_path is not None, f"{init_method} must be initialized from a checkpoint."
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"{ckpt_path} does not exist.")



def load_ctp_checkpoint(ckpt_path, verbose=False):
    """
    Loads CTP checkpoint.

    Args:
        ckpt_path (str): path to checkpoint
        verbose (bool, optional): whether to print out the loaded checkpoint. Defaults to False.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"{ckpt_path} does not exist.")
    
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = checkpoint["state_dict"]
    
    # preprocess the state dict to make it compatible with R2+1D backbone
    csd = {k.replace("backbone.", ""):v for k,v in csd.items()}
    mapping = {
        "stem.conv_s": "stem.0",
        "stem.bn_s": "stem.1",
        "stem.conv_t": "stem.3",
        "stem.bn_t": "stem.4",
        "layer\\d{1}.\\d{1}.conv1.conv_s": "layer\\d{1}.\\d{1}.conv1.0.0",
        "layer\\d{1}.\\d{1}.conv1.bn_s": "layer\\d{1}.\\d{1}.conv1.0.1",
        "layer\\d{1}.\\d{1}.conv1.relu_s": "layer\\d{1}.\\d{1}.conv1.0.2",
        "layer\\d{1}.\\d{1}.conv1.conv_t": "layer\\d{1}.\\d{1}.conv1.0.3",
        "layer\\d{1}.\\d{1}.bn1": "layer\\d{1}.\\d{1}.conv1.1",
        "layer\\d{1}.\\d{1}.conv2.conv_s": "layer\\d{1}.\\d{1}.conv2.0.0",
        "layer\\d{1}.\\d{1}.conv2.bn_s": "layer\\d{1}.\\d{1}.conv2.0.1",
        "layer\\d{1}.\\d{1}.conv2.relu_s": "layer\\d{1}.\\d{1}.conv2.0.2",
        "layer\\d{1}.\\d{1}.conv2.conv_t": "layer\\d{1}.\\d{1}.conv2.0.3",
        "layer\\d{1}.\\d{1}.bn2": "layer\\d{1}.\\d{1}.conv2.1",
        "layer\\d{1}.\\d{1}.downsample": "layer\\d{1}.\\d{1}.downsample.0",
        "layer\\d{1}.\\d{1}.downsample_bn": "layer\\d{1}.\\d{1}.downsample.1",
        "layer\\d{1}.\\d{1}.downsample.conv": "layer\\d{1}.\\d{1}.downsample.0",
        "layer\\d{1}.\\d{1}.downsample_bn": "layer\\d{1}.\\d{1}.downsample.1",
    }
    
    # obtain mapping from checkpoint keys to backbone keys
    csd_keys_to_bsd_keys = dict()
    for k in csd.keys():
        
        for x in mapping:
            pattern = re.compile(x)
            if pattern.match(k):
                if x.startswith("layer"):
                    ori = ".".join((x.split(".")[2:]))
                    new = ".".join((mapping[x].split(".")[2:]))
                    replaced = k.replace(ori, new)
                else:
                    ori = x
                    new = mapping[x]
                    replaced = k.replace(ori, new)
                    
                disp = "\t\t".join([k, ori, new, replaced])
                if verbose:
                    print(disp)

                csd_keys_to_bsd_keys[k] = replaced

    # construct a new state dict that is fully compatible with R2+1D backbone
    new_csd = dict()
    for k,v in csd.items():
        if k in csd_keys_to_bsd_keys:
            new_csd[csd_keys_to_bsd_keys[k]] = csd[k]
        else:
            new_csd[k] = csd[k]

    return new_csd


def load_gdt_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter out audio network related keys
    csd = {k:v for k,v in csd.items() if not k.startswith(("audio_network", "mlp_a"))}
    
    # define mapping from csd keys to backbone keys
    mapping = lambda x: x.replace("video_network.base.", "")
    
    # construct a new state dict that is fully compatible with R2+1D backbone
    new_csd = {mapping(k):v for k,v in csd.items()}
    
    return new_csd


def load_rspnet_checkpoint(ckpt_path, verbose=False):
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    csd = ckpt["model"]
    
    # filter out encoder_k related keys
    # csd = {k:v for k,v in csd.items() if not k.startswith(("encoder_k.", "mlp_a"))}
    
    prefix = 'encoder_q.'
    blacklist = ['fc.', 'linear', 'head', 'new_fc', 'fc8', 'encoder_fuse']

    def filter(k):
        return k.startswith(prefix) and not any(k.startswith(f'{prefix}{fc}') for fc in blacklist)

    new_csd = {k[len(prefix):]: v for k, v in csd.items() if filter(k)}
    
    # remove prefixes
    new_csd = {k.replace("encoder.base_network.", "layer"): v for k, v in new_csd.items()}
    # replace layer0 with stem
    new_csd = {k.replace("layer0.", "stem."): v for k, v in new_csd.items()}
    
    return new_csd


def load_backbone(backbone="r2plus1d_18", init_method="scratch", ckpt_path=None):
    """
    Loads given backbone (e.g. R2+1D from `torchvision.models`) with weights
    initialized from given VSSL method checkpoint.

    Args:
        backbone (str, optional): Backbone from `torchvision.models`.
            Defaults to "r2plus1d_18".
        init_method (str, optional): VSSL methods from which to initialize weights.
            Defaults to "scratch".
        ckpt_path ([str, None], optional): path to checkpoint for the given VSSL method.
            Defaults to None.
    """

    _check_inputs(backbone, init_method, ckpt_path)
    
    backbone = getattr(video_models, backbone)(pretrained=(init_method == "supervised"))
    message = f"Checkpoint path not needed for {init_method} backbone."
    
    if init_method not in ["scratch", "supervised"]:
        state_dict = eval(f"load_{init_method.lower()}_checkpoint")(ckpt_path)
        message = backbone.load_state_dict(state_dict, strict=False)
        
    print_update(f"Loaded {init_method} checkpoint")
    print("Path: {}".format(ckpt_path))
    print("Message: {}".format(message))

    return backbone


if __name__ == "__main__":
    
    # setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = load_backbone("r2plus1d_18", "scratch")

    # Print summary
    # summary(model.to(device), (3, 16, 112, 112))
    
    # test RSPNet
    model = load_backbone(
        "r2plus1d_18",
        "RSPNet",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/rspnet/snellius_checkpoint_epoch_200.pth.tar",
    )
    
    # test GDT
    model = load_backbone(
        "r2plus1d_18",
        "GDT",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/gdt/gdt_K400.pth",
    )
    
    # test CTP
    model = load_backbone(
        "r2plus1d_18",
        "CTP",
        ckpt_path="/home/pbagad/models/checkpoints_pretraining/ctp/snellius_r2p1d18_ctp_k400_epoch_90.pth",
    )