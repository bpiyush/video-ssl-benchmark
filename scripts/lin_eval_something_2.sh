
#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/something/112x112x32.yaml configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco

#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/something/112x112x32.yaml configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/something/112x112x32.yaml configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet
