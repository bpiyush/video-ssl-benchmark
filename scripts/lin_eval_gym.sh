
#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet

#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco
#
CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast
