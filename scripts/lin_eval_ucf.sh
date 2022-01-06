

CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/ucf/112x112x32-fold1-linear.yaml  configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/ucf/112x112x32-fold1-linear.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision

#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/ntu60/112x112x32-fold1-linear.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision
