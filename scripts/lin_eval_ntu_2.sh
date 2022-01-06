
#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/ntu60/112x112x16-fold1.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/ntu60/112x112x16-fold1.yaml  configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast
