CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

#CUDA_VISIBLE_DEVICES=0,1,2,3 python linear_eval.py  configs/benchmark/gym99/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision

