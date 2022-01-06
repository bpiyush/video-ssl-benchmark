 #CUDA_VISIBLE_DEVICES=0,1,2,3 python eval-action-recg_rsp_something.py configs/benchmark/something/something.yaml configs/main/rspnet/kinetics/pretext.yaml 

 #CUDA_VISIBLE_DEVICES=0,1,2,3 python eval-action-recg-something-something.py configs/benchmark/something/112x112x32.yaml configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco

 #CUDA_VISIBLE_DEVICES=0,1,2,3 python eval-action-recg-something-something.py configs/benchmark/something/112x112x32.yaml configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

 #CUDA_VISIBLE_DEVICES=0,1,2,3 python eval-action-recg-something-something.py configs/benchmark/something/112x112x32_scratch.yaml configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

 #CUDA_VISIBLE_DEVICES=0,1,2,3 python eval-action-recg-ucf.py  configs/benchmark/something/112x112x16.yaml configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

python eval-action-recg-ucf.py configs/benchmark/something/112x112x16.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision
