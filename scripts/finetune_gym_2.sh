#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet

#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco
#
#python finetune.py configs/benchmark/gym99/112x112x32.yaml configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast

#python finetune.py configs/benchmark/gym99/112x112x32.yaml configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision


#######################################################

python finetune.py configs/benchmark/gym_event/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision
