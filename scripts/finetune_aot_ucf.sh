
cfg=configs/benchmark/ucf/112x112x8-fold1-aot_base_lr1e-4_sched-5,8.yaml


CUDA_VISIBLE_DEVICES=3 python aot.py $cfg configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet --wandb_run_name diva/ssl_benchmark/rspnet
#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet

CUDA_VISIBLE_DEVICES=2 python aot.py $cfg configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco --wandb_run_name diva/ssl_benchmark/video_moco
#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco

CUDA_VISIBLE_DEVICES=1 python aot.py $cfg configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast --wandb_run_name diva/ssl_benchmark/pretext_contrast
#python finetune.py configs/benchmark/gym99/112x112x32.yaml configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast

CUDA_VISIBLE_DEVICES=0 python aot.py $cfg configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp --wandb_run_name diva/ssl_benchmark/ctp
#python finetune.py configs/benchmark/gym99/112x112x32.yaml configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

CUDA_VISIBLE_DEVICES=0 python aot.py $cfg configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt --wandb_run_name diva/ssl_benchmark/gdt
#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

CUDA_VISIBLE_DEVICES=3 python aot.py $cfg configs/main/selavi/kinetics/pretext.yaml --pretext-model selavi --wandb_run_name diva/ssl_benchmark/selavi


CUDA_VISIBLE_DEVICES=0 python aot.py $cfg configs/main/tclr/kinetics/pretext.yaml --pretext-model tclr --wandb_run_name diva/ssl_benchmark/tclr

CUDA_VISIBLE_DEVICES=1 python aot.py $cfg configs/main/avid_cma/kinetics/pretext_avid_cma.yaml --pretext-model avid_cma --wandb_run_name diva/ssl_benchmark/avid_cma

CUDA_VISIBLE_DEVICES=2 python aot.py $cfg configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp --wandb_run_name diva/ssl_benchmark/ctp_2

#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt


#python finetune.py configs/benchmark/gym99/112x112x32.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision

# ~rspnet~, ~avid~, ~ctp~ -> use different checkpoints