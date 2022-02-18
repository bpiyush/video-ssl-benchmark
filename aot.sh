cfg=configs/benchmark/ucf/112x112x32-fold1-aot.yaml
cfg=configs/benchmark/ucf/112x112x8-fold1-aot_base_lr1e-4_sched-5,8.yaml
main=configs/main/full_supervision/kinetics/pretext.yaml

python aot.py $cfg $main --wandb_run_name $1 --pretext-model full_supervision

echo ":::: Running AoT on UCF with MoCo ::::"
cfg=main=configs/main/full_supervision/kinetics/pretext.yaml
main=configs/main/moco/kinetics/pretext.yaml
python aot.py $cfg $main --pretext-model moco --no_wandb

# no wandb
# python aot.py $cfg $main --pretext-model full_supervision --no_wandb

# with wandb
# python aot.py $cfg $main --pretext-model full_supervision --wandb_run_name $1