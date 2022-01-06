cfg=configs/benchmark/ucf/112x112x32-fold1-aot.yaml
main=configs/main/full_supervision/kinetics/pretext.yaml

python aot.py $cfg $main --wandb_run_name $1 --pretext-model full_supervision


# no wandb
# python aot.py $cfg $main --pretext-model full_supervision --no_wandb

# with wandb
# python aot.py $cfg $main --pretext-model full_supervision --wandb_run_name $1