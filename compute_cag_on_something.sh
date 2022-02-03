# Computes the CAG score for various models on Something-Something-v2
# Usage: ./compute_cag_on_something.sh

echo "::::::::::::: Compute CAG on Something-Something-v2 :::::::::::::"

# setup
conda activate open-mmlab
export PYTHONPATH=$PWD
log_base_dir=./logs/ssv2-contrastive-action-groups/
mkdir -p $log_base_dir

echo "::::::::::::: Model: CTP :::::::::::::"
cfg=configs/benchmark/something/112x112x32.yaml
main=configs/main/ctp/kinetics/pretext.yaml
ckpt=/var/scratch/fmthoker/ssl_benchmark/checkpoints/CTP_2/Kinetics/downstream/eval-something-full_finetune_112X112x32/fold-01/model_best.pth.tar
python test_ss_cag.py $cfg $main --pretext-model ctp --ckpt $ckpt > $log_base_dir/ctp.log
tail $log_base_dir/ctp.log
