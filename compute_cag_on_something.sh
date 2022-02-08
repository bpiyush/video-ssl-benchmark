# Computes the CAG score for various models on Something-Something-v2
# Usage: ./compute_cag_on_something.sh

echo "::::::::::::: Compute CAG on Something-Something-v2 :::::::::::::"

# setup
conda activate open-mmlab
export PYTHONPATH=$PWD
log_base_dir=./logs/ssv2-contrastive-action-groups/
mkdir -p $log_base_dir

cfg=configs/benchmark/something/112x112x32.yaml

# echo "::::::::::::: Model: CTP :::::::::::::"
# main=configs/main/ctp/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/checkpoints/CTP_2/Kinetics/downstream/eval-something-full_finetune_112X112x32/fold-01/model_best.pth.tar
# python test_ss_cag.py $cfg $main --pretext-model ctp --ckpt $ckpt > $log_base_dir/ctp.log
# tail $log_base_dir/ctp.log

# echo "::::::::::::: Model: RSPNet :::::::::::::"
# main=configs/main/rspnet/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/checkpoints/RSPNET_2/Kinetics/downstream/eval-something-full_finetune_112X112x32/fold-01/model_best.pth.tar
# python test_ss_cag.py $cfg $main --pretext-model rspnet --ckpt $ckpt > $log_base_dir/rspnet.log
# tail $log_base_dir/rspnet.log


# echo "::::::::::::: Model: TCLR :::::::::::::"
# main=configs/main/tclr/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/checkpoints/TCLR/Kinetics/downstream/eval-something-full_finetune_112X112x32/fold-01/model_best.pth.tar
# python test_ss_cag.py $cfg $main --pretext-model tclr --ckpt $ckpt > $log_base_dir/tclr.log
# tail $log_base_dir/tclr.log

echo "::::::::::::: Model: Full supervision :::::::::::::"
main=configs/main/full_supervision/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/common_sense_checkpoints/full_supervision/model_best.pth.tar
ckpt=/home/pbagad/models/common_sense_checkpoints/full_supervision/model_best.pth.tar
python test_ss_cag.py $cfg $main --pretext-model full_supervision --ckpt $ckpt > $log_base_dir/full_supervision.log
tail $log_base_dir/full_supervision.log

echo "::::::::::::: Model: Scratch :::::::::::::"
main=configs/main/full_supervision/kinetics/pretext_scratch.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/common_sense_checkpoints/from_scratch/model_best.pth.tar
ckpt=/home/pbagad/models/common_sense_checkpoints/from_scratch/model_best.pth.tar
python test_ss_cag.py $cfg $main --pretext-model from_scratch --ckpt $ckpt > $log_base_dir/from_scratch.log
tail $log_base_dir/from_scratch.log

echo "::::::::::::: Model: SELAVI :::::::::::::"
main=configs/main/selavi/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/common_sense_checkpoints/selavi/model_best.pth.tar
ckpt=/home/pbagad/models/common_sense_checkpoints/selavi/model_best.pth.tar
python test_ss_cag.py $cfg $main --pretext-model selavi --ckpt $ckpt > $log_base_dir/selavi.log
tail $log_base_dir/selavi.log

echo "::::::::::::: Model: VIDOEMOCO :::::::::::::"
main=configs/main/video_moco/kinetics/pretext.yaml
# ckpt=/var/scratch/fmthoker/ssl_benchmark/common_sense_checkpoints/video_moco/model_best.pth.tar
ckpt=/home/pbagad/models/common_sense_checkpoints/video_moco/model_best.pth.tar
python test_ss_cag.py $cfg $main --pretext-model video_moco --ckpt $ckpt > $log_base_dir/video_moco.log
tail $log_base_dir/video_moco.log
