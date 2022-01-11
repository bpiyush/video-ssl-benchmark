
cfg=configs/benchmark/ucf/112x112x16-fold1-video_retrieval_base_lr1e-4.yaml


echo "Running video retrieval with scratch"
python video_retrieval.py $cfg configs/main/full_supervision/kinetics/pretext_scratch.yaml --pretext-model full_supervision --wandb_run_name pretrain_full_supervision --no_wandb

echo "Running video retrieval with full_supervision"
python video_retrieval.py $cfg configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision --wandb_run_name pretrain_full_supervision --no_wandb

echo "Running video retrieval with rspnet"
python video_retrieval.py $cfg configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet --wandb_run_name pretrain_rspnet --no_wandb

echo "Running video retrieval with video_moco"
python video_retrieval.py $cfg configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco --wandb_run_name pretrain_video_moco --no_wandb

echo "Running video retrieval with pretext_contrast"
python video_retrieval.py $cfg configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast --wandb_run_name pretrain_pretext_contrast --no_wandb

echo "Running video retrieval with gdt"
python video_retrieval.py $cfg configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt --wandb_run_name pretrain_gdt --no_wandb

echo "Running video retrieval with selavi"
python video_retrieval.py $cfg configs/main/selavi/kinetics/pretext.yaml --pretext-model selavi --wandb_run_name pretrain_selavi --no_wandb

echo "Running video retrieval with tclr"
python video_retrieval.py $cfg configs/main/tclr/kinetics/pretext.yaml --pretext-model tclr --wandb_run_name pretrain_tclr --no_wandb

echo "Running video retrieval with avid_cma"
python video_retrieval.py $cfg configs/main/avid_cma/kinetics/pretext_avid_cma.yaml --pretext-model avid_cma --wandb_run_name pretrain_avid_cma --no_wandb

echo "Running video retrieval with ctp"
python video_retrieval.py $cfg configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp --wandb_run_name pretrain_ctp_2 --no_wandb
