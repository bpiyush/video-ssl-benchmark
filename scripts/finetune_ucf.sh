
#python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/ctp/kinetics/pretext.yaml --pretext-model ctp

#python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

#python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast


#python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco
#
#python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/rspnet/kinetics/pretext.yaml --pretext-model rspnet

python eval-action-recg-ucf.py configs/benchmark/ucf/112x112x32-fold1.yaml  configs/main/full_supervision/kinetics/pretext.yaml --pretext-model full_supervision
