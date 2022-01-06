
#python eval-action-recg-ucf.py configs/benchmark/gym99/112x112x16.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt

#python eval-action-recg-ucf.py configs/benchmark/gym99/112x112x16.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco
#
#python eval-action-recg-ucf.py configs/benchmark/gym99/112x112x16.yaml configs/main/pretext_contrast/kinetics/pretext.yaml --pretext-model pretext_contrast

#python eval-action-recg-ucf.py configs/benchmark/gym99/112x112x32.yaml  configs/main/gdt/kinetics/pretext.yaml --pretext-model gdt


python eval-action-recg-ucf.py configs/benchmark/ntu60/112x112x16-fold1.yaml  configs/main/video_moco/kinetics/pretext.yaml --pretext-model video_moco
