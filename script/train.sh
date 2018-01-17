#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset coco --net res101 --cuda --bs 1
# CUDA_VISIBLE_DEVICES=0 python trainval_net.py --dataset pascal_voc --net vgg16 --cuda --bs 1 --epochs 8 \
#    --start_epoch 7 --r True --checkepoch 6 --checkpoint 10021
python trainval_net.py --dataset coco --net res101 --cuda --mGPUs --bs 16 --num_workers 8 --gpu_ids 0 1 2 3 4 5 6 7 \
    --lr 0.01 --lr_decay_step 4 --epochs 7 | tee logs/coco_res101.log
