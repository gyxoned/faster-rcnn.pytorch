#!/bin/bash
CUDA_VISIBLE_DEIVICES=1 python test_net.py --dataset coco --net res50 --checksession 1 --checkepoch 6 --checkpoint 14657 --cuda | tee logs/coco_res50_test.log
