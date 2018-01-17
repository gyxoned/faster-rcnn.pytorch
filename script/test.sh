#!/bin/bash
CUDA_VISIBLE_DEIVICES=1 python test_net.py --dataset coco --net res101 --checksession 1 --checkepoch 6 --checkpoint 14657 --cuda --vis | tee logs/coco_res101_test.log
