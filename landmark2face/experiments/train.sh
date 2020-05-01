#!/usr/bin/env bash

~/anaconda3/bin/python3 train.py --name man1_Res9 \
                                --gpu_ids 0 \
                                --batch_size 12 \
                                --img_size 256 \
                                --model l2face \
                                --dataset_mode l2face \
                                --netG resnet_9blocks_l2face
