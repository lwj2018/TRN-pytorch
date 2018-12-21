CUDA_VISIBLE_DEVICES=0 python pretrain.py ucf101 RGB --arch BNInception --num_segments 3 --consensus_type TRNmultiscale --batch-size 64
