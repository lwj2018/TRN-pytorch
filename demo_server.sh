python demo_action_detection_server.py pbd-v0 RGB \
/mnt/data/old/liweijie/c3d_models/trn/TRN_pbd-v0.1_RGB_BNInception_TRNmultiscale_segment3_best.pth.tar \
--arch BNInception --crop_fusion_type TRNmultiscale --test_segments 3  --threshold 20 --seq_length 10