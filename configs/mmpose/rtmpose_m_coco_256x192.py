# RTMPose-m, 256x192, COCO-17 keypoints.
# Matches models/rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth.
# The bundled mmpose 1.3.2 config, with the ImageNet backbone init removed
# (the checkpoint has all weights). Phase 7 fine-tunes from this.
_base_ = 'mmpose::body_2d_keypoint/rtmpose/coco/rtmpose-m_8xb256-420e_coco-256x192.py'

model = dict(backbone=dict(init_cfg=None))
