# RTMDet-m person detector (COCO + Objects365 person), 640x640 input.
# Matches models/rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth.
# Copied from mmpose 1.3.2 demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py,
# with the ImageNet backbone init removed (the checkpoint has all weights).
_base_ = 'mmdet::rtmdet/rtmdet_m_8xb32-300e_coco.py'

model = dict(
    backbone=dict(init_cfg=None),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))
train_dataloader = dict(dataset=dict(metainfo=dict(classes=('person', ))))
val_dataloader = dict(dataset=dict(metainfo=dict(classes=('person', ))))
test_dataloader = val_dataloader
