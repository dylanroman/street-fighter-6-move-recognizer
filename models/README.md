# models/

Model weights live here and are gitignored (everything except this README).
Get the pretrained ones with:

```bash
python scripts/download_weights.py
```

That script checks each file against a pinned SHA-256.

| File | What | Phase |
|---|---|---|
| `rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth` | RTMDet-m person detector (COCO + Objects365), OpenMMLab | 2 |
| `rtmpose-m_simcc-coco_pt-aic-coco_420e-256x192-d8dd5ca4_20230127.pth` | RTMPose-m 256x192, COCO-17, OpenMMLab | 2 |
| `rtmpose-m_sf6.pth` | our RTMPose-m fine-tune on SF6 frames | 7 |
| `stgcnpp_<character>.pth` | our ST-GCN++ move classifiers | 5 |
