## requirement

```bash
pip install mmcv-full
mmsegmentation
```

## timm cls model

### cls feature

```bash
python bgs.py \
    --src_v /dataset/mz/outside_data/vibe_data/source_v.avi \
    --bg_out /dataset/mz/outside_data/vibe_data/cls_0.05_l1.avi \
    --model hrnet_w48 \
    --feature_index 0 \
    --img_size 512 512 \
    --feature_type cls
```


## mmseg model

### download checkpoints

```bash
wget https://download.openmmlab.com/mmsegmentation/v0.5/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth
```


### seg feature

```bash
python bgs.py \
    --src_v /dataset/mz/outside_data/vibe_data/source_v2.avi \
    --bg_out /dataset/mz/outside_data/vibe_data/deeplabv3plus_decode_0.35_l1.avi \
    --seg_config configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py \
    --seg_checkpoints checkpoints/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth \
    --feature_type seg
```

调整 seg layer 特征：`feature.py` 中的 `inference_segmentor` 函数。


```bash
python bgs.py \
    --src_v /dataset/mz/outside_data/vibe_data/source_v2.avi \
    --bg_out /dataset/mz/outside_data/vibe_data/unet_backbone0_0.45_l1.avi \
    --seg_config configs/unet/deeplabv3_unet_s5-d16_256x256_40k_hrf.py \
    --seg_checkpoints checkpoints/deeplabv3_unet_s5-d16_256x256_40k_hrf_20201226_094047-3a1fdf85.pth \
    --feature_type seg
```

```bash
python bgs.py \
    --src_v /dataset/mz/outside_data/vibe_data/source_v2.avi \
    --bg_out /dataset/mz/outside_data/vibe_data/hrnet_backbone0_0.45_l1.avi \
    --seg_config configs/hrnet/fcn_hr18_512x1024_40k_cityscapes.py \
    --seg_checkpoints checkpoints/fcn_hr18_512x1024_40k_cityscapes_20200601_014216-f196fb4e.pth \
    --feature_type seg
```
