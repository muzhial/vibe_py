import numpy as np
import cv2
import torch
import timm

from feature import (
    get_seg_model, extract_class_feature,
    extract_segmentation_feature, visualize_feature)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def main():
    feature_type = 'seg'  # [seg|cls]
    source_video = "/dataset/mz/outside_data/vibe_data/source_v.avi"
    out_video = '/dataset/mz/outside_data/vibe_data/seg_out.avi'

    vc = cv2.VideoCapture(source_video)
    fw = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    f_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print(f'frame size: (w: {fw}, h: {fh})')
    print(f'frame rate: {fps}')
    print(f'frame count: {f_num}')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(
        out_video,
        fourcc, fps, (512, 512))
    c = 0
    if not vc.isOpened():
        raise RuntimeError('video read init error')

    if feature_type == 'cls':
        m = timm.create_model(
            'hrnet_w48',
            features_only=True,
            pretrained=True)
    elif feature_type == 'seg':
        m = get_seg_model(
            'configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py',
            'checkpoints/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth',
            'cuda:0'
        )

    rval = True
    while rval:
        rval, frame = vc.read()
        if not rval:
            break

        if feature_type == 'cls':
            proc_feat = extract_class_feature(frame, m)
        elif feature_type == 'seg':
            proc_feat = extract_segmentation_feature(m, frame)

        # proc_feat = proc_feat * 255
        # proc_feat = proc_feat.astype(np.uint8)
        # proc_feat = cv2.cvtColor(proc_feat, cv2.COLOR_GRAY2RGB)
        # out.write(proc_feat)

        cv2.imwrite(f'out/frame_{c}.png', proc_feat)

        c = c + 1
        print(f'===> {c} / {f_num}')
        if c >= 1:
            break

    vc.release()
    out.release()
    cv2.destroyAllWindows()

def test_seg_feature():
    img_file = '/dataset/mz/segmentation/cityscapes/leftImg8bit/test/munich/munich_000265_000019_leftImg8bit.png'
    seg_config = 'configs/deeplabv3plus/deeplabv3plus_r18-d8_512x1024_80k_cityscapes.py'
    checkpoint = 'checkpoints/deeplabv3plus_r18-d8_512x1024_80k_cityscapes_20201226_080942-cff257fe.pth'
    model = get_seg_model(seg_config, checkpoint, device)
    frame = cv2.imread(img_file)
    result = extract_segmentation_feature(model, frame)
    result = result.cpu().numpy()
    print(result.shape)


if __name__ == '__main__':
    # main()
    test_seg_feature()
