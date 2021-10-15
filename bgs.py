import argparse
import time

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torchvision import transforms
import timm

from feature import (
    extract_class_feature, visualize_feature,
    get_seg_model, extract_segmentation_feature)
from vibe_pt import ViBe


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--src_v',
        type=str,
        default=None,
        help='source video')
    parser.add_argument(
        '--feat_out',
        type=str,
        default=None,
        help='output path')
    parser.add_argument('--feature_index', type=int, default=0)
    # parser.add_argument(
    #     '--feat_video_size',
    #     nargs='+',
    #     type=int,
    #     default=(512, 512),
    #     help='output video frame size')
    parser.add_argument(
        '--img_size',
        nargs='+',
        type=int,
        default=(512, 512))
    # parser.add_argument(
    #     '--bg_video_size',
    #     nargs='+',
    #     type=int,
    #     default=(512, 256),
    #     help='bg model output video size')
    parser.add_argument(
        '--bg_out',
        type=str,
        default=None,
        help='bg model output video'
    )
    parser.add_argument('--model', default='hrnet_w48')
    parser.add_argument('--feature_type', type=str, default='cls')

    parser.add_argument(
        '--seg_config',
        type=str)
    parser.add_argument(
        '--seg_checkpoints',
        type=str)

    parser.add_argument('--device', default=None)

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    vc = cv2.VideoCapture(args.src_v)
    fw = vc.get(cv2.CAP_PROP_FRAME_WIDTH)
    fh = vc.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = vc.get(cv2.CAP_PROP_FPS)
    f_num = vc.get(cv2.CAP_PROP_FRAME_COUNT)
    print('=' * 5, 'source video info', '=' * 5)
    print(f'frame size: (w: {fw}, h: {fh})')
    print(f'frame rate: {fps}')
    print(f'frame count: {f_num}\n')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    if not vc.isOpened():
        raise RuntimeError('video read init error')
    else:
        rval, frame = vc.read()

    if args.feature_type == 'cls':
        m = timm.create_model(
            args.model,
            features_only=True,
            pretrained=True)
        m.to(device)
        feat_result = extract_class_feature(
            m, frame, args.feature_index, args.img_size, device=device)
    elif args.feature_type == 'seg':
        m = get_seg_model(args.seg_config, args.seg_checkpoints, device)
        feat_result = extract_segmentation_feature(m, frame)

    assert isinstance(feat_result, torch.Tensor
        ) and len(feat_result.size()) == 3, \
            'feat_result must be <torch.Tensor> type and 3-dim'

    height, width = feat_result.shape[1:]

    # feature writer
    if args.feat_out is not None:
        print(f'feat out: ({height}, {width}), {args.feat_out}')
        feat_out = cv2.VideoWriter(
            args.feat_out, fourcc, fps, (width, height))

    # bg model result writer
    if args.bg_out is not None:
        print(f'bg out: ({height}, {width}), {args.bg_out}')
        bg_out = cv2.VideoWriter(
            args.bg_out, fourcc, fps, (width, height))

    # ViBe init
    vibe = ViBe(
        num_sam=50,
        min_match=2,
        radiu=0.45,
        rand_sam=4,
        dist_type='l1',
        device=device)
    vibe.ProcessFirstFrame(feat_result)

    print(f'extract feature ...')
    pbar = tqdm(total=int(f_num - 1))
    while rval:
        rval, frame = vc.read()
        if not rval:
            break

        # visualize feature map
        # if count == 0:
        #     visualize_feature(frame, m)
        if args.feature_type == 'cls':
            feat_result = extract_class_feature(
                m, frame, args.feature_index, args.img_size, device=device)
            # proc_feat = proc_feat * 255
            # proc_feat = proc_feat.astype(np.uint8)
            # proc_feat = cv2.cvtColor(proc_feat, cv2.COLOR_GRAY2RGB)
        elif args.feature_type == 'seg':
            feat_result = extract_segmentation_feature(m, frame)
        vibe.Update(feat_result)
        seg_mat = vibe.getFGMask()
        seg_mat = seg_mat.detach().cpu().numpy().astype(np.uint8)

        if args.feat_out is not None:
            feat_out.write(cv2.cvtColor(seg_mat, cv2.COLOR_GRAY2BGR))
        if args.bg_out is not None:
            bg_out.write(cv2.cvtColor(seg_mat, cv2.COLOR_GRAY2BGR))
        pbar.update(1)
    pbar.close()

    # torch.cuda.empty_cache()
    vc.release()
    if args.feat_out is not None:
        feat_out.release()
    if args.bg_out is not None:
        bg_out.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
