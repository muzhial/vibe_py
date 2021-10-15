import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
import timm
from mmseg.apis import init_segmentor, show_result_pyplot
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.datasets.pipelines import Compose


PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
           [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
           [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
           [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
           [0, 80, 100], [0, 0, 230], [119, 11, 32]]


# for cls feature
def extract_class_feature(m, frame, layer_index=0,
                       img_size=(512, 512), reduction='mean',
                       top_n_feature=16, device='cpu'):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    img = np.array(img)
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(device)
    with torch.no_grad():
        o = m(img)

    resize_feat = F.interpolate(
        o[layer_index],
        size=img_size,
        mode='bilinear',
        align_corners=False)

    importance = torch.sum(resize_feat, dim=(0, 2, 3))
    sorted_importance, sorted_idx = torch.sort(
        importance, dim=0, descending=True)

    resize_feat = torch.index_select(resize_feat, 1, sorted_idx[:top_n_feature])

    if reduction == 'mean':
        l_feature = resize_feat.squeeze().mean(dim=0)
    elif reduction == 'sum':
        l_feature = resize_feat.squeeze().sum(dim=0)

    if len(l_feature.size()) < 3:
        l_feature = l_feature.unsqueeze(0)

    return l_feature


# for seg feature
class LoadImage:

    def __call__(self, results):
        results['filename'] = None
        results['ori_filename'] = None
        results['img'] = results['img']
        results['img_shape'] = results['img'].shape
        results['ori_shape'] = results['img'].shape
        return results

def inference_segmentor(model, img, feat_hierarchy='backbone:0'):
    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = [LoadImage()] + cfg.data.test.pipeline[1:]
    test_pipeline = Compose(test_pipeline)
    data = dict(img=img)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        data = scatter(data, [device])[0]
    else:
        data['img_metas'] = [i.data[0] for i in data['img_metas']]

    # cvimg = data['img'][0].detach().cpu().numpy()
    with torch.no_grad():
        if feat_hierarchy == 'logits':
            result = model(return_loss=False, rescale=True, **data)
            result = result[0]  # -> numpy
        elif feat_hierarchy.startswith('backbone'):
            feat_index = int(feat_hierarchy.split(':')[-1])
            result = model.extract_feat(data['img'][0])
            result = result[feat_index]  # -> tensor, 4 feats: 4x, 8x, 8x, 8x
        elif feat_hierarchy == 'decode':
            x = model.extract_feat(data['img'][0])
            model.decode_head.conv_seg = torch.nn.Identity()
            result = model.decode_head.forward_test(
                x, data['img_metas'], None)  # -> tensor
    return result

def get_seg_model(config_file, checkpoint_file, device='cpu'):
    model = init_segmentor(config_file, checkpoint_file, device=device)
    return model

def extract_segmentation_feature(model, frame, opacity=0.5):
    # img = frame.copy()
    # result = inference_segmentor(model, frame)
    # seg = result[0]
    # palette = np.array(PALETTE)
    # color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    # for label, color in enumerate(palette):
    #     color_seg[seg == label, :] = color
    # # convert to BGR
    # color_seg = color_seg[..., ::-1]

    # img = img * (1 - opacity) + color_seg * opacity
    # img = img.astype(np.uint8)
    # return img

    result = inference_segmentor(model, frame)
    if isinstance(result, np.ndarray):
        result = torch.from_numpy(result)
    if len(result.size()) == 4:
        result = result.squeeze()
    if len(result.size()) == 2:
        result = result.unsqueeze(0)
    return result.detach()

def visualize_feature(frame, m):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor()
    ])
    img = np.array(img)
    img = transform(img)
    plt.axis('off')
    plt.imshow(img.numpy().transpose((1, 2, 0)))
    plt.savefig('out/src.png')
    img = img.unsqueeze(0)

    with torch.no_grad():
        o = m(img)

    print('=' * 5, 'feature size', '=' * 5)
    for x in o:
        print(x.shape)

    for l in range(len(o) - 1):
        plt.figure(figsize=(30, 30))
        l_viz = o[l][0, :, :, :]
        l_viz = l_viz.detach().numpy()
        for i, ffeat in enumerate(l_viz):
            if i == 64:
                break
            plt.subplot(8, 8, i + 1)
            plt.imshow(ffeat, cmap='gray')
            plt.axis('off')
        plt.savefig(f'out/featuremap8x8_{l}.png')
        plt.close()
