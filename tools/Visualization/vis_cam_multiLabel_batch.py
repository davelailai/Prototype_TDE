import argparse
import copy
import math
import pkg_resources
from functools import partial
from pathlib import Path
import mmcv
import numpy as np
import torch.nn as nn
from mmcv.transforms import Compose
from mmengine.config import Config, DictAction
from mmengine.dataset import default_collate
from mmengine.utils import to_2tuple
from mmengine.utils.dl_utils import is_norm
from mmpretrain import digit_version
from mmpretrain.apis import get_model
from mmpretrain.registry import TRANSFORMS
import pickle
import pandas as pd
import os
import sys

# Add the current working directory to sys.path
current_dir = os.getcwd()
sys.path.append(current_dir)
from mmpretrain.utils import register_all_modules
from models import *
from datasets import *
from evaluation import *
# from datasets.FFA import Compose

try:
    import pytorch_grad_cam as cam
    from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
    from pytorch_grad_cam.utils.image import show_cam_on_image
except ImportError:
    raise ImportError('Please run `pip install "grad-cam>=1.3.6"` to install '
                      '3rd party package pytorch_grad_cam.')

from concurrent.futures import ThreadPoolExecutor

# Alias name
METHOD_MAP = {
    'gradcam++': cam.GradCAMPlusPlus,
}
METHOD_MAP.update({
    cam_class.__name__.lower(): cam_class
    for cam_class in cam.base_cam.BaseCAM.__subclasses__()
})


def parse_args():
    parser = argparse.ArgumentParser(description='Visualize CAM')
    parser.add_argument('--img', help='Image files (comma separated)', default='data/Endo/Endo/test_list.pkl')
    parser.add_argument('--config', help='Config file',default='work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/all_loss/fold0/Endo_CounterFactual_TDE.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='work_dirs/work_dirs_Prototype_TDE_0602/Endo/Component_Ablation/all_loss/fold0/best_multi-class_mAUC_multiclass_epoch_27.pth')
    parser.add_argument(
        '--target-layers',
        # default=['head.swav_input_proj.projection_neck'],
        default=['head.transformer.encoder'],
        nargs='+',
        type=str,
        help='The target layers to get CAM, if not set, the tool will '
        'specify the norm layer in the last block. Backbones '
        'implemented by users are recommended to manually specify'
        ' target layers in commmad statement.')
    parser.add_argument(
        '--preview-model',
        default=False,
        action='store_true',
        help='To preview all the model layers') 
    parser.add_argument(
        '--method',
        default='GradCAM',
        help='Type of method to use, supports '
        f'{", ".join(list(METHOD_MAP.keys()))}.')
    parser.add_argument(
        '--target-category',
        default=[],
        nargs='+',
        type=int,
        help='The target category to get CAM, default to use result '
        'get from given model.')
    parser.add_argument(
        '--eigen-smooth',
        default=False,
        action='store_true',
        help='Reduce noise by taking the first principle componenet of '
        '``cam_weights*activations``')
    parser.add_argument(
        '--aug-smooth',
        default=False,
        action='store_true',
        help='Whether to use test time augmentation, default not to use')
    parser.add_argument(
        '--save-path',
        type=Path,
        help='The path to save visualize cam image, default not to save.',
        default='grad_cam.png')
    parser.add_argument('--device', default='cpu', help='Device to use cpu')
    parser.add_argument(
        '--vit-like',
        action='store_true',
        default= True,
        help='Whether the network is a ViT-like network.')
    parser.add_argument(
        '--num-extra-tokens',
        type=int,
        default=1,
        help='The number of extra tokens in ViT-like backbones. Defaults to'
        ' use num_extra_tokens of the backbone.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')

    return args


def reshape_transform(tensor, model, args):
    """Build reshape_transform for `cam.activations_and_grads`, which is
    necessary for ViT-like networks."""
    if isinstance(tensor, tuple):
        tensor = tensor[0]
    if tensor.ndim == 4:
        return tensor
    elif tensor.ndim == 3:
        if not args.vit_like:
            raise ValueError(f"The tensor shape is {tensor.shape}, if it's a "
                             'vit-like backbone, please specify `--vit-like`.')
        num_extra_tokens = args.num_extra_tokens or getattr(
            model.backbone, 'num_extra_tokens', 1)
        tensor = tensor.permute(1, 0, 2)  # [N B C] -> [B, N, C]
        # tensor = tensor[:, num_extra_tokens:, :]
        heat_map_area = tensor.size()[1]
        height, width = to_2tuple(int(math.sqrt(heat_map_area)))
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.permute(0, 3, 1, 2)
        return result
    else:
        raise ValueError(f'Unsupported tensor shape {tensor.shape}.')


def init_cam(method, model, target_layers, use_cuda, reshape_transform):
    """Construct the CAM object once."""
    GradCAM_Class = METHOD_MAP[method.lower()]
    cam = GradCAM_Class(
        model=model, target_layers=target_layers)
    cam.activations_and_grads.release()
    cam.activations_and_grads = ActivationsAndGradients(
        cam.model, cam.target_layers, reshape_transform)
    return cam


def get_layer(layer_str, model):
    """Get model layer from given str."""
    for name, layer in model.named_modules():
        if name == layer_str:
            return layer
    raise AttributeError(
        f'Cannot get the layer "{layer_str}". Please choose from: \n' +
        '\n'.join(name for name, _ in model.named_modules()))


def show_cam_grad(grayscale_cam, src_img, title, out_path=None):
    """Fuse src_img and grayscale_cam and show or save."""
    grayscale_cam = grayscale_cam[0, :]
    src_img = np.float32(src_img) / 255
    visualization_img = show_cam_on_image(
        src_img, grayscale_cam, use_rgb=False)

    if out_path:
        mmcv.imwrite(visualization_img, str(out_path))
    else:
        mmcv.imshow(visualization_img, win_name=title)


def get_default_target_layers(model, args):
    """Get default target layers from given model."""
    norm_layers = [
        (name, layer)
        for name, layer in model.backbone.named_modules(prefix='backbone')
        if is_norm(layer)
    ]
    if args.vit_like:
        num_extra_tokens = args.num_extra_tokens or getattr(
            model.backbone, 'num_extra_tokens', 1)
        out_type = getattr(model.backbone, 'out_type', 'avg_featmap')
        if out_type == 'cls_token' or num_extra_tokens > 0:
            name, layer = norm_layers[-3]
            return [layer]

    name, layer = norm_layers[-1]
    return [layer]


# class debugClassifierOutputTarget:
#     def __init__(self, category):
#         self.category = category

#     def __call__(self, model_output):
#         print(model_output)
#         if len(model_output.shape) == 1:
#             return model_output[self.category]
#         return model_output[:, self.category]
    
class ClassificationWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        # out = self.model.predict(x)
        out = self.model(x)
        # 假设 out 是 dict，取 'classification' 键
        if isinstance(out, dict):
            return out['classification']
        return out

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]

class SimilarityTarget:
    def __init__(self, k, prototype):
        self.k = k
        self.prototype = prototype
    def __call__(self, model_output):
        F_k = model_output[:, self.k]
        return torch.sum(F_k * self.prototype)
    
class PrototypeRemovalTarget:
    def __init__(self, k, F_k_minus_k):
        self.k = k
        self.F_k_minus_k = F_k_minus_k
    def __call__(self, model_output):
        F_k = model_output[:, self.k]
        # f_k: 你定义的分类头
        out_with = f_k(F_k)
        out_without = f_k(self.F_k_minus_k)
        return out_with - out_without
    

    
def process_image(image_path, label, model, args, cam, target_layers):
    """Process a single image with its label."""
    cfg = Config.fromfile(args.config)
    transforms = Compose([TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])
    
    data = transforms({'img_path': image_path})
    src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
    data = model.data_preprocessor(default_collate([data]), False)
    # src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)  # Convert to HWC format
    concatenated_result = copy.deepcopy(src_img)
    src_img = np.float32(src_img) / 255

    # 获取预测概率
    input_tensor = data['inputs']
    if isinstance(input_tensor, tuple):
        input_tensor = input_tensor[0]
    if isinstance(input_tensor, torch.Tensor) and input_tensor.ndim == 5:
        input_tensor = input_tensor[0]
    if isinstance(input_tensor, tuple):
        input_tensor = input_tensor[0]
    model.head.current_epoch=100
    model.head.warmup_epoch=10
    # with torch.no_grad():
    #     pred = model.predict(input_tensor)
    #     if isinstance(pred, dict):
    #         pred = pred['classification']
    #     pred = torch.sigmoid(pred).cpu().numpy().flatten()  # 多标签sigmoid概率

    targets = None
    for i in range(model.head.num_class):
        target_category = [i]
        if target_category:
            grad_cam_v = pkg_resources.get_distribution('grad_cam').version
            if digit_version(grad_cam_v) >= digit_version('1.3.7'):
                targets = [ClassifierOutputTarget(c) for c in target_category]
            else:
                targets = args.target_category
    
        grayscale_cam = cam(
            input_tensor,
            targets,
            eigen_smooth=args.eigen_smooth,
            aug_smooth=args.aug_smooth)
        
        grayscale_cam = grayscale_cam[0, :]
        
        visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
        # # Convert grayscale CAM (1, 448, 448) to (3, 448, 448)
        # grayscale_cam_resized = grayscale_cam[0, :, :]  # (448, 448)
        # grayscale_cam_resized = np.expand_dims(grayscale_cam_resized, axis=-1)  # (448, 448, 1)
        # grayscale_cam_resized = np.repeat(grayscale_cam_resized, 3, axis=-1)  # (448, 448, 3)

         # Concatenate the image and CAM along the width (axis=1)
        concatenated_result = np.concatenate([concatenated_result, visualization_img], axis=1)
        # show_cam_grad(grayscale_cam, src_img, title=args.method, out_path=args.save_path)
        # 生成 label(pre) 字符串
    label_pre = [f"{int(l)}({p:.2f})" for l, p in zip(label, pred)]
    label_name = '_'.join(label_pre)
    # label_name = '_'.join(map(str, label.tolist())) 
    image_name = os.path.basename(image_path)
    filename = os.path.join(os.path.dirname(args.checkpoint),'vis_cam',label_name+'_'+image_name)
    mmcv.imwrite(concatenated_result, filename)


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model: nn.Module = get_model(cfg, args.checkpoint, device=args.device)

    if args.preview_model:
        print(model)
        print('\n Please remove `--preview-model` to get the CAM.')
        return

    if args.target_layers:
        target_layers = [get_layer(layer, model) for layer in args.target_layers]
    else:
        target_layers = get_default_target_layers(model, args)

    use_cuda = ('cuda' in args.device)
    model_for_cam = ClassificationWrapper(model)
    cam = init_cam(args.method, model_for_cam, target_layers, use_cuda,
                   partial(reshape_transform, model=model_for_cam, args=args))

    # Load data
    f = open(args.img, 'rb')
    data = pickle.load(f)
    image_paths = []
    labels = []
    for item in data['data_list']:
        image_path = item['img_path']
        label = item['gt_score']
        image_paths.append(image_path)
        labels.append(label)
    for image_path,label in zip(image_paths, labels):
        process_image(image_path, label, model, args, cam, target_layers)
    # image_paths =['/users/lailai/sharedscratch/openmmlab/mmpretrain/image (2).png']
    # label =[1,1,1,0,0,0]
    # Parallel processing of images using ThreadPoolExecutor
    # with ThreadPoolExecutor() as executor:
    #     for image_path, label in zip(image_paths, labels):
    #         executor.submit(process_image, image_path, label, model, args, cam, target_layers)


if __name__ == '__main__':
    main()