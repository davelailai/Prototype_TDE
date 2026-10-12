import argparse
import copy
import math
import pkg_resources
from functools import partial
from pathlib import Path
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
import json

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
    parser.add_argument('--img', help='Image files (comma separated)', default='data/OIA-ODIR/onsite_test.pkl')
    parser.add_argument('--config', help='Config file',default='work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Component_Ablation/baseline/fold0/ODIR_CounterFactual_TDE.py')
    parser.add_argument('--checkpoint', help='Checkpoint file',default='work_dirs/work_dirs_Prototype_TDE_0602/ODIR/Component_Ablation/baseline/fold0/best_multi-class_mAUC_multiclass_epoch_44.pth')
    parser.add_argument(
        '--target-layers',
        # default=['head.task_heads.classification.input_proj'],
        default=['head.task_heads.classification.transformer.encoder'],
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
    parser.add_argument(
        '--cam-align-top-ratios',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3],
        help='Top activation ratios used to binarize CAMs for Dice/IoU alignment.')
    args = parser.parse_args()
    if args.method.lower() not in METHOD_MAP.keys():
        raise ValueError(f'invalid CAM type {args.method},'
                         f' supports {", ".join(list(METHOD_MAP.keys()))}.')
    if any(not 0 < ratio < 1 for ratio in args.cam_align_top_ratios):
        raise ValueError('--cam-align-top-ratios values must be in (0, 1).')

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


def normalize_cam(cam_map):
    cam_map = np.asarray(cam_map, dtype=np.float32)
    min_val = np.nanmin(cam_map)
    max_val = np.nanmax(cam_map)
    return (cam_map - min_val) / (max_val - min_val + 1e-8)


def top_ratio_mask(cam_map, top_ratio=0.2):
    cam_map = normalize_cam(cam_map)
    threshold = np.quantile(cam_map, 1.0 - top_ratio)
    return cam_map >= threshold


def pairwise_cam_alignment(cam_a, cam_b, top_ratio=0.2):
    cam_a = normalize_cam(cam_a)
    cam_b = normalize_cam(cam_b)
    mask_a = top_ratio_mask(cam_a, top_ratio)
    mask_b = top_ratio_mask(cam_b, top_ratio)
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    dice_den = mask_a.sum() + mask_b.sum()
    dice = (2.0 * intersection / dice_den) if dice_den > 0 else np.nan
    iou = (intersection / union) if union > 0 else np.nan

    flat_a = cam_a.reshape(-1)
    flat_b = cam_b.reshape(-1)
    pearson = np.corrcoef(flat_a, flat_b)[0, 1] if np.std(flat_a) > 0 and np.std(flat_b) > 0 else np.nan
    cosine = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-8)
    return {
        'dice': float(dice),
        'iou': float(iou),
        'pearson': float(pearson),
        'cosine': float(cosine),
    }


def pairwise_cam_continuous_alignment(cam_a, cam_b):
    cam_a = normalize_cam(cam_a)
    cam_b = normalize_cam(cam_b)
    flat_a = cam_a.reshape(-1)
    flat_b = cam_b.reshape(-1)
    pearson = np.corrcoef(flat_a, flat_b)[0, 1] if np.std(flat_a) > 0 and np.std(flat_b) > 0 else np.nan
    cosine = np.dot(flat_a, flat_b) / (np.linalg.norm(flat_a) * np.linalg.norm(flat_b) + 1e-8)
    return {
        'pearson': float(pearson),
        'cosine': float(cosine),
    }


def compute_three_cam_alignment(cam_feature, cam_similarity, cam_removal, top_ratios=(0.1, 0.2, 0.3)):
    metrics = {}

    cls_proto_cont = pairwise_cam_continuous_alignment(cam_feature, cam_similarity)
    cls_tde_cont = pairwise_cam_continuous_alignment(cam_feature, cam_removal)
    proto_tde_cont = pairwise_cam_continuous_alignment(cam_similarity, cam_removal)
    for metric_name in ('pearson', 'cosine'):
        metrics[f'cls_proto_{metric_name}'] = cls_proto_cont[metric_name]
        metrics[f'cls_tde_{metric_name}'] = cls_tde_cont[metric_name]
        metrics[f'proto_tde_{metric_name}'] = proto_tde_cont[metric_name]
        metrics[f'camalign_{metric_name}'] = float(np.nanmean([
            cls_proto_cont[metric_name],
            cls_tde_cont[metric_name],
            proto_tde_cont[metric_name],
        ]))

    for top_ratio in top_ratios:
        ratio_tag = int(round(top_ratio * 100))
        cls_proto = pairwise_cam_alignment(cam_feature, cam_similarity, top_ratio)
        cls_tde = pairwise_cam_alignment(cam_feature, cam_removal, top_ratio)
        proto_tde = pairwise_cam_alignment(cam_similarity, cam_removal, top_ratio)
        for metric_name in ('dice', 'iou'):
            metric_tag = f'{metric_name}@{ratio_tag}'
            metrics[f'cls_proto_{metric_tag}'] = cls_proto[metric_name]
            metrics[f'cls_tde_{metric_tag}'] = cls_tde[metric_name]
            metrics[f'proto_tde_{metric_tag}'] = proto_tde[metric_name]
            metrics[f'camalign_{metric_tag}'] = float(np.nanmean([
                cls_proto[metric_name],
                cls_tde[metric_name],
                proto_tde[metric_name],
            ]))
    return metrics


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
 ###### -----------For Q2L-TDE-----------------##########   
class ClassificationWrapper(torch.nn.Module):
    def __init__(self, model, target='PrototypeRemoval'):
        super().__init__()
        self.model = model
        self.target= target
    def forward(self, x):
        # out = self.model.predict(x)
        out = self.model(x)
        # 假设 out 是 dict，取 'classification' 键
        if isinstance(out, dict):
            out= out['classification']
        classifier = self.model.head.task_heads['classification'].classifier
        if self.target == 'Feature': # 只返回当前结果判断的依据
            cls_score=classifier(out[0])
            return cls_score
        if self.target == 'Feature&Prototype': #返回原型和当前特征的相似度
            feature = out[0]
            prototype = out[2]
            similarity = F.cosine_similarity(feature, prototype, dim=-1)
            return similarity
        if self.target == 'PrototypeRemoval': # 返回移除原型后的结果变化
            feature = out[0]
            prototype = out[2]
            intervened_feats_sub = self.model.head.task_heads['classification'].intervention_combiner(feature, prototype, direction="sub")
            # return intervened_feats_sub.squeeze()
            # f_k: 你定义的分类头
            # classifier = self.model.head.task_heads['classification'].classifier
            out_with = classifier(feature)
            out_without = classifier(intervened_feats_sub)
            return out_with - out_without
###########----------For Q2L-------------##########    
# class ClassificationWrapper(torch.nn.Module):
#     def __init__(self, model, target='PrototypeRemoval'):
#         super().__init__()
#         self.model = model
#         self.target= target
#     def forward(self, x):
#         # out = self.model.predict(x)
#         out = self.model(x)
#         # 假设 out 是 dict，取 'classification' 键
#         if isinstance(out, dict):
#             out= out['classification']
#         classifier = self.model.head.task_heads['classification'].fc
#         if self.target == 'Feature': # 只返回当前结果判断的依据
#             cls_score=classifier(out)
#             return cls_score
#         if self.target == 'Feature&Prototype': #返回原型和当前特征的相似度
#             feature = out
#             prototype = self.model.head.task_heads['classification'].get_prototypes().unsqueeze(0)
#             similarity = F.cosine_similarity(feature, prototype, dim=-1)
#             return similarity
#         if self.target == 'PrototypeRemoval': # 返回移除原型后的结果变化
#             feature = out
#             prototype = self.model.head.task_heads['classification'].get_prototypes().unsqueeze(0)
#             intervened_feats_sub = self.model.head.task_heads['classification'].intervention_combiner(feature, prototype, direction="sub")
#             # return intervened_feats_sub.squeeze()
#             # f_k: 你定义的分类头
#             # classifier = self.model.head.task_heads['classification'].classifier
#             out_with = classifier(feature)
#             out_without = classifier(intervened_feats_sub)
#             return out_with - out_without



        # return out
    # def SimilarityTarget(self, k, prototype):
        

class ClassifierOutputTarget:
    def __init__(self, category):
        self.category = category

    def __call__(self, model_output):
        if len(model_output.shape) == 1:
            return model_output[self.category]
        return model_output[:, self.category]


    
def process_image(input_tensor, src_img, model, args, cam):
    """Process a single image with its label."""
    # cfg = Config.fromfile(args.config)
    # transforms = Compose([TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])
    
    # data = transforms({'img_path': image_path})
    # src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
    # data = model.data_preprocessor(default_collate([data]), False)
    # # src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)  # Convert to HWC format
    # concatenated_result = copy.deepcopy(src_img)
    # src_img = np.float32(src_img) / 255

    # # 获取预测概率
    # input_tensor = data['inputs']
    # if isinstance(input_tensor, tuple):
    #     input_tensor = input_tensor[0]
    # if isinstance(input_tensor, torch.Tensor) and input_tensor.ndim == 5:
    #     input_tensor = input_tensor[0]
    # if isinstance(input_tensor, tuple):
    #     input_tensor = input_tensor[0]
    # model.head.current_epoch=100
    # model.head.warmup_epoch=10
    # with torch.no_grad():
    #     pred = model.predict(input_tensor)
    #     pred=pred[0].pred_score
    #     # if isinstance(pred, dict):
    #     #     pred = pred['classification']
    #     # pred = torch.sigmoid(pred).cpu().numpy().flatten()  # 多标签sigmoid概率
    concatenated_result = copy.deepcopy(src_img)
    src_img = np.float32(src_img) / 255
    cam_maps = []
    
    targets = None
    for i in range(model.head.task_heads['classification'].num_classes):
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
        cam_maps.append(grayscale_cam.copy())
        
        visualization_img = show_cam_on_image(src_img, grayscale_cam, use_rgb=False)
        # # Convert grayscale CAM (1, 448, 448) to (3, 448, 448)
        # grayscale_cam_resized = grayscale_cam[0, :, :]  # (448, 448)
        # grayscale_cam_resized = np.expand_dims(grayscale_cam_resized, axis=-1)  # (448, 448, 1)
        # grayscale_cam_resized = np.repeat(grayscale_cam_resized, 3, axis=-1)  # (448, 448, 3)

         # Concatenate the image and CAM along the width (axis=1)
        concatenated_result = np.concatenate([concatenated_result, visualization_img], axis=1)
        # show_cam_grad(grayscale_cam, src_img, title=args.method, out_path=args.save_path)
        # 生成 label(pre) 字符串
    return concatenated_result, cam_maps


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
     # 三种target wrapper
    model_for_cam_feature = ClassificationWrapper(model, target='Feature')
    model_for_cam_similarity = ClassificationWrapper(model, target='Feature&Prototype')
    model_for_cam_removal = ClassificationWrapper(model, target='PrototypeRemoval')

    cam_feature = init_cam(args.method, model_for_cam_feature, target_layers, ('cuda' in args.device), partial(reshape_transform, model=model_for_cam_feature, args=args))
    cam_similarity = init_cam(args.method, model_for_cam_similarity, target_layers, ('cuda' in args.device), partial(reshape_transform, model=model_for_cam_similarity, args=args))
    cam_removal = init_cam(args.method, model_for_cam_removal, target_layers, ('cuda' in args.device), partial(reshape_transform, model=model_for_cam_removal, args=args))
    # model_for_cam = ClassificationWrapper(model)
    # cam = init_cam(args.method, model_for_cam, target_layers, use_cuda,
    #                partial(reshape_transform, model=model_for_cam, args=args))

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
    vis_dir = os.path.join(os.path.dirname(args.checkpoint), 'vis_cam')
    os.makedirs(vis_dir, exist_ok=True)
    alignment_rows = []
    for image_path,label in zip(image_paths, labels):
        cfg = Config.fromfile(args.config)
        transforms = Compose([TRANSFORMS.build(t) for t in cfg.test_dataloader.dataset.pipeline])
        
        data = transforms({'img_path': image_path})
        src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)
        data = model.data_preprocessor(default_collate([data]), False)
        # src_img = copy.deepcopy(data['inputs']).numpy().transpose(1, 2, 0)  # Convert to HWC format
        
        # src_img = np.float32(src_img) / 255

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
        with torch.no_grad():
            pred = model.predict(input_tensor)
            pred=pred[0].pred_score
            pred_scores = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else np.asarray(pred)


            
        grayscale_cam_feature, cam_maps_feature = process_image(input_tensor, src_img, model, args, cam_feature)
        grayscale_cam_similarity, cam_maps_similarity = process_image(input_tensor, src_img, model, args, cam_similarity)
        grayscale_cam_removal, cam_maps_removal = process_image(input_tensor, src_img, model, args, cam_removal)
        concatenated_result = np.concatenate([
                grayscale_cam_feature,
                grayscale_cam_similarity,
                grayscale_cam_removal
            ], axis=0)


        label_pre = [f"{int(l)}({p:.2f})" for l, p in zip(label, pred_scores)]
        label_name = '_'.join(label_pre)
        # label_name = '_'.join(map(str, label.tolist())) 
        image_name = os.path.basename(image_path)
        filename = os.path.join(vis_dir, label_name+'_'+image_name)
        mmcv.imwrite(concatenated_result, filename)

        num_classes = model.head.task_heads['classification'].num_classes
        for class_idx in range(num_classes):
            if int(label[class_idx]) != 1:
                continue
            metrics = compute_three_cam_alignment(
                cam_maps_feature[class_idx],
                cam_maps_similarity[class_idx],
                cam_maps_removal[class_idx],
                top_ratios=args.cam_align_top_ratios)
            alignment_rows.append({
                'image_path': image_path,
                'image_name': image_name,
                'class_idx': class_idx,
                'gt': int(label[class_idx]),
                'pred_score': float(pred_scores[class_idx]),
                'cam_align_top_ratios': ','.join(str(r) for r in args.cam_align_top_ratios),
                **metrics,
            })

    if alignment_rows:
        alignment_df = pd.DataFrame(alignment_rows)
        per_class_path = os.path.join(vis_dir, 'cam_alignment_positive_per_class.csv')
        summary_path = os.path.join(vis_dir, 'cam_alignment_positive_summary.csv')
        json_path = os.path.join(vis_dir, 'cam_alignment_positive_summary.json')

        alignment_df.to_csv(per_class_path, index=False)
        metric_cols = [col for col in alignment_df.columns if col.startswith(
            ('cls_proto_', 'cls_tde_', 'proto_tde_', 'camalign_'))]
        summary_df = alignment_df[metric_cols].agg(['mean', 'std', 'count']).T.reset_index()
        summary_df.columns = ['metric', 'mean', 'std', 'count']
        summary_df.to_csv(summary_path, index=False)

        class_summary_df = alignment_df.groupby('class_idx')[metric_cols].mean().reset_index()
        summary_payload = {
            'top_ratios': args.cam_align_top_ratios,
            'num_positive_rows': int(len(alignment_df)),
            'overall': summary_df.to_dict(orient='records'),
            'per_class_mean': class_summary_df.to_dict(orient='records'),
        }
        with open(json_path, 'w') as f:
            json.dump(summary_payload, f, indent=2)

        print(f'Positive-class CAM alignment per-class results saved to: {per_class_path}')
        print(f'Positive-class CAM alignment summary saved to: {summary_path}')
        print(f'Positive-class CAM alignment JSON summary saved to: {json_path}')
    # image_paths =['/users/lailai/sharedscratch/openmmlab/mmpretrain/image (2).png']
    # label =[1,1,1,0,0,0]
    # Parallel processing of images using ThreadPoolExecutor
    # with ThreadPoolExecutor() as executor:
    #     for image_path, label in zip(image_paths, labels):
    #         executor.submit(process_image, image_path, label, model, args, cam, target_layers)


if __name__ == '__main__':
    main()
