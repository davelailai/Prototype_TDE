"""Patient-split annotations and image transforms shared by SOTA adapters."""
import pickle
import random
import copy
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ANNOTATIONS = {
    'FFA': ('FFA/train_class6.pkl', 'FFA/val_class6.pkl', 'FFA/test_class6.pkl'),
    'Endo': tuple('Endo/Endo/patient_disjoint_balanced_seed42/' + p + '_list.pkl' for p in ('train', 'val', 'test')),
    'ODIR': ('OIA-ODIR/train.pkl', 'OIA-ODIR/val.pkl', 'OIA-ODIR/onsite_test.pkl'),
}


def resolve_image(path, image_root):
    """An explicit image root takes precedence over stale absolute paths."""
    path = Path(path)
    root = Path(image_root).resolve()
    parts = path.parts
    if 'data' in parts:
        relative = Path(*parts[len(parts) - 1 - parts[::-1].index('data') + 1:])
    elif path.is_absolute():
        raise ValueError(f'Cannot relocate absolute path without a data component: {path}')
    else:
        relative = path
    candidate = root / relative
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def read_annotations(path):
    # Input PKLs must be trusted local annotation files, never arbitrary uploads.
    with open(path, 'rb') as handle:
        payload = pickle.load(handle)
    rows = payload['data_list']
    classes = list(payload['metainfo']['classes'])
    if not rows or not classes:
        raise ValueError(f'Empty annotation file: {path}')
    y = np.asarray([r['gt_score'] for r in rows], dtype=np.float32)
    if y.shape != (len(rows), len(classes)) or not np.isin(y, [0, 1]).all():
        raise ValueError(f'Expected complete binary labels in {path}, got {y.shape}')
    return classes, rows


class ReferenceView:
    """Use the same registered image operations as the reference MM pipeline."""
    def __init__(self, operations, size):
        from mmpretrain.utils import register_all_modules
        from mmpretrain.registry import TRANSFORMS
        register_all_modules(init_default_scope=False)
        self.operations = []
        for operation in copy.deepcopy(operations):
            if operation['type'] == 'Resize':
                operation['scale'] = round(size * operation['scale'] / 448)
            if operation['type'] in ('RandomResizedCrop', 'CenterCrop', 'RandomCrop'):
                key = 'scale' if operation['type'] == 'RandomResizedCrop' else 'crop_size'
                operation[key] = size
            self.operations.append(TRANSFORMS.build(operation))

    def __call__(self, image):
        array = np.asarray(image)[:, :, ::-1].copy()
        data = dict(img=array, img_shape=array.shape[:2], ori_shape=array.shape[:2])
        for operation in self.operations:
            data = operation(data)
        return Image.fromarray(data['img'][:, :, ::-1].copy())


class Images(Dataset):
    def __init__(self, annotation, image_root, size=448, train=False, method='ProCI', grid=2, reference=None):
        self.classes, self.rows = read_annotations(annotation)
        self.paths = [str(resolve_image(r['img_path'], image_root)) for r in self.rows]
        self.method, self.train, self.grid, self.size = method, train, grid, size
        normalize = T.Normalize([.48145466, .4578275, .40821073], [.26862954, .26130258, .27577711])
        self.to_tensor = T.Compose([T.ToTensor(), normalize])
        self.geometry = T.Compose([
            T.Resize(round(size * 512 / 448)),
            T.RandomResizedCrop(size, scale=(.9, 1), ratio=(1, 1)),
            T.RandomHorizontalFlip(),
        ]) if train else T.Compose([T.Resize(round(size * 512 / 448)), T.CenterCrop(size)])
        self.augment = T.Compose([T.RandomGrayscale(.2), T.RandomApply([T.GaussianBlur(5, (.1, 1))], p=.2)])
        if reference is not None:
            normalization = reference['normalization']
            self.to_tensor = T.Compose([T.ToTensor(), T.Normalize(
                [x/255 for x in normalization['mean']], [x/255 for x in normalization['std']])])
            if train:
                photometric = ('RandomGrayscale', 'GaussianBlur')
                self.geometry = ReferenceView([op for op in reference['train_view'] if op['type'] not in photometric], size)
                self.augment = ReferenceView([op for op in reference['train_view'] if op['type'] in photometric], size)
            else:
                self.geometry = ReferenceView(reference['eval_view'], size)

    def __len__(self):
        return len(self.rows)

    def _image(self, index):
        with Image.open(self.paths[index]) as handle:
            return handle.convert('RGB')

    def __getitem__(self, index):
        image = self.geometry(self._image(index))
        label = torch.tensor(self.rows[index]['gt_score'], dtype=torch.float32)
        if self.method == 'MLC-PAT':
            # Patch-major ordering is produced by default_collate, then cat in the model.
            views = [self.to_tensor(self.augment(image.copy()) if self.train else image)]
            for i in range(self.grid):
                for j in range(self.grid):
                    patch = image.crop((i*self.size//self.grid, j*self.size//self.grid,
                                        (i+1)*self.size//self.grid, (j+1)*self.size//self.grid))
                    patch = patch.resize((self.size, self.size), Image.Resampling.BILINEAR)
                    views.append(self.to_tensor(self.augment(patch) if self.train else patch))
            return views, label, index
        if self.method == 'CIPL' and self.train:
            # Pair only with training images; teacher/student share spatial geometry.
            other = random.randrange(len(self))
            second = self.geometry(self._image(other))
            labels = torch.tensor(self.rows[other]['gt_score'], dtype=torch.float32)
            views = [self.to_tensor(image), self.to_tensor(second),
                     self.to_tensor(self.augment(image.copy())), self.to_tensor(self.augment(second.copy()))]
            return views, torch.stack([label, labels]), index
        return self.to_tensor(self.augment(image) if self.train else image), label, index
