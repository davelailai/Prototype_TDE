"""Controlled ResNet-50 adaptations. See README for upstream differences."""
import importlib
import importlib.util
import copy
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from common import append_normal, asymmetric_loss, paired_ce, patch_logits, ramp
from protocol import audit_backbone


class ResSpatial(nn.Module):
    def __init__(self, checkpoint, backbone_config=None):
        super().__init__()
        from mmpretrain.models.backbones import ResNet
        config = copy.deepcopy(backbone_config or dict(type='ResNet', depth=50, out_indices=(3,)))
        config.pop('type')
        config['init_cfg'] = dict(type='Pretrained', checkpoint=checkpoint, prefix='backbone') if checkpoint else None
        self.net = ResNet(**config)
        from mmengine.logging import MMLogger
        logger = MMLogger.get_current_instance()
        level = logger.level
        logger.setLevel('WARNING')
        try:
            self.net.init_weights()
        finally:
            logger.setLevel(level)
        self.initialization_audit = audit_backbone(self.net, checkpoint)

    def forward(self, image):
        return self.net(image)[-1]


class PAT(nn.Module):
    """PAT-T, cross attention: three classifiers plus patch-wise logit attention."""
    def __init__(self, backbone, channels, classes, memory_checkpointing=False):
        super().__init__()
        self.backbone = backbone
        self.heads = nn.ModuleList([nn.Linear(channels, classes) for _ in range(3)])
        self.classification_loss = asymmetric_loss
        self.memory_checkpointing = memory_checkpointing

    def backward_loss(self, loss):
        # Checkpointed ResNet blocks recompute BN during backward. Keep only the
        # single original forward's running-stat updates, including its counter.
        states = [(module, {name: value.detach().clone() for name, value in module.named_buffers(recurse=False)})
                  for module in self.backbone.modules()
                  if isinstance(module, nn.modules.batchnorm._BatchNorm)] if self.memory_checkpointing else []
        try:
            loss.backward()
        finally:
            with torch.no_grad():
                for module, buffers in states:
                    for name, value in buffers.items():
                        getattr(module, name).copy_(value)

    def logits(self, images):
        batch = images[0].shape[0]
        feature = self.backbone(torch.cat(images)).mean((2, 3))
        original = self.heads[0](feature[:batch])
        attention = self.heads[1](feature[batch:])
        values = self.heads[2](feature[batch:])
        return original, patch_logits(attention, values, batch)

    def loss(self, images, labels, epoch, step=0):
        original, patches = self.logits(images)
        return self.classification_loss(original, labels) + self.classification_loss(patches, labels)

    def forward(self, images):
        original, patches = self.logits(images)
        return ((original + patches) / 2).sigmoid()


def load_miccai_head(source):
    name = '_official_miccai.models.heads.multi_label_Q2L_head_new'
    if name in sys.modules:
        return sys.modules[name].Qeruy2Label_New
    from mmpretrain.models.heads import multi_label_cls_head
    root = source/'mmpretrain/models'
    for package_name, path in [('_official_miccai', source/'mmpretrain'),
                                ('_official_miccai.models', root),
                                ('_official_miccai.models.heads', root/'heads'),
                                ('_official_miccai.models.utils', root/'utils')]:
        package = types.ModuleType(package_name)
        package.__path__ = [str(path)]
        sys.modules[package_name] = package
    utils = sys.modules['_official_miccai.models.utils']
    dag = importlib.import_module('_official_miccai.models.utils.DAG_unit')
    utils.LocallyConnected, utils.DagmaMLP = dag.LocallyConnected, dag.DagmaMLP
    sys.modules['_official_miccai.models.heads.multi_label_cls_head'] = multi_label_cls_head
    from mmpretrain.registry import MODELS
    # Some local MMPretrain installs already contain an experimental class with
    # this name. Import upstream in isolation, then restore that registration.
    previous = MODELS.module_dict.pop('Qeruy2Label_New', None)
    try:
        return importlib.import_module(name).Qeruy2Label_New
    finally:
        if previous is not None:
            MODELS.module_dict['Qeruy2Label_New'] = previous


class MICCAI2024(nn.Module):
    """Official query/DAG head, on the controlled shared backbone."""
    def __init__(self, backbone, channels, classes, source, size):
        super().__init__()
        self.backbone = backbone
        dtype = torch.get_default_dtype()
        try:
            self.head = load_miccai_head(source)(in_channels=channels, num_classes=classes,
                position_embedding='sine', img_size=size, enc_layers=1, dec_layers=2,
                dim_feedforward=256, hidden_dim=128, nheads=4, d2=30,
                causal=True, feature_reg=True, weight=.1,
                loss=dict(type='AsymmetricLoss', loss_weight=1., use_sigmoid=True))
        finally:
            torch.set_default_dtype(dtype)

    def forward(self, images):
        return self.head((self.backbone(images),)).sigmoid()

    def loss(self, images, labels, epoch, step=0):
        from mmpretrain.structures import DataSample
        samples = [DataSample().set_gt_score(row) for row in labels]
        losses = self.head.loss((self.backbone(images),), samples)
        return sum(value.mean() for key, value in losses.items() if 'loss' in key)


def load_cipl(source):
    # Upstream uses unqualified local imports; adapters run in their own process.
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location('_official_cipl', source / 'model.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CIPL


class CIPL(nn.Module):
    def __init__(self, backbone, classes, source, settings, size):
        super().__init__()
        self.settings = settings
        shape = (settings['prototypes_per_class']*(classes+1), settings['prototype_dim'], 1, 1)
        self.core = load_cipl(source)(features=backbone, img_size=size, prototype_shape=shape,
                                     proto_layer_rf_info=None, num_classes=classes+1,
                                     add_on_layers_type='bottleneck')
        self.core.last_layer.requires_grad_(False)

    def get_extra_state(self):
        return {'ema_initialized': self.core._ema_param_initialized}

    def set_extra_state(self, state):
        self.core._ema_param_initialized = state['ema_initialized']

    def forward(self, image):
        logits, _, _ = self.core.forward_infer(image)
        return (logits[:, :-1] - logits[:, -1:]).sigmoid()

    def cluster_separation(self, distances, labels):
        identity = self.core.prototype_class_identity.to(distances.device).bool()
        cluster, separation = [], []
        for distance, label in zip(distances, labels):
            positive = label.bool()
            per_class = distance[:, None].expand_as(identity).masked_fill(~identity, float('inf')).min(0).values
            cluster.append(per_class[positive].sum())
            wrong = ~identity[:, positive].any(1)
            separation.append(distance[wrong].min() if wrong.any() else distance.sum()*0)
        return torch.stack(cluster).mean(), torch.stack(separation).mean()

    def loss(self, images, labels, epoch, step=0, last_only=False):
        labels = append_normal(labels)
        paired = labels.ndim == 3
        if paired:
            labels = torch.cat([labels[:, 0], labels[:, 1]])
        warm = epoch < self.settings['warm_epochs']
        if warm:
            warm_images = torch.cat(images[:2]) if paired else images
            logits, distances, _ = self.core.forward_infer(warm_images)
        else:
            if not paired:
                raise ValueError('CIPL joint/refinement stages require shared-class image pairs')
            logits, distances, co_logits, maps, teacher_maps, teacher_logits = self.core(images[:2], images[2:])
        cluster, separation = self.cluster_separation(distances, labels)
        classification = paired_ce(logits, labels)
        separation_hinge = F.relu(2-separation)
        loss = classification + .02*cluster + .02*separation_hinge
        components = dict(classification=classification, cluster=cluster, separation_hinge=separation_hinge)
        if not warm:
            pair = labels.chunk(2)
            intersection = torch.cat([pair[0]*pair[1]]*2)
            components['cross_image'] = paired_ce(co_logits, intersection)
            loss = loss + .5*components['cross_image']
            student = torch.stack([logits[:, -1:].expand_as(logits[:, :-1]), logits[:, :-1]], -1).softmax(-1)
            teacher = torch.stack([teacher_logits[:, -1:].expand_as(teacher_logits[:, :-1]),
                                   teacher_logits[:, :-1]], -1).softmax(-1).detach()
            gram_s = torch.einsum('bkc,dkc->kbd', student, student)
            gram_t = torch.einsum('bkc,dkc->kbd', teacher, teacher)
            components['prediction_alignment'] = F.mse_loss(gram_s, gram_t)
            loss = loss + 10*components['prediction_alignment']
            similarity = (F.normalize(maps, dim=1)*F.normalize(teacher_maps.detach(), dim=1)).sum(1)
            components['map_alignment'] = (1-similarity).sum((1, 2)).mean()
            loss = loss + .5*components['map_alignment']
        self.last_loss_components = {name: value.detach() for name, value in components.items()}
        return loss

    @torch.no_grad()
    def constrain_last_layer(self):
        self.core.last_layer.weight.mul_(self.core.prototype_class_identity.t().to(self.core.last_layer.weight))
        self.core.last_layer.weight.clamp_(min=0)

    @torch.no_grad()
    def push(self, loader, device):
        """Official greedy projection: class-eligible patches from distinct images."""
        self.eval()
        prototypes = self.core.prototype_vectors
        identity = self.core.prototype_class_identity.to(device)
        distances_by_image, positions_by_image, identifiers = [], [], []
        for images, labels, indices in loader:
            features, _ = self.core.conv_features(images.to(device))
            distances = self.core._l2_convolution(features, prototypes)
            labels = append_normal(labels.to(device))
            allowed = (labels @ identity.t()).bool()
            best, location = distances.flatten(2).min(2)
            distances_by_image.append(best.masked_fill(~allowed, float('inf')).cpu().numpy())
            positions_by_image.append(location.cpu().numpy())
            identifiers.extend(torch.as_tensor(indices).tolist())
        distances_by_image = np.concatenate(distances_by_image)
        positions_by_image = np.concatenate(positions_by_image)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError('CIPL projection requires a non-repeated, non-shuffled training search set')
        selected, used = {}, set()
        for j in range(len(prototypes)):
            for row in np.argsort(distances_by_image[:, j], kind='stable'):
                if not np.isfinite(distances_by_image[row, j]):
                    break
                index = identifiers[row]
                if index not in used:
                    selected[index] = (j, int(positions_by_image[row, j]))
                    used.add(index)
                    break
        # Re-extract only chosen images to avoid caching every image's feature map.
        for images, _, indices in loader:
            chosen = [(b, selected[int(index)]) for b, index in enumerate(indices) if int(index) in selected]
            if not chosen:
                continue
            features, _ = self.core.conv_features(images[[b for b, _ in chosen]].to(device))
            for b, (_, (j, location)) in enumerate(chosen):
                h, w = divmod(location, features.shape[3])
                prototypes[j, :, 0, 0].copy_(features[b, :, h, w])
        return dict(projected=len(selected), total=len(prototypes),
                    image_to_prototype={str(index): j for index, (j, _) in selected.items()})


def load_proci(source):
    # Keep upstream's relative models imports separate from this project's models.
    package = types.ModuleType('_official_proci')
    package.__path__ = [str(source)]
    sys.modules['_official_proci'] = package
    models = types.ModuleType('_official_proci.models')
    models.__path__ = [str(source / 'models')]
    sys.modules['_official_proci.models'] = models
    causal = importlib.import_module('_official_proci.causal')
    previous = sys.modules.get('causal')
    sys.modules['causal'] = causal
    try:
        module = importlib.import_module('_official_proci.models.resnet_with_table')
    finally:
        if previous is None:
            sys.modules.pop('causal', None)
        else:
            sys.modules['causal'] = previous
    return module.GenericBackboneWithTable


class ProCI(nn.Module):
    def __init__(self, backbone, channels, classes, source, settings):
        super().__init__()
        self.settings = settings
        self.classification_loss = asymmetric_loss
        self.aligned_loss = False
        self.core = load_proci(source)(backbone, channels, len(classes), types.SimpleNamespace(classes_names=classes))

    def forward(self, images):
        return self.core(images)[0].sigmoid()

    def loss(self, images, labels, epoch, step=0, write_memory=True, memory_snapshot=None):
        s = self.settings
        counts = self.core.memory.counts.float().clamp_min(1)
        caps = (s['cap_per_class']*(counts/counts.median()).pow(-.5)).round().clamp(4, 12).long()
        cfg = dict(thr=.85+(.60-.85)*ramp(epoch, 5, 20),
                   proto_topq_warm=.30, proto_min_pos=1, proto_sim_gate=0., proto_min_count_sim=5,
                   cap_k=caps, proto_write_start=5)
        output = self.core.forward_with_protos(images, labels, epoch=epoch, cfg=cfg,
                                               ema_probs_override=getattr(self, 'confidence_override', None))
        logits, z, batch = output['logits'], output['Z'], output['batch_protos']
        old = self.core.memory.protos.detach().clone() if memory_snapshot is None else memory_snapshot
        selected = torch.where(output['updated_mask_loss'][:, None], batch, old)
        wc = ramp(epoch, s['blend_start'], s['blend_end'])
        confounders = F.normalize((1-wc)*F.normalize(selected, dim=-1)+wc*F.normalize(old, dim=-1), dim=-1)
        # Preserve the upstream single-device two-stage memory update, but clone
        # pre-write snapshots to avoid autograd version-counter errors.
        with torch.no_grad():
            m = .80+(s['proto_momentum']-.80)*ramp(epoch, 5, 20)
            write = output['updated_mask_write'][:, None]
            first = torch.where(write, m*old+(1-m)*batch.detach(), old)
            new = torch.where(write, F.normalize(m*first+(1-m)*batch.detach(), dim=-1), first)
            if write_memory:
                self.core.memory.protos.copy_(new)
                self.core.memory.counts.add_(output['write_counts'])
            else:
                new = self.core.memory.protos.detach().clone()
        prior = (self.core.memory.counts.float()+1)
        prior = prior/prior.sum()
        cause = F.normalize((z*labels[:, :, None]).sum(1)/labels.sum(1, keepdim=True).clamp_min(1), dim=-1)
        query = F.normalize(z, dim=-1)
        causal = self.core.causal_module(cause[:, None].expand_as(query).reshape(-1, z.shape[-1]),
                                         query.reshape(-1, z.shape[-1]), confounders, prior,
                                         tau=s['causal_tau']).reshape_as(labels)
        ws = ramp(epoch, s['blend_start'], s['blend_end'])
        scu_prototypes = F.normalize((1-ws)*F.normalize(selected, dim=-1)+ws*F.normalize(new.detach(), dim=-1), dim=-1)
        logp = torch.einsum('bkd,jd->bkj', query, scu_prototypes).div(.3).log_softmax(-1)
        diagonal = logp.diagonal(dim1=1, dim2=2)
        scu = -diagonal[labels.bool()].mean() if labels.any() else logp.sum()*0
        weight = ramp(epoch, 10, 30)
        causal_weight = .05*min(1., step/500) if epoch == 10 else weight
        auxiliary = (s['scu_weight']*weight*scu
                     + s['causal_weight']*causal_weight*asymmetric_loss(causal, labels))
        # Apply the same batch normalization to the complete native objective,
        # keeping auxiliary coefficients rather than silently amplifying them.
        if self.aligned_loss:
            auxiliary = auxiliary / labels.shape[0]
        return self.classification_loss(logits, labels) + auxiliary


def make_model(method, classes, source_root, settings, size, checkpoint, backbone=None, channels=2048,
               classification_loss=None, backbone_config=None):
    if method == 'MLC-PAT' and settings.get('activation_checkpointing', False):
        backbone_config = copy.deepcopy(backbone_config or dict(type='ResNet', depth=50, out_indices=(3,)))
        backbone_config['with_cp'] = True
    backbone = backbone if backbone is not None else ResSpatial(checkpoint, backbone_config)
    if method == 'MLC-PAT':
        model = PAT(backbone, channels, len(classes), memory_checkpointing=settings.get('activation_checkpointing', False))
    elif method == 'CIPL':
        model = CIPL(backbone, len(classes), Path(source_root)/method, settings, size)
    elif method == 'ProCI':
        model = ProCI(backbone, channels, classes, Path(source_root)/method, settings)
    elif method == 'MICCAI2024':
        model = MICCAI2024(backbone, channels, len(classes), Path(source_root)/method, size)
    else:
        raise ValueError(method)
    if classification_loss is not None:
        if method == 'MICCAI2024':
            # Native head passes avg_factor=batch_size itself.
            model.head.loss_module = classification_loss.module
        elif method in ('MLC-PAT', 'ProCI'):
            model.classification_loss = classification_loss
            if method == 'ProCI':
                model.aligned_loss = True
    return model
