# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule, ModuleDict

from mmpretrain.registry import MODELS
from mmpretrain.structures import MultiTaskDataSample
from evaluation.Rankingmulti_label import *
from mmpretrain.structures import DataSample
from mmengine.evaluator import Evaluator
from evaluation.Rankingmulti_label import *


def loss_convertor(loss_func, task_name):

    def wrapped(inputs, data_samples, **kwargs):
        mask = torch.empty(len(data_samples), dtype=torch.bool)
        task_data_samples = []
        for i, data_sample in enumerate(data_samples):
            assert isinstance(data_sample, MultiTaskDataSample)
            sample_mask = task_name in data_sample
            mask[i] = sample_mask
            if sample_mask:
                task_data_samples.append(data_sample.get(task_name))

        if len(task_data_samples) == 0:
            # This makes it possible to perform loss.backward when a
            # task does not have gt_labels within a batch.
            loss = (inputs[0] * 0).sum()
            return {'loss': loss, 'mask_size': torch.tensor(0.)}

        # Mask the inputs of the task
        def mask_inputs(inputs, mask):
            if isinstance(inputs, Sequence):
                return type(inputs)(
                    [mask_inputs(input, mask) for input in inputs])
            elif isinstance(inputs, torch.Tensor):
                return inputs[mask]

        masked_inputs = mask_inputs(inputs, mask)
        loss_output = loss_func(masked_inputs, task_data_samples, **kwargs)
        loss_output['mask_size'] = mask.sum().to(torch.float)
        return loss_output

    return wrapped


@MODELS.register_module()
class MultiLabelMultiTaskHead(BaseModule):
    """Multi task head.

    Args:
        task_heads (dict): Sub heads to use, the key will be use to rename the
            loss components.
        common_cfg (dict): The common settings for all heads. Defaults to an
            empty dict.
        init_cfg (dict, optional): The extra initialization settings.
            Defaults to None.
    """

    def __init__(self, task_heads, init_cfg=None, **kwargs):
        super(MultiLabelMultiTaskHead, self).__init__(init_cfg=init_cfg)

        assert isinstance(task_heads, dict), 'The `task_heads` argument' \
            "should be a dict, which's keys are task names and values are" \
            'configs of head for the task.'

        self.task_heads = ModuleDict()

        for task_name, sub_head in task_heads.items():
            if not isinstance(sub_head, nn.Module):
                sub_head = MODELS.build(sub_head, default_args=kwargs)
            # sub_head.loss = loss_convertor(sub_head.loss, task_name)
            self.task_heads[task_name] = sub_head

    def forward(self, feats):
        """The forward process."""
        return {
            task_name: head(feats)
            for task_name, head in self.task_heads.items()
        }

    def loss(self, feats: Tuple[torch.Tensor],
             data_samples: List[MultiTaskDataSample], **kwargs) -> dict:
        """Calculate losses from the classification score.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
            data_samples (List[MultiTaskDataSample]): The annotation data of
                every samples.
            **kwargs: Other keyword arguments to forward the loss module.

        Returns:
            dict[str, Tensor]: a dictionary of loss components, each task loss
                key will be prefixed by the task_name like "task1_loss"
        """
        # if 'Voting' in self.task_heads:
        #     Voting_head = self.task_heads['Voting']
        #     data_samples = Voting_head.predict(feats, data_samples)
        # classification_head = self.task_heads['classification']
        # data_samples = classification_head.predict(feats, data_samples)
        # evaluator = Evaluator(metrics=RankingMultiLabelMetric(thr=0.5))
        # evaluator.process(data_samples)
        # evaluator.evaluate(2)

        # if 'Voting' in self.task_heads:
        #     Voting_head = self.task_heads['Voting']
        #     data_samples = Voting_head.predict(feats, data_samples)
        # classification_head = self.task_heads['classification']
        # data_samples = classification_head.predict(feats, data_samples)

        losses = dict()
        classification_head = self.task_heads['classification']
        if 'Voting' in self.task_heads:
            Voting_head = self.task_heads['Voting']
            data_samples = Voting_head.predict(feats, data_samples)
            # data_samples = classification_head.predict(feats, data_samples)
            
        # data_samples = classification_head.predict(feats, data_samples)
        # evaluator = Evaluator(metrics=RankingMultiLabelMetric(thr=0.5,items=['f1-score', 'precision', 'recall', 'support'],Ranking=True,))
        # evaluator.process(data_samples)
        # evaluator.evaluate(2)
# 
        classification_losses = classification_head.loss(feats, data_samples)
        for k, v in classification_losses.items():
            losses[f'classification_{k}'] = v

        # # else: 
        #     # losses = classification_head.loss(feats, data_samples,Voting_head)
        # task_heads = []
        # for task_name, head in self.task_heads.items():
        #     head_loss = head.loss(feats, data_samples, **kwargs)
        #     for k, v in head_loss.items():
        #         losses[f'{task_name}_{k}'] = v
        return losses

    def predict(
        self,
        feats: Tuple[torch.Tensor],
        data_samples: List[MultiTaskDataSample] = None,
        **kwargs
    ) -> List[MultiTaskDataSample]:
        """Inference without augmentation.

        Args:
            feats (tuple[Tensor]): The features extracted from the backbone.
            data_samples (List[MultiTaskDataSample], optional): The annotation
                data of every samples. If not None, set ``pred_label`` of
                the input data samples. Defaults to None.

        Returns:
            List[MultiTaskDataSample]: A list of data samples which contains
            the predicted results.
        """
        if 'Voting' in self.task_heads:
            Voting_head = self.task_heads['Voting']
            data_samples = Voting_head.predict(feats, data_samples)
        classification_head = self.task_heads['classification']
        
        data_samples = classification_head.predict(feats, data_samples,**kwargs)

        return data_samples
