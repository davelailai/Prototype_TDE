import math
import torch
import copy
import torch.nn.functional as F
from torch import nn, Tensor
from typing import List, Optional,Tuple
from torch.nn import MultiheadAttention

from mmengine.model import BaseModule
from mmpretrain.registry import MODELS
from mmpretrain.structures import DataSample
from .Multi_label_RankCls_head import MultiLabelRankClsHead  # adjust import path as needed

class Identical(nn.Module):
    def forward(self, x, *args, **kwargs):
        return x

class ConsistentPrototypeSystem(nn.Module):
    def __init__(self, num_diseases, feat_dim, 
                prototype_contextual: bool = True, 
                adaptation_strength: float = 0.2,
                ):
        super().__init__()
        self.num_diseases = num_diseases
        self.feat_dim = feat_dim
        self.prototype_contextual = prototype_contextual

        # self.base_prototypes = nn.Parameter(torch.randn(num_diseases, feat_dim))
        
        # 上下文适配器（负责灵活性）
        if self.prototype_contextual:
            self.context_adapter = nn.Sequential(
                nn.Linear(feat_dim, feat_dim//2),
                nn.ReLU(),
                nn.Linear(feat_dim//2, feat_dim)
            )   
            # 适配强度参数（可学习或固定）
            self.adaptation_strength = nn.Parameter(torch.tensor(adaptation_strength))
        else:
            self.context_adapter = Identical()
            self.adaptation_strength = nn.Parameter(torch.tensor(0.0), requires_grad=False)


    def forward(self, base_prototypes, disease_features=None):
        if self.prototype_contextual:
            B, K, D = disease_features.shape
            # 将疾病特征reshape为 [B*K, D]
            disease_features_flat = disease_features.reshape(-1, D)
            
            # 通过适配器网络得到适配量 [B*K, D]
            adaptation_flat = self.context_adapter(disease_features_flat)
            adaptation = adaptation_flat.reshape(B, K, D)   # reshape回 [B, K, D]
            
            
            base_prototypes_expanded = base_prototypes.unsqueeze(0).expand(B, -1, -1)
            
            
            # 应用受限的适配
            limited_adaptation = adaptation * torch.sigmoid(self.adaptation_strength)
            contextual_prototypes = base_prototypes_expanded + limited_adaptation
            return contextual_prototypes, limited_adaptation
        else:
            B, K, D = disease_features.shape
            base_prototypes_expanded = base_prototypes.unsqueeze(0).expand(B, -1, -1)
            return base_prototypes_expanded, None

    
    # 通过强约束保证基础原型的区分度
    def base_prototype_constraints(self, base_prototypes):
        """
        基础原型的强约束：确保不同疾病的原型相互区分
        """
        # 正交性约束：鼓励原型相互正交
        prototypes_norm = F.normalize(base_prototypes, p=2, dim=1)
        similarity = torch.mm(prototypes_norm, prototypes_norm.t())
        eye = torch.eye(base_prototypes.size(0), device=base_prototypes.device)
        ortho_loss = torch.norm(similarity - eye, p='fro')
        
        return  ortho_loss

    def contextual_consistency_loss(self, contextual_prototypes):
        """
        鼓励上下文原型在不同样本间保持一致性
        """
        if contextual_prototypes.dim() == 3 and contextual_prototypes.size(0) > 1:
                # 计算批次内原型方差（鼓励一致性）
            prototype_variance = contextual_prototypes.var(dim=0).mean()
            return prototype_variance
        else:
            return 0
        
    def contrastive_prototype_loss_base(self, features, base_prototypes, labels, temperature=0.1):
        """
        基于基础原型的多标签对比损失
        features: [B, K, D]
        base_prototypes: [K, D]
        labels: [B, K]
        """
        B, K, D = features.shape
        device = features.device

        # 归一化
        features_norm = F.normalize(features, p=2, dim=-1)  # [B, K, D]
        base_prototypes_norm = F.normalize(base_prototypes, p=2, dim=1)  # [K, D]
        base_prototypes_expanded = base_prototypes_norm.unsqueeze(0).expand(B, -1, -1)  # [B, K, D]

        # 相似度 [B, K, K]
        similarity_matrix = torch.bmm(
            features_norm,  # [B, K, D]
            base_prototypes_expanded.transpose(1, 2)  # [B, D, K]
        ) / temperature  # [B, K, K]

        # 正例（对角线）
        diag_idx = torch.arange(K, device=device)
        pos_similarity = similarity_matrix[:, diag_idx, diag_idx]  # [B, K]

        # 负例（非对角线）
        mask_offdiag = ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0)  # [1, K, K]
        neg_similarity = similarity_matrix[mask_offdiag.expand(B, -1, -1)].view(B, K, K-1)  # [B, K, K-1]

        # 只对正标签计算损失
        label_mask = labels > 0.5  # [B, K]
        if not label_mask.any():
            return torch.tensor(0.0, device=device)

        valid_pos_similarity = pos_similarity[label_mask]  # [N_pos]
        valid_neg_similarity = neg_similarity[label_mask]  # [N_pos, K-1]

        numerator = torch.exp(valid_pos_similarity)  # [N_pos]
        denominator = numerator + torch.exp(valid_neg_similarity).sum(dim=1)  # [N_pos]
        loss = -torch.log(numerator / denominator).mean()
        return loss

@MODELS.register_module()
class CounterfactualTDEHead(MultiLabelRankClsHead):

    """
    Query2Label-like head with prototype-based intervention. 
    Separation of forward / loss / predict.
    """
    def __init__(self,
                 position_embedding='sine',img_size=448, hidden_dim=512, nheads=8, enc_layers=6,
                 dec_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", pre_norm=False,
                 return_intermediate_dec=False, 
                 keep_other_self_attn_dec=True, keep_first_self_attn_dec=True,
                 prototype_mapper_hidden: int = 512,
                 intervention_margin: float = 0.4,  # 干预提升的最小幅度
                 lambda_interv_pos: float = 1.0,
                 lambda_interv_neg: float = 1.0,
                 orthogonality_weight: float = 0.1,
                 prototype_init: str = 'independent',  # 'query_mlp' or 'independent'
                 prototype_contextual: bool = False,
                 intervention_mode: str = "proj_gate", # "mlp", "add", or "gate",
                 train_with_TDE: bool = False,
                 task='multi-label',# 'multi-label' or 'binary-class' 'multi-class'
                 **kwargs):
        super().__init__(**kwargs)
        # self.num_classes = self.num_classes
        num_classes = self.num_classes
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv2d(self.in_channels, hidden_dim, kernel_size=1)
        # disease-specific queries
        self.query_embed = nn.Embedding(self.num_classes, hidden_dim)
        self.prototype_init = prototype_init
        self.intervention_mode = intervention_mode
        self.prototype_contextual = prototype_contextual
        self.train_with_TDE = train_with_TDE
        self.intervention_margin = intervention_margin
        self.task = task


        self.transformer =  Transformer(d_model=hidden_dim,
                                        dropout=dropout,
                                        nhead=nheads,
                                        dim_feedforward=dim_feedforward,
                                        num_encoder_layers=enc_layers,
                                        num_decoder_layers=dec_layers,
                                        normalize_before=pre_norm,
                                        return_intermediate_dec=return_intermediate_dec,
                                        rm_self_attn_dec=not keep_other_self_attn_dec, 
                                        rm_first_self_attn=not keep_first_self_attn_dec)
        self.position_encoding = self.build_position_encoding(hidden_dim, position_embedding, img_size)
        self.prototype_context_module = ConsistentPrototypeSystem(num_classes, hidden_dim, prototype_contextual, adaptation_strength=0.2)

        if self.prototype_init == 'query_mlp':
            # self.query_to_proto = nn.Sequential(
            #     nn.Linear(hidden_dim, prototype_mapper_hidden),
            #     nn.ReLU(),
            #     nn.Linear(prototype_mapper_hidden, hidden_dim),
            #     nn.LayerNorm(hidden_dim)
            # )
            self.query_to_proto = Identical()
            self.prototype = None
        elif self.prototype_init == 'independent':
            self.prototype = nn.Embedding(num_classes, hidden_dim)
            # self.prototype_ln = nn.LayerNorm(hidden_dim)
            self.query_to_proto = None
        else:
            raise ValueError(f"Unknown prototype_init: {self.prototype_init}")

        # intervention combiner: concat [query; prototype] -> transformed query
        self.intervention_combiner = InterventionCombiner(
            feat_dim=hidden_dim,
            proto_dim=hidden_dim,
            mode=self.intervention_mode  
        )

        # final classifier: 主分支和干预分支各自一个fc
        self.classifier = GroupWiseLinear(self.num_classes, hidden_dim, bias=True)  # 主分支
        # self.classifier_interv = GroupWiseLinear(self.num_classes, hidden_dim, bias=True)  # 干预分支

        # loss hyperparams
        self.lambda_interv_pos = lambda_interv_pos
        self.lambda_interv_neg = lambda_interv_neg
        self.orthogonality_weight = orthogonality_weight

    def build_position_encoding(self,hidden_dim, position_embedding,img_size,downsample_ratio=32):
        N_steps = hidden_dim // 2

        # if args.backbone in ['CvT_w24'] :
        #     downsample_ratio = 16
        # else:
        downsample_ratio = downsample_ratio

        if position_embedding in ('v2', 'sine'):
            # TODO find a better way of exposing other arguments
            assert img_size % 32 == 0, "args.img_size ({}) % 32 != 0".format(img_size)
            position_embedding = PositionEmbeddingSine(N_steps, normalize=True, maxH=img_size // downsample_ratio, maxW=img_size // downsample_ratio)
            # import ipdb; ipdb.set_trace()
        else:
            raise ValueError(f"not supported {position_embedding}")

        return position_embedding
    
    def get_prototypes(self):
        if self.prototype_init == 'query_mlp':
            contextual_q = self.query_embed.weight  # [K,D]
            return self.query_to_proto(contextual_q)  # [K,D]
        elif self.prototype_init == 'independent':
            return self.prototype.weight  # [K,D]
        else:
            raise ValueError(f"Unknown prototype_init: {self.prototype_init}")



    def head_feature(self,features):
        features = self.pre_logits(features)
        out: List[Tensor] = []
        pos = []
        if isinstance(features, dict):
            for name, x in features.items():
                out.append(x)
                # position encoding
                pos.append(self.position_encoding(x).to(x.dtype))
                # pos.append(self[1](x).to(x.dtype))
        else:
            # for swin Transformer
            out.append(features)
            pos.append(self.position_encoding(features).to(features.dtype))
            # pos.append(self[1](features).to(features.dtype))

        # src, pos = self.backbone(input)
        src, pos = out[-1], pos[-1]
        # import ipdb; ipdb.set_trace()

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(src), query_input, pos) # B,K,d
        out = hs[0][-1]
        attention = hs[-1]

        return out, attention
    def forward(self, features, **kwargs):
        """
        Produces raw logits for intervention=1 path and also retains logits_0 for intervention comparison.
        features is unused here; real input is internal query pipeline.
        Returns:
            logits_0: [B, K] without prototype intervention
            logits_1: [B, K] with intervention
        """
        # contextual queries
        hs, attention_matrix = self.head_feature(features)
        base_prototype = self.get_prototypes()  # [K, D]
        contextual_prototypes, _ = self.prototype_context_module(base_prototype, disease_features=hs)

        return hs, attention_matrix, contextual_prototypes
    def get_rampup_weight(self, base_weight, cur_epoch, rampup_epochs=5):
        if rampup_epochs <= 0:
            return base_weight
        ramp = min(cur_epoch / rampup_epochs, 1.0)
        return base_weight * ramp


    def uplift_loss_diag_from_feat(self, logits_0, intervened_feats_sub, labels,
                                    gamma_pos=2.0, gamma_neg=2.0, k_margin=0.7, epsilon=1e-8):
            """
            对角线 uplift loss:
            - 正例使用自适应 margin + focal weight
            - 负例作用在 uplift_diag 上，用 focal weight 强调下降过多的负例
            """
            B, K = logits_0.shape
            device = logits_0.device
            logits_1_list = []

            # 干预后的 logits
            for j in range(K):
                feat_cj = intervened_feats_sub[:, :, j, :]  # [B, K, D]
                logit_cj = self.classifier(feat_cj)        # [B, K]
                logits_1_list.append(logit_cj.unsqueeze(2))  # [B, K, 1]
            logits_1 = torch.cat(logits_1_list, dim=2)      # [B, K, K]

            # 预测概率
            probs_0 = logits_0
            probs_1 = logits_1

            diag_indices = torch.arange(K, device=device)
            uplift_diag = probs_0 - probs_1[:, diag_indices, diag_indices]  # [B,K]

            # 正例 mask
            pos_mask = labels > 0.5
            margin_pos = self.intervention_margin * torch.abs(logits_0)
            # 正例 focal weight
            # relative_drop = uplift_diag / (probs_0.abs() + epsilon)
            # focal_weight_pos = ((1 - relative_drop).clamp(min=0,max=5)) ** gamma_pos
            
            drop_loss_positive = F.relu(margin_pos - uplift_diag)
            pos_loss=drop_loss_positive[pos_mask]
            # pos_loss = (focal_weight_pos * drop_loss_positive)[pos_mask]
            pos_loss = pos_loss.mean() if pos_loss.numel() > 0 else torch.tensor(0.0, device=device)

            # 负例 mask
            neg_mask = labels <= 0.5
            if neg_mask.sum() > 0:
                # focal loss 在 uplift_diag 上
                uplift_diag_neg = torch.abs(uplift_diag)
                # relative_drop = uplift_diag_neg / (probs_0.abs() + epsilon)
                # relative_drop = torch.clamp(relative_drop, min=0, max=5)
                # focal_weight_neg = (relative_drop) ** gamma_neg
                neg_loss = F.relu(uplift_diag_neg)
                neg_loss = neg_loss[neg_mask].mean()
                # neg_loss = (focal_weight_neg * neg_loss)[neg_mask].mean()
            else:
                neg_loss = torch.tensor(0.0, device=device)

            return pos_loss + neg_loss

    def invariance_loss_offdiag_from_feat(self, logits_0, intervened_feats_sub):
        """
        非对角线invariance loss: 去除非本类prototype后输出应与原始一致
        classifier: 分类器（如self.classifier）
        logits_0: [B, K] 原始logits
        intervened_feats_sub: [B, K, K, D] 干预后特征（减去prototype）
        """
        B, K = logits_0.shape
        device = logits_0.device
        logits_1_list = []
        for j in range(K):
            feat_cj = intervened_feats_sub[:, :, j, :]           # [B, K, D]
            logit_cj = self.classifier(feat_cj)       # [B, K] 干预分支用独立fc
            logits_1_list.append(logit_cj.unsqueeze(2))      # [B, K, 1]
        logits_1 = torch.cat(logits_1_list, dim=2)  # [B, K, K] 
        
        probs_0 = logits_0
        probs_1 = logits_1
        logits_0_exp = probs_0.unsqueeze(2).expand(B, K, K)  # [B, K, K]
        mask_offdiag = ~torch.eye(K, dtype=torch.bool, device=device).unsqueeze(0).expand(B, K, K)  # [B,K,K]
        loss_invariance = F.mse_loss(probs_1[mask_offdiag], logits_0_exp[mask_offdiag])
        return loss_invariance
    
    def loss(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample],
             **kwargs) -> dict:
        """
        Compute all losses given model outputs and data_samples.
        """
        probe_attn_feats, attention_matrix, prototypes_contextual = self(feats)
        B, K, D = probe_attn_feats.shape
        # intervention=0
        logits_0 = self.classifier(probe_attn_feats)  
        # [B,K]
        # Supervised loss
        losses = self._get_loss(logits_0[:len(data_samples),:], data_samples, **kwargs)

        # losses.pop('loss')
        f_c = probe_attn_feats.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
            
        # intervention=1: for each disease c, combine its feature with all prototypes j -> [B, K, K, D]
        proto = prototypes_contextual.view(B, 1, K, D).expand(B, K, K, D)
        # proto = self.get_prototypes().view(1, 1, K, D).expand(B, K, K, D)  # [B, K, K, D]
        # intervened_feats = self.intervention_combiner(f_c, proto)  # [B, K, K, D]
        intervened_feats_sub = self.intervention_combiner(f_c, proto, direction="sub")
        if self.train_with_TDE:
                TDE_logits = self.TDE_inference(logits_0, intervened_feats_sub)
                losses_tde = self._get_loss(TDE_logits[:len(data_samples),:], data_samples, **kwargs)
                losses['TDE_cls_loss'] = losses_tde['loss']
        
        labels = torch.stack([ds.gt_score for ds in data_samples]).float()  # [B, K]
        ### Prototype pureily losses
        base_prototype_constraints = self.prototype_context_module.base_prototype_constraints(self.get_prototypes())
        losses['base_prototype_ortho_loss'] = base_prototype_constraints * self.orthogonality_weight
        # contextual_consistency = self.prototype_context_module.contextual_consistency_loss(prototypes_contextual)
        # losses['contextual_prototype_consistency_loss'] = contextual_consistency * self.orthogonality_weight
        contrastive_loss_base = self.prototype_context_module.contrastive_prototype_loss_base(probe_attn_feats, self.get_prototypes(), labels, temperature=0.1)
        losses['contrastive_loss_base'] = contrastive_loss_base * self.orthogonality_weight


        if self.current_epoch > self.warmup_epoch:
            cur_epoch=self.current_epoch-self.warmup_epoch
            interv_pos = self.get_rampup_weight(self.lambda_interv_pos, cur_epoch, rampup_epochs=self.transition_length)
            interv_neg = self.get_rampup_weight(self.lambda_interv_neg, cur_epoch, rampup_epochs=self.transition_length)
        ####---------------  反事实uplift和invariance损失：logits level的loss------------###
    
            uplift = self.uplift_loss_diag_from_feat(logits_0, intervened_feats_sub, labels, k_margin=self.intervention_margin)
            invariance = self.invariance_loss_offdiag_from_feat(logits_0, intervened_feats_sub)
            
            losses['uplift_loss'] = uplift * interv_pos
            losses['invariance_loss'] = invariance * interv_neg        
        return losses
    def TDE_inference(self, logits_0, intervened_feats_sub):
        """
        TDE inference: average logits from original and prototype-only paths.
        feats: expects patch_feats of shape [B,N,K,D]
        """
        B, K = logits_0.shape
        device = logits_0.device
        logits_1_list = []
        for j in range(K):
            feat_cj = intervened_feats_sub[:, :, j, :]           # [B, K, D]
            logit_cj = self.classifier(feat_cj)       # [B, K] 干预分支用独立fc
            logits_1_list.append(logit_cj.unsqueeze(2))      # [B, K, 1]
        logits_1 = torch.cat(logits_1_list, dim=2)  # [B, K, K] 

        probs_0 = torch.sigmoid(logits_0)
        probs_1 = torch.sigmoid(logits_1)
        probs_0_exp = probs_0.unsqueeze(2).expand(B, K, K)  # [B, K, K]
        uplift= probs_0_exp - probs_1
        diag_indices = torch.arange(K, device=device)
        uplift_diag = uplift[:, diag_indices, diag_indices]  # [B,K]
        # uplift_diag = (probs_0 - probs_1[:, diag_indices, diag_indices])  # [B,K]
        return uplift_diag, uplift
    def predict(self, feats: Tuple[torch.Tensor], data_samples: List[DataSample] = None, **kwargs):
        """
        Compatibility wrapper to match expected API.
        feats: expects patch_feats of shape [B,N,K,D]
        """
        kwargs['data_sample']=data_samples
        # patch_feats = feats  # naming alignment
        probe_attn_feats, attention_matrix, prototypes_contextual = self(feats)
        B, K, D = probe_attn_feats.shape
        # intervention=0
        cls_score = self.classifier(probe_attn_feats) 
        proto = prototypes_contextual.view(B, 1, K, D).expand(B, K, K, D)
        f_c = probe_attn_feats.unsqueeze(2).expand(B, K, K, D)  # [B, K, K, D]
        intervened_feats_sub = self.intervention_combiner(f_c, proto, direction="sub")
        TDE_logits,uplift = self.TDE_inference(cls_score, intervened_feats_sub)
        # reuse MultiLabelRankClsHead._get_predictions if exists
        predictions = self._get_predictions(cls_score, TDE_logits,data_samples)
        if not self.training:
            if not hasattr(self, 'test_features'):
                self.test_features = []
            if not hasattr(self, 'test_data_samples'):
                self.test_data_samples = []
            if not hasattr(self, 'test_TDE_logits'):
                self.test_TDE_logits = []
            self.test_features.append(probe_attn_feats.detach().cpu())
            self.test_TDE_logits.append(uplift.detach().cpu())  
            if data_samples is not None:
                self.test_data_samples.extend(data_samples)
        return predictions
    def _get_predictions(self, cls_score: torch.Tensor, TDE_logits: torch.Tensor,   
                         data_samples: List[DataSample]):
        """Post-process the output of head.

        Including softmax and set ``pred_label`` of data samples.
        """
        if self.task == 'multi-label':
            pred_scores = torch.sigmoid(cls_score)
        else:  # 'binary-class'
            pred_scores = torch.softmax(cls_score, dim=1)
        # TDE_pred_scores = torch.sigmoid(TDE_logits)
        TDE_pred_scores = TDE_logits
        # pred_scores = cls_score

        if data_samples is None:
            data_samples = [DataSample() for _ in range(cls_score.size(0))]
        for data_sample, score, TDE_logit in zip(data_samples, pred_scores, TDE_pred_scores):
            if 'disease_count' in data_sample:
                # Thr = torch.sigmoid(data_sample.disease_count.clone())
                Thr = data_sample.disease_count.clone()
                # count_labels = count.argmax(dim=0, keepdim=True)
                # if count.numel()==0:
                #     label = count
                # else:
                score = torch.sigmoid(score-Thr)
                # label = torch.where(score >= Thr)[0]
                # label = torch.where(score >= self.thr)[0]

                # label = (score > Thr).nonzero(as_tuple=True)[0]
                # _, label = score.topk(count_labels.item())
            if self.thr is not None:
                # a label is predicted positive if larger than thr
                label = torch.where(score >= self.thr)[0]
            else:
                # top-k labels will be predicted positive for any example
                _, label = score.topk(self.topk)
            data_sample.set_pred_score(score).set_pred_label(label)
            # 保存TDE_logits
            if TDE_logits is not None:
                data_sample.TDE_logits = TDE_logit.detach().cpu()  # 或 .numpy() 视需求

        return data_samples


import torch
import torch.nn as nn
import torch.nn.functional as F

class InterventionCombiner(nn.Module):
    def __init__(self, feat_dim, proto_dim, mode="proj_gate", dropout=0.1):
        """
        mode: ["mlp", "gate", "add", "proj_gate"]
        - mlp:  [z, p] -> MLP -> residual update
        - gate: simple gated add/sub (no projection)
        - add:  direct add/sub
        - proj_gate: projection onto proto direction, scaled by gate
        """
        super().__init__()
        self.mode = mode
        self.feat_dim = feat_dim
        self.proto_dim = proto_dim

        if mode == "mlp":
            self.combiner = nn.Sequential(
                nn.Linear(feat_dim + proto_dim, feat_dim),
                nn.ReLU(),
                nn.Linear(feat_dim, feat_dim),
                nn.LayerNorm(feat_dim)
            )
            if dropout > 0:
                self.combiner.add_module("dropout", nn.Dropout(dropout))

        elif mode == "gate":
            self.gate_layer = nn.Sequential(
                nn.Linear(feat_dim + proto_dim, feat_dim),
                nn.Sigmoid()
            )

        elif mode == "proj_gate":
            pass

        elif mode == "add":
            pass

        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.ln = nn.LayerNorm(feat_dim)

    def forward(self, feat, proto, direction="add"):
        """
        feat: [B, d]
        proto: [B, d] 或 [d]
        direction: "add" 或 "sub"
        """
        if proto.dim() == 1:
            proto = proto.unsqueeze(0).expand_as(feat)

        if self.mode == "mlp":
            if direction == "add":
                combined = torch.cat([feat, proto], dim=-1)
            elif direction == "sub":
                combined = torch.cat([feat, -proto], dim=-1)
            out = self.combiner(combined)
            out = feat + out

        elif self.mode == "gate":
            combined = torch.cat([feat, proto], dim=-1)
            gate = self.gate_layer(combined)
            if direction == "add":
                out = feat + gate * proto
            elif direction == "sub":
                out = feat - gate * proto

        elif self.mode == "add":
            if direction == "add":
                out = feat + proto
            elif direction == "sub":
                out = feat - proto

        elif self.mode == "proj_gate":
            # 归一化 proto，保证 projection 稳定
            proto_norm = F.normalize(proto, dim=-1, eps=1e-6)
            coeff = (feat * proto_norm).sum(dim=-1, keepdim=True)
            proj = coeff * proto_norm  # [B, d]

            if direction == "sub":
                out = feat - proj
            elif direction == "add":
                out = feat + proj

        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        out = self.ln(out)
        return out

class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, maxH=30, maxW=30):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.maxH = maxH
        self.maxW = maxW
        pe = self._gen_pos_buffer()
        self.register_buffer('pe', pe)

    def _gen_pos_buffer(self):
        _eyes = torch.ones((1, self.maxH, self.maxW))
        y_embed = _eyes.cumsum(1, dtype=torch.float32)
        x_embed = _eyes.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t,2,rounding_mode='trunc') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

    def forward(self, input: Tensor):
        x = input
        return self.pe.repeat((x.size(0),1,1,1))

class GroupWiseLinear(nn.Module):
    # could be changed to: 
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False, 
                 rm_self_attn_dec=True, rm_first_self_attn=True,
                 ):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        if num_decoder_layers > 0:
            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead
        self.rm_self_attn_dec = rm_self_attn_dec
        self.rm_first_self_attn = rm_first_self_attn

        if self.rm_self_attn_dec or self.rm_first_self_attn:
            self.rm_self_attn_dec_func()

        # self.debug_mode = False
        # self.set_debug_mode(self.debug_mode)

    def rm_self_attn_dec_func(self):
        total_modifie_layer_num = 0
        rm_list = []
        for idx, layer in enumerate(self.decoder.layers):
            if idx == 0 and not self.rm_first_self_attn:
                continue
            if idx != 0 and not self.rm_self_attn_dec:
                continue
            
            layer.omit_selfattn = True
            del layer.self_attn
            del layer.dropout1
            del layer.norm1

            total_modifie_layer_num += 1
            rm_list.append(idx)
        # remove some self-attention layer
        # print("rm {} layer: {}".format(total_modifie_layer_num, rm_list))

    def set_debug_mode(self, status):
        print("set debug mode to {}!!!".format(status))
        self.debug_mode = status
        if hasattr(self, 'encoder'):
            for idx, layer in enumerate(self.encoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)
        if hasattr(self, 'decoder'):
            for idx, layer in enumerate(self.decoder.layers):
                layer.debug_mode = status
                layer.debug_name = str(idx)


    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed, mask=None):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1) 
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        # Encoder (原样)
        if self.num_encoder_layers > 0:
            memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        else:
            memory = src

        tgt = torch.zeros_like(query_embed)
        hs, attn_weights = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        return hs.transpose(1, 2), memory[:h*w].permute(1, 2, 0).view(bs, c, h, w), attn_weights


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        attn_weights_list = []
        intermediate = []

        for layer in self.layers:
            output, sim_mat_2 = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            attn_weights_list.append(sim_mat_2)
            if self.return_intermediate:
                if self.norm is not None:
                    intermediate.append(self.norm(output))
                else:
                    intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate), attn_weights_list

        return output.unsqueeze(0), attn_weights_list


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2, corr = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        

        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
                            
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.debug_mode = False
        self.debug_name = None
        self.omit_selfattn = False

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        if not self.omit_selfattn:
            tgt2, sim_mat_1 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                                key_padding_mask=tgt_key_padding_mask)

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        tgt2, sim_mat_2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                key=self.with_pos_embed(memory, pos),
                                value=memory, attn_mask=memory_mask,
                                key_padding_mask=memory_key_padding_mask)
        
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, sim_mat_2

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
                            
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=False,
        rm_self_attn_dec=not args.keep_other_self_attn_dec, 
        rm_first_self_attn=not args.keep_first_self_attn_dec,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu/glu, not {activation}.")
