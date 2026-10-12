_base_ = [
    './backbone.py',
    '../_base_/datasets/Endo.py',
    '../_base_/schedules/FFA_bs512_100e_coslr_warmup.py',
    '../_base_/default_runtime.py'
]

work_dir = './work_dirs/Endo_Prototype_TDE'

model = dict(
    # type='ImageGanClassifier',
    # backbone=dict(
    #     type='EfficientNetV2', 
    #     arch='s',
    #     init_cfg=dict(
    #         type='Pretrained',
    #         checkpoint='https://download.openmmlab.com/mmclassification/v0/efficientnetv2/efficientnetv2-s_3rdparty_in1k_20221220-f0eaff9d.pth',
    #         prefix='backbone')),
    head=dict(
        type='MultiLabelMultiTaskHead',  # Multitask head class
            task_heads={
                "classification":dict(
                    type='CounterfactualTDEHead',
                    # num_class=6,
                    # in_channels=1280,
                    position_embedding='sine',
                    enc_layers=1,
                    dec_layers=2,
                    # # dim_feedforward=256,
                    hidden_dim=512,
                    nheads=8,
                    # img_size=448,
                    loss=dict(type='MILAsymmetricLoss', loss_weight=1.0, use_sigmoid=True),
                    prototype_init= 'independent',  # 'query_mlp' or 'independent'
                    intervention_mode= "proj_gate",
                    lambda_interv_pos=1.0,
                    lambda_interv_neg=1.0,
                    orthogonality_weight=0.1,
                    prototype_contextual=True,
                    intervention_margin=0.5,
                    train_with_TDE=False,
                    # Visualization='work_dirs_test/Visua',
                    # disease_description_file = '/users/lailai/sharedscratch/openmmlab/mmpretrain/data/FFA/disease_only.txt'
                    # causal=True,
                    # d2=30,
                    # feature_reg=True,
                    # causal_reg=True,
                    # causal_num=20,
                    # s=1.0, 
                    # lambda1=1e-1, 
                    # m=0.1,
                    # weight=0.1,
                    )}
    ),
    neck=None)

# train_dataloader = dict(
#     batch_size=4)
# If you want standard test, please manually configure the test dataset
# val_evaluator = [
#     dict(type='AveragePrecision'),
#     dict(type='AverageAUC'),
#     dict(average=None, type='AverageAUC'),
#     dict(type='MultiLabelMetric',
#         items=['f1-score','precision', 'recall', 'f1-score','support'],
#         # average='both',
#         thr=0.5),
#     dict(type='MultiLabelMetric',
#         items=['precision', 'recall', 'f1-score','support'],
#         average='micro',
#         thr=0.5),
#     dict(type='MultiLabelMetric',
#         items=['precision', 'recall', 'f1-score','support'],
#         average=None,
#         thr=0.5),
# ]
# test_evaluator = val_evaluator
model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True,
        # detect_anomalous_params=True
        )
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
custom_hooks = [
    dict(type='EpochWarmupHook',
        priority='VERY_HIGH',
         warmup_epoch=-1, 
         transition_length=10),  # transition_length is the length of the transition period
         ]
val_evaluator = [
    # dict(type='RankingMultiLabelMetric',
    #     items=['f1-score'],
    #     # average='both',
    #     # Ranking=Ranking,
    #     thr=0.5),
    
    dict(type='RankingROC_AUC', Validation_mode='original'),
    dict(type='RankingROC_AUC', Validation_mode='TDE'),

    dict(type='RankingMultiLabelMetric',
        items=['precision', 'recall', 'f1-score','support'],
        thr=0.5),
]


test_evaluator = [
    dict(type='RankingROC_AUC', Validation_mode='TDE'),
    dict(type='RankingROC_AUC', Validation_mode='original'),
    dict(type='MultiLabel_AveragePrecision', Validation_mode='TDE'),
    dict(type='MultiLabel_AveragePrecision', Validation_mode='original'),

    dict(type='RankingMultiLabelMetric',
        items=['f1-score', 'precision', 'recall', 'support'],
        # Ranking=Ranking,
        # average='both',
        thr=0.5),
    dict(type='RankingMultiLabelMetric', Validation_mode='TDE',
        items=['precision', 'recall', 'f1-score','support'],
        thr=0.25),
    dict(type='RankingROC_AUC',average=None),
]
# optim_wrapper = dict(
#     optimizer=dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001))

# default_hooks = dict(
#     # record the time of every iteration.
#     logger=dict(type='LoggerHook', interval=50),

#     checkpoint=dict(type='CheckpointHook', interval=20,save_best='auto'),
# )
