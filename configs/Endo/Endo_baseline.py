_base_ = [
    './backbone.py',
    '../_base_/datasets/Endo.py',
    '../_base_/schedules/FFA_bs512_100e_coslr_warmup.py',
    '../_base_/default_runtime.py',
]

work_dir = './work_dirs/Endo_patient_disjoint_direct/Main_Comparison/Base'

model = dict(
    neck=dict(type='GlobalAveragePooling'),
    head=dict(task_heads=dict(classification=dict(
        type='MultiLabelRankClsHead',
        loss=dict(type='CrossEntropyLoss')))),
)

train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_evaluator = [dict(type='RankingROC_AUC', Validation_mode='original')]
test_evaluator = [
    dict(type='RankingROC_AUC', Validation_mode='original'),
    dict(type='MultiLabel_AveragePrecision', Validation_mode='original'),
    dict(type='RankingMultiLabelMetric', Validation_mode='original',
         items=['precision', 'recall', 'f1-score', 'support'], thr=0.5),
]
default_hooks = dict(checkpoint=dict(
    save_best='multi-class/mAUC_multiclass', rule='greater'))
