_base_ = ['./Endo_baseline.py']

work_dir = './work_dirs/Endo_patient_disjoint_direct/Main_Comparison/MLDecoder'

model = dict(
    neck=None,
    head=dict(task_heads=dict(classification=dict(
        _delete_=True,
        type='Multilabel_MLdecoderHead', num_classes=4, in_channels=2048,
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True)))),
)
model_wrapper_cfg = dict(type='MMDistributedDataParallel', find_unused_parameters=True)

