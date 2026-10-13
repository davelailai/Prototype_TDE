_base_ = ['./Endo_baseline.py']

work_dir = './work_dirs/Endo_patient_disjoint_direct/Main_Comparison/Q2L'

model = dict(
    neck=None,
    head=dict(task_heads=dict(classification=dict(
        _delete_=True,
        type='Multilabel_Qeruy2Label', num_classes=4, in_channels=2048,
        position_embedding='sine', enc_layers=1, dec_layers=2,
        hidden_dim=512, nheads=8, img_size=448,
        loss=dict(type='AsymmetricLoss', loss_weight=1.0, use_sigmoid=True)))),
)
