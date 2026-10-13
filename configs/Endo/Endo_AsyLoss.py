_base_ = ['./Endo_baseline.py']

work_dir = './work_dirs/Endo_patient_disjoint_direct/Main_Comparison/Asy'

model = dict(head=dict(task_heads=dict(classification=dict(
    loss=dict(_delete_=True, type='AsymmetricLoss')))))

