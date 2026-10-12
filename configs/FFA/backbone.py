## Efficientnetb4
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='EfficientNet', arch='b4',
#                   init_cfg=dict(
#                         checkpoint=
#                         'https://download.openmmlab.com/mmclassification/v0/efficientnet/efficientnet-b4_3rdparty_8xb32-aa_in1k_20220119-45b8bd2b.pth',
#                         prefix='backbone',
#                         type='Pretrained'),),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=6,
#         in_channels=1792,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         # topk=(1, 5),
#     ))

## ResNext50
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(
#         type='ResNeXt',
#         depth=50,
#         num_stages=4,
#         out_indices=(3, ),
#         groups=32,
#         width_per_group=4,
#         style='pytorch',
#         init_cfg=dict(
#             checkpoint=
#             'https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth',
#             prefix='backbone',
#             type='Pretrained'),),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='LinearClsHead',
#         num_classes=6,
#         in_channels=2048,
#         loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
#         # topk=(1, 5),
#     ))

## resnet50
model = dict(
    type='MultiLabelImageClassifier',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        init_cfg=dict(
            checkpoint='https://download.openmmlab.com/mmselfsup/1.x/swav/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96/swav_resnet50_8xb32-mcrop-2-6-coslr-200e_in1k-224-96_20220825-5b3fc7fc.pth',
            # 'https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth',
            prefix='backbone',
            type='Pretrained'),),
     head=dict(
        type='MultiLabelMultiTaskHead',  # Multitask head class
        task_heads={
            "classification": dict(
                num_classes=6,
                in_channels=2048,
            )
        },
    )
        )

####----swin transformer ----####
# model = dict(
#     type='MultiLabelImageClassifier',
#     backbone=dict(
#         type='SwinTransformer',
#         arch='small',  # Options: tiny, small, base, large
#         img_size=448,
#         patch_size=4,
#         drop_path_rate=0.2,
#         out_indices=(3,),
#         init_cfg=dict(type='Pretrained', 
#                       prefix='backbone',
#                       checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_small_patch4_window7_224-cc7a01c9.pth')
#     ),
#     head=dict(
#         type='MultiLabelMultiTaskHead',  # Multitask head class
#         task_heads={
#             "classification": dict(
#                 num_classes=6,
#                 in_channels=768,
#             )},
#             )
#         )
##### ------- ConvNeXt ------- ####
# model = dict(
#     type='MultiLabelImageClassifier',
#     backbone=dict(type='ConvNeXt', 
#                   arch='small',  # Options: tiny, small, base, large
#                   drop_path_rate=0.5,
#                   gap_before_final_norm=False,
#                   init_cfg=dict(type='Pretrained', 
#                       prefix='backbone',
#                       checkpoint='https://download.openmmlab.com/mmclassification/v0/convnext/convnext-small_32xb128_in1k_20221207-4ab7052c.pth')
#                   ),
#     head=dict(
#         type='MultiLabelMultiTaskHead',  # Multitask head class
#         task_heads={
#             "classification": dict(
#                 num_classes=6,
#                 in_channels=768,
#             )},
#             )
#         )


# InceptionV3
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='InceptionV3', num_classes=0, aux_logits=False,
#                    init_cfg=dict(
#                         checkpoint=
#                         'https://download.openmmlab.com/mmclassification/v0/inception-v3/inception-v3_3rdparty_8xb32_in1k_20220615-dcd4d910.pth',
#                         prefix='backbone',
#                         type='Pretrained'),),
#     # neck=None,
#     head=dict(
#             num_classes=6,
#             in_channels=2048,
#             )
# )


## VGG19
# model = dict(
#     type='ImageClassifier',
#     backbone=dict(type='VGG', depth=19, num_classes=0,
#                   init_cfg=dict(
#                         checkpoint=
#                         'https://download.openmmlab.com/mmclassification/v0/vgg/vgg19_batch256_imagenet_20210208-e6920e4a.pth',
#                         prefix='backbone',
#                         type='Pretrained'),),

#     head=dict(
#         num_classes=6,
#         in_channels=512,
#         ))

## DenseNet121
# model = dict(
#     type='MultiLabelImageClassifier',
#     backbone=dict(type='DenseNet', arch='121',
#                   init_cfg=dict(
#                         checkpoint=
#                         'https://download.openmmlab.com/mmclassification/v0/densenet/densenet121_4xb256_in1k_20220426-07450f99.pth',
#                         prefix='backbone',
#                         type='Pretrained'),),
#     # neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='MultiLabelMultiTaskHead',  # Multitask head class
#         task_heads={
#             "classification": dict(
#                 # Ranking = True,
#                 # Voting = 'smoe',
#                 # alpha = 0.1,
#                 num_classes=6,
#                 in_channels=1024,
                
#                 # Voting = dict(average=True),num_class=1),)
#             ),

#             # "Voting": dict(
#             #     type='MultiLabelVoteHead',
#             #     num_classes=1,
#             #     in_channels=1024,
#             #     # loss=dict(type='FocalLoss', loss_weight=1.0),
#             #     # topk=1,
#             #     # average=True,
#             #     # in_channels=2048,
#             # ),    
#         },
#     ))
## DINOV2
# model = dict(
#     type='MultiLabelImageClassifier',
#      backbone=dict(
#         type='VisionTransformer',
#         arch='base',
#         out_type='featmap',
#         img_size=448,
#         patch_size=14,
#         layer_scale_init_value=1e-5,
#         init_cfg=dict(
#             checkpoint=
#             'https://download.openmmlab.com/mmpretrain/v1.0/dinov2/vit-base-p14_dinov2-pre_3rdparty_20230426-ba246503.pth',
#             prefix='backbone',
#             type='Pretrained'),),
#     neck=dict(type='GlobalAveragePooling'),
#     head=dict(
#         type='MultiLabelMultiTaskHead',  # Multitask head class
#         task_heads={
#             "classification": dict(
#                 # Ranking = True,
#                 num_classes=6,
#                 in_channels=768,
#             ),
#         },
#     ))