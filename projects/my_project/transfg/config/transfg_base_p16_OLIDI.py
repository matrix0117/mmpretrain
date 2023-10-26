num_classes=57
classes=['ash_gourd__K', 'ash_gourd__K_Mg', 'ash_gourd__N', 'ash_gourd__N_K', 'ash_gourd__N_Mg', 'ash_gourd__PM', 'ash_gourd__healthy',
          'bitter_gourd__DM', 'bitter_gourd__JAS', 'bitter_gourd__K', 'bitter_gourd__K_Mg', 'bitter_gourd__LS', 'bitter_gourd__N', 'bitter_gourd__N_K', 'bitter_gourd__N_Mg', 'bitter_gourd__healthy', 
          'bottle_gourd__DM', 'bottle_gourd__JAS', 'bottle_gourd__JAS_MIT', 'bottle_gourd__K', 'bottle_gourd__LS', 'bottle_gourd__N', 'bottle_gourd__N_K', 'bottle_gourd__N_Mg', 'bottle_gourd__healthy', 
          'cucumber__K', 'cucumber__N', 'cucumber__N_K', 'cucumber__healthy', 
          'eggplant__EB', 'eggplant__FB', 'eggplant__JAS', 'eggplant__K', 'eggplant__MIT', 'eggplant__MIT_EB', 'eggplant__N', 'eggplant__N_K', 'eggplant__healthy', 
          'ridge_gourd__N', 'ridge_gourd__N_Mg', 'ridge_gourd__PC', 'ridge_gourd__PLEI', 'ridge_gourd__PLEI_IEM', 'ridge_gourd__PLEI_MIT', 'ridge_gourd__healthy', 
          'snake_gourd__K', 'snake_gourd__LS', 'snake_gourd__N', 'snake_gourd__N_K', 'snake_gourd__healthy', 
          'tomato__JAS_MIT', 'tomato__K', 'tomato__LM', 'tomato__MIT', 'tomato__N', 'tomato__N_K', 'tomato__healthy']
data_root='/home/ypx/project/ljn/mmpretrain/dataset/OLID I'
batch_size=32

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='TransFG',
        arch='b',
        img_size=224,
        patch_size=16,
        drop_rate=0.1,
        init_cfg=[
            dict(
                type='Kaiming',
                layer='Conv2d',
                mode='fan_in',
                nonlinearity='linear')
        ],
        patch_cfg=dict(stride=12,padding=0)),
    neck=None,
    head=dict(
        type='TransFGClsHead',
        num_classes=num_classes,
        in_channels=768,
        loss=dict(type='TransFGLoss'),
    ),
     init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/vit/vit-base-p16_pt-32xb128-mae_in1k_20220623-4c544545.pth')
)


# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', scale=224, backend='pillow'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeEdge', scale=256, edge='short', backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=24,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='img',
        classes=classes,
        ann_file="meta/train.txt",
        pipeline=train_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=24,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix='img',
        classes=classes,
        ann_file="meta/val.txt",
        pipeline=test_pipeline),
    sampler=dict(type='DefaultSampler', shuffle=False),
)
val_evaluator = dict(type='Accuracy', topk=(1, 5))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator


# optimizer
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=0.003, weight_decay=0.3),
    # specific to vit pretrain
    paramwise_cfg=dict(custom_keys={
        '.cls_token': dict(decay_mult=0.0),
        '.pos_embed': dict(decay_mult=0.0)
    }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-4,
        by_epoch=True,
        begin=0,
        end=30,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(
        type='CosineAnnealingLR',
        T_max=270,
        by_epoch=True,
        begin=30,
        end=300,
    )
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=batch_size)


# defaults to use registries in mmpretrain
default_scope = 'mmpretrain'

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type='IterTimerHook'),

    # print log every 100 iterations.
    logger=dict(type='LoggerHook', interval=100),

    # enable the parameter scheduler.
    param_scheduler=dict(type='ParamSchedulerHook'),

    # save checkpoint per epoch.
    checkpoint=dict(interval=1, type='CheckpointHook', save_best='auto', rule='greater',max_keep_ckpts=2),

    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type='DistSamplerSeedHook'),

    # validation results visualization, set True to enable it.
    visualization=dict(type='VisualizationHook', enable=True),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,

    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),

    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(type='UniversalVisualizer', vis_backends=vis_backends)

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = '/home/ypx/project/ljn/mmpretrain/work_dirs/transfg_base_p16_OLIDI/best_accuracy_top1_epoch_192.pth'

# whether to resume training from the loaded checkpoint
resume = True

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=22, deterministic=False)

