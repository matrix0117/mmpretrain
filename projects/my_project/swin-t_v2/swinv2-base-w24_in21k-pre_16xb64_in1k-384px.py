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
# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='SwinTransformerV2',
        arch='base',
        img_size=256,
        drop_path_rate=0.5,
        window_size=[16, 16, 16, 8]
        ),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=num_classes,
        in_channels=1024,
        init_cfg=None,  # suppress the default init_cfg of LinearClsHead.
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1, mode='original'),
        cal_acc=False),
        init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.),
        dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/swin-v2/swinv2-base-w16_in21k-pre_3rdparty_in1k-256px_20220803-8d7aa8ad.pth')
    ],
    train_cfg=dict(augments=[
        dict(type='Mixup', alpha=0.8),
        dict(type='CutMix', alpha=1.0)
    ])    
)

# dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    # convert image from BGR to RGB
    to_rgb=False,
)

bgr_mean = data_preprocessor['mean'][::-1]
bgr_std = data_preprocessor['std'][::-1]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='RandomResizedCrop',
        scale=256,
        backend='pillow',
        interpolation='bicubic'),
    dict(type='RandomFlip', prob=0.5, direction='horizontal'),
    dict(
        type='RandAugment',
        policies='timm_increasing',
        num_policies=2,
        total_level=10,
        magnitude_level=9,
        magnitude_std=0.5,
        hparams=dict(
            pad_val=[round(x) for x in bgr_mean], interpolation='bicubic')),
    dict(
        type='RandomErasing',
        erase_prob=0.25,
        mode='rand',
        min_area_ratio=0.02,
        max_area_ratio=1 / 3,
        fill_color=bgr_mean,
        fill_std=bgr_std),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='ResizeEdge',
        scale=292,  # ( 256 / 224 * 256 )
        edge='short',
        backend='pillow',
        interpolation='bicubic'),
    dict(type='CenterCrop', crop_size=256),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=16,
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
    batch_size=16,
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


# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=5e-4 * 32 / 512,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            '.absolute_pos_embed': dict(decay_mult=0.0),
            '.relative_position_bias_table': dict(decay_mult=0.0)
        }),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=True,
        end=20,
        # update by iter
        convert_to_iter_based=True),
    # main learning rate scheduler
    dict(type='CosineAnnealingLR', eta_min=1e-5, by_epoch=True, begin=20)
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=32)


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
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=22, deterministic=False)


