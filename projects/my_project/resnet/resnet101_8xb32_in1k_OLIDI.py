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
#dataset settings
dataset_type = 'CustomDataset'
data_preprocessor = dict(
    num_classes=num_classes,
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]

train_dataloader = dict(
    batch_size=64,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=train_pipeline,
        data_prefix='img',
        classes=classes,
        ann_file="meta/train.txt"
        ),
    num_workers=16,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        pipeline=test_pipeline,
        data_prefix='img',
        classes=classes,
        ann_file="meta/val.txt"
        ),
    num_workers=16,
    sampler=dict(shuffle=False, type='DefaultSampler'))

val_evaluator = dict(
    topk=(1,5), type='Accuracy')

test_dataloader = val_dataloader
test_evaluator = val_evaluator
#dataset settings



#model settings
model = dict(
    backbone=dict(
        depth=101,
        num_stages=4,
        out_indices=(3, ),
        style='pytorch',
        type='ResNet'),
    head=dict(
        in_channels=2048,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=num_classes,
        topk=(1),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier',
    init_cfg=dict(type='Pretrained', checkpoint='https://download.openmmlab.com/mmclassification/v0/resnet/resnet101_8xb32_in1k_20210831-539c63f8.pth'))
#model settings

#schedules settings

#optimizer
optim_wrapper = dict(
    optimizer=dict(lr=0.1, momentum=0.9, type='SGD', weight_decay=0.0001))

#learning policy
param_scheduler = dict(
    by_epoch=True, gamma=0.1, milestones=[
        30,
        60,
        90,
    ], type='MultiStepLR')

#train, val ,test setting
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=64)

#schedules settings


#runtime settings
default_scope = 'mmpretrain'
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(interval=1, type='CheckpointHook', save_best='auto', rule='greater',max_keep_ckpts=2),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', 
    vis_backends=vis_backends,
)
log_level = 'INFO'

load_from = None
resume = False

randomness = dict(deterministic=False, seed=22)