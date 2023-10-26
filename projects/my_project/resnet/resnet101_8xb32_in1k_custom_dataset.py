num_classes=30

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
        data_root='/home/ypx/project/ljn/mmpretrain/dataset/Custom Dataset',
        pipeline=train_pipeline,
        data_prefix='img',
        classes=['Blackgram__Anthracnose', 'Blackgram__Healthy', 'Blackgram__Leaf Crinckle', 'Blackgram__Powdery Mildew', 'Blackgram__Yellow Mosaic', 'Cucumber__Anthracnose', 'Cucumber__Bacterial Wilt', 'Cucumber__Belly Rot', 'Cucumber__Downy Mildew', 'Cucumber__Fresh Cucumber', 'Cucumber__Fresh Leaf', 'Cucumber__Gummy Stem Blight', 'Cucumber__Pythium Fruit Rot', 'MangoLeaf__Anthracnose', 'MangoLeaf__Bacterial Canker', 'MangoLeaf__Cutting Weevil', 'MangoLeaf__Die Back', 'MangoLeaf__Gall Midge', 'MangoLeaf__Healthy', 'MangoLeaf__Powdery Mildew', 'MangoLeaf__Sooty Mould', 'Sugarcane__Healthy', 'Sugarcane__Mosaic', 'Sugarcane__RedRot', 'Sugarcane__Rust', 'Sugarcane__Yellow', 'Watermelon__Anthracnose', 'Watermelon__Downy_Mildew', 'Watermelon__Healthy', 'Watermelon__Mosaic_Virus'],
        ann_file="meta/train.txt"
        ),
    num_workers=16,
    sampler=dict(shuffle=True, type='DefaultSampler'))

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        type=dataset_type,
        data_root='/home/ypx/project/ljn/mmpretrain/dataset/Custom Dataset',
        pipeline=test_pipeline,
        data_prefix='img',
        classes=['Blackgram__Anthracnose', 'Blackgram__Healthy', 'Blackgram__Leaf Crinckle', 'Blackgram__Powdery Mildew', 'Blackgram__Yellow Mosaic', 'Cucumber__Anthracnose', 'Cucumber__Bacterial Wilt', 'Cucumber__Belly Rot', 'Cucumber__Downy Mildew', 'Cucumber__Fresh Cucumber', 'Cucumber__Fresh Leaf', 'Cucumber__Gummy Stem Blight', 'Cucumber__Pythium Fruit Rot', 'MangoLeaf__Anthracnose', 'MangoLeaf__Bacterial Canker', 'MangoLeaf__Cutting Weevil', 'MangoLeaf__Die Back', 'MangoLeaf__Gall Midge', 'MangoLeaf__Healthy', 'MangoLeaf__Powdery Mildew', 'MangoLeaf__Sooty Mould', 'Sugarcane__Healthy', 'Sugarcane__Mosaic', 'Sugarcane__RedRot', 'Sugarcane__Rust', 'Sugarcane__Yellow', 'Watermelon__Anthracnose', 'Watermelon__Downy_Mildew', 'Watermelon__Healthy', 'Watermelon__Mosaic_Virus'],
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