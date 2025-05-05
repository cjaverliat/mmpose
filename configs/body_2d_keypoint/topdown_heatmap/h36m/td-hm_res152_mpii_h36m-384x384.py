_base_ = ["../../../_base_/default_runtime.py"]

# runtime
train_cfg = dict(max_epochs=210, val_interval=10)

# optimizer
optim_wrapper = dict(
    optimizer=dict(
        type="Adam",
        lr=5e-4,
    )
)

# learning policy
param_scheduler = [
    dict(
        type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False
    ),  # warm-up
    dict(
        type="MultiStepLR",
        begin=0,
        end=210,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

n_keypoints = 17

# automatically scaling LR based on the actual training batch size
auto_scale_lr = dict(base_batch_size=512)

# hooks
default_hooks = dict(checkpoint=dict(save_best="PCK", rule="greater"))

# codec settings
codec = dict(type="MSRAHeatmap", input_size=(384, 384), heatmap_size=(96, 96), sigma=2)

# model settings
model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type="ResNet",
        depth=152,
        init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet152"),
    ),
    head=dict(
        type="HeatmapHead",
        deconv_out_channels=(256, 256, 256),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=None,
        in_channels=2048,
        out_channels=n_keypoints,
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=False,
        shift_heatmap=True,
    ),
)

# base dataset settings
dataset_type = "Human36mCocoDataset"
data_mode = "topdown"
data_root = "data/"

# pipelines
train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomBBoxTransform", shift_prob=0),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]
val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"]),
    dict(type="PackPoseInputs"),
]

# data loaders
train_dataloader = dict(
    batch_size=64,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="h36m/h36m_train_coco_10fps.json",
        data_prefix=dict(img="h36m/images/"),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    batch_size=32,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_mode=data_mode,
        ann_file="h36m/h36m_test_coco_10fps.json",
        data_prefix=dict(img="h36m/images/"),
        pipeline=val_pipeline,
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

test_dataloader = val_dataloader

# hooks
default_hooks = dict(
    checkpoint=dict(save_best="coco/AP", rule="greater", max_keep_ckpts=1, interval=1),
)

custom_hooks = [
    dict(
        type="EMAHook",
        ema_type="ExpMomentumEMA",
        momentum=0.0002,
        update_buffers=True,
        priority=49,
    ),
]

# evaluators
val_evaluator = dict(
    type="CocoMetric",
    ann_file=data_root + "h36m/h36m_test_coco_10fps.json",
)
test_evaluator = val_evaluator