backend_args = None
dataset_to_be_tested = '/Data/video_data/rainy_day/'
data_root = '/Data/video_data/clear_day/'
dataset_type = 'YouTubeVISDataset'
dataset_version = '_day'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(draw=False, type='TrackVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = 'work_dirs/masktrack-rcnn_carla_clear_day/epoch_12.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
metainfo = dict(
    classes=(
        'pedestrian',
        'rider',
        'car',
        'truck',
        'bus',
        'train',
        'motorcycle',
        'bicycle',
    ))
model = dict(
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_mask=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='TrackDataPreprocessor'),
    detector=dict(
        backbone=dict(
            depth=50,
            frozen_stages=1,
            init_cfg=dict(
                checkpoint='torchvision://resnet50', type='Pretrained'),
            norm_cfg=dict(requires_grad=True, type='BN'),
            norm_eval=True,
            num_stages=4,
            out_indices=(
                0,
                1,
                2,
                3,
            ),
            style='pytorch',
            type='ResNet'),
        init_cfg=dict(
            checkpoint=
            'https://download.openmmlab.com/mmdetection/v2.0/mask_rcnn/mask_rcnn_r50_fpn_1x_coco/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth',
            type='Pretrained'),
        neck=dict(
            in_channels=[
                256,
                512,
                1024,
                2048,
            ],
            num_outs=5,
            out_channels=256,
            type='FPN'),
        roi_head=dict(
            bbox_head=dict(
                bbox_coder=dict(
                    target_means=[
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ],
                    target_stds=[
                        0.1,
                        0.1,
                        0.2,
                        0.2,
                    ],
                    type='DeltaXYWHBBoxCoder'),
                fc_out_channels=1024,
                in_channels=256,
                loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
                loss_cls=dict(
                    loss_weight=1.0,
                    type='CrossEntropyLoss',
                    use_sigmoid=False),
                num_classes=40,
                reg_class_agnostic=False,
                roi_feat_size=7,
                type='Shared2FCBBoxHead'),
            bbox_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=7, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            mask_head=dict(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=dict(
                    loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
                num_classes=40,
                num_convs=4,
                type='FCNMaskHead'),
            mask_roi_extractor=dict(
                featmap_strides=[
                    4,
                    8,
                    16,
                    32,
                ],
                out_channels=256,
                roi_layer=dict(
                    output_size=14, sampling_ratio=0, type='RoIAlign'),
                type='SingleRoIExtractor'),
            type='StandardRoIHead'),
        rpn_head=dict(
            anchor_generator=dict(
                ratios=[
                    0.5,
                    1.0,
                    2.0,
                ],
                scales=[
                    8,
                ],
                strides=[
                    4,
                    8,
                    16,
                    32,
                    64,
                ],
                type='AnchorGenerator'),
            bbox_coder=dict(
                target_means=[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                target_stds=[
                    1.0,
                    1.0,
                    1.0,
                    1.0,
                ],
                type='DeltaXYWHBBoxCoder'),
            feat_channels=256,
            in_channels=256,
            loss_bbox=dict(loss_weight=1.0, type='L1Loss'),
            loss_cls=dict(
                loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
            type='RPNHead'),
        test_cfg=dict(
            rcnn=dict(
                mask_thr_binary=0.5,
                max_per_img=100,
                nms=dict(iou_threshold=0.5, type='nms'),
                score_thr=0.01),
            rpn=dict(
                max_per_img=200,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=200)),
        train_cfg=dict(
            rcnn=dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=128,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            rpn=dict(
                allowed_border=-1,
                assigner=dict(
                    ignore_iof_thr=-1,
                    match_low_quality=True,
                    min_pos_iou=0.3,
                    neg_iou_thr=0.3,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=False,
                    neg_pos_ub=-1,
                    num=64,
                    pos_fraction=0.5,
                    type='RandomSampler')),
            rpn_proposal=dict(
                max_per_img=200,
                min_bbox_size=0,
                nms=dict(iou_threshold=0.7, type='nms'),
                nms_pre=200)),
        type='MaskRCNN'),
    track_head=dict(
        embed_head=dict(
            fc_out_channels=1024,
            in_channels=256,
            num_fcs=2,
            roi_feat_size=7,
            type='RoIEmbedHead'),
        roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='SingleRoIExtractor'),
        train_cfg=dict(
            assigner=dict(
                ignore_iof_thr=-1,
                match_low_quality=True,
                min_pos_iou=0.5,
                neg_iou_thr=0.5,
                pos_iou_thr=0.5,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=True,
                neg_pos_ub=-1,
                num=128,
                pos_fraction=0.25,
                type='RandomSampler')),
        type='RoITrackHead'),
    tracker=dict(
        match_weights=dict(det_label=10.0, det_score=1.0, iou=2.0),
        num_frames_retain=20,
        type='MaskTrackRCNNTracker'),
    type='MaskTrackRCNN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=35, norm_type=2),
    optimizer=dict(lr=0.00125, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=False,
        end=500,
        start_factor=0.3333333333333333,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=12,
        gamma=0.1,
        milestones=[
            8,
            11,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/rgb', img_path='val/rgb'),
        data_root=dataset_to_be_tested,
        dataset_version='2019',
        metainfo=dict(
            classes=(
                'pedestrian',
                'rider',
                'car',
                'truck',
                'bus',
                'train',
                'motorcycle',
                'bicycle',
            )),
        pipeline=[
            dict(
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        640,
                        360,
                    ), type='Resize'),
                    dict(type='LoadTrackAnnotations', with_mask=True),
                ],
                type='TransformBroadcaster'),
            dict(type='PackTrackInputs'),
        ],
        test_mode=True,
        type='YouTubeVISDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        format_only=False,
        metric=[
            'bbox',
            'segm',
            'proposal',
        ],
        outfile_prefix='./coco_metric',
        type='CocoVideoMetric',
        classwise=True),
]
test_pipeline = [
    dict(
        transforms=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                360,
            ), type='Resize'),
            dict(type='LoadTrackAnnotations', with_mask=True),
        ],
        type='TransformBroadcaster'),
    dict(type='PackTrackInputs'),
]
train_cfg = dict(max_epochs=12, type='EpochBasedTrainLoop', val_begin=13)
train_dataloader = dict(
    batch_sampler=dict(type='TrackAspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        ann_file='train/annotations.json',
        data_prefix=dict(img='train/rgb', img_path='train/rgb'),
        data_root='/Data/video_data/clear_day/',
        dataset_version='2019',
        metainfo=dict(
            classes=(
                'pedestrian',
                'rider',
                'car',
                'truck',
                'bus',
                'train',
                'motorcycle',
                'bicycle',
            )),
        pipeline=[
            dict(
                filter_key_img=True,
                frame_range=100,
                num_ref_imgs=1,
                type='UniformRefFrameSample'),
            dict(
                share_random_params=True,
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(type='LoadTrackAnnotations', with_mask=True),
                    dict(keep_ratio=True, scale=(
                        640,
                        360,
                    ), type='Resize'),
                    dict(prob=0.5, type='RandomFlip'),
                ],
                type='TransformBroadcaster'),
            dict(type='PackTrackInputs'),
        ],
        type='YouTubeVISDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='TrackImgSampler'))
train_pipeline = [
    dict(
        filter_key_img=True,
        frame_range=100,
        num_ref_imgs=1,
        type='UniformRefFrameSample'),
    dict(
        share_random_params=True,
        transforms=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadTrackAnnotations', with_mask=True),
            dict(keep_ratio=True, scale=(
                640,
                360,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
        ],
        type='TransformBroadcaster'),
    dict(type='PackTrackInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='val/annotations.json',
        data_prefix=dict(img='val/rgb', img_path='val/rgb'),
        data_root='/Data/video_data/clear_day/',
        dataset_version='2019',
        metainfo=dict(
            classes=(
                'pedestrian',
                'rider',
                'car',
                'truck',
                'bus',
                'train',
                'motorcycle',
                'bicycle',
            )),
        pipeline=[
            dict(
                transforms=[
                    dict(backend_args=None, type='LoadImageFromFile'),
                    dict(keep_ratio=True, scale=(
                        640,
                        360,
                    ), type='Resize'),
                    dict(type='LoadTrackAnnotations', with_mask=True),
                ],
                type='TransformBroadcaster'),
            dict(type='PackTrackInputs'),
        ],
        test_mode=True,
        type='YouTubeVISDataset'),
    drop_last=False,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(round_up=False, shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    format_only=True,
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_results',
    type='YouTubeVISMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='TrackLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = './work_dirs/masktrack-rcnn_carla_clear_day'
