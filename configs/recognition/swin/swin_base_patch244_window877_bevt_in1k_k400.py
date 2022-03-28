_base_ = [
    '../../_base_/default_runtime.py'
]

# encoder.pkl saved under tokenizer_path
tokenizer_path = '/path/to/save/dall_e_tokenizer_weight'

model=dict(type='BEVTMultiSource',
           backbone=dict(type='SwinTransformerBEVT',
                         patch_size=(2,4,4),
                         depths=[2, 2, 18, 2],
                         embed_dim=128,
                         num_heads=[4, 8, 16, 32],
                         window_size=(8,7,7),
                         mlp_ratio=4.,
                         qkv_bias=True,
                         qk_scale=None,
                         drop_rate=0.,
                         attn_drop_rate=0.,
                         patch_norm=True,
                         drop_path_rate=0.3),
           test_cfg=dict(max_testing_views=4),
           cls_head=dict(type='BEVTMultiSourceHead', num_classes=8192, in_channels=1024, vae_weight_path=tokenizer_path),
           )
omnisource = True

# dataset settings
dataset_type = 'VideoDataset'
data_root = 'DATASET/videodata/kinetics/kinetics400_256/train_256'
data_root_val = 'DATASET/videodata/kinetics/kinetics400_256/val_256'
ann_file_train = 'DATASET/videodata/kinetics/kinetics400_256/train_256.txt'
ann_file_val = 'DATASET/videodata/kinetics/kinetics400_256/val_256.txt'
ann_file_test = 'DATASET/videodata/kinetics/kinetics400_256/val_256.txt'

image_data_root = 'DATASET/imagenet/train'
image_ann_file_train = 'ILSVRC2012_name_train_list.txt'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
video_train_pipeline = [
    dict(type='DecordInit'),
    dict(type='SampleFrames', clip_len=16, frame_interval=8, num_clips=1),
    dict(type='DecordDecode'),
    dict(type='RandomRescale', scale_range=(256, 320), interpolation="bicubic"),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='CopyResize', new_keys='imgs_second', scale=(112, 112), interpolation="lanczos"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShapeByKey', input_format='NCHW', format_key='imgs_second'),
    dict(type='FormatShape', input_format='NCTHW'),
    dict(type='MaskedPositionGenerate', input_size=14,
         num_masking_patches=98, min_num_patches=16, max_num_patches=None,
         masked_tube_range=(0.5, 1.0), temporal_scale=2),
    dict(type='Collect', keys=['imgs', 'imgs_second', 'input_position_masks', 'output_position_masks'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_second', 'input_position_masks', 'output_position_masks']),
    dict(type='MapPixels', keys=['imgs_second'])
]
image_train_pipeline = [
    dict(type='ImageDecode'),
    dict(type='RandomRescale', scale_range=(256, 320), interpolation="bicubic"),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0., frame_independent=True),
    dict(type='RandomCrop', size=224),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='CopyResize', new_keys='imgs_second', scale=(112, 112), interpolation="lanczos"),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShapeByKey', input_format='NCHW', format_key='imgs_second'),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Masked2DPositionGenerate', input_size=14,
         num_masking_patches=98, min_num_patches=16, max_num_patches=None),
    dict(type='Collect', keys=['imgs', 'imgs_second', 'input_position_masks', 'output_position_masks'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'imgs_second', 'input_position_masks', 'output_position_masks']),
    dict(type='MapPixels', keys=['imgs_second'])
]
data = dict(
    videos_per_gpu=8,
    omni_videos_per_gpu=[8, 64],
    workers_per_gpu=4,
    data_type=['video', 'image'],
    train=[
        dict(
            type=dataset_type,
            ann_file=ann_file_train,
            data_prefix=data_root,
            pipeline=video_train_pipeline
        ),
        dict(
            type='RepeatDataset',
            times=5,
            dataset=dict(
                type='ImageDataset',
                ann_file=image_ann_file_train,
                data_prefix=image_data_root,
                pipeline=image_train_pipeline
            )
        ),
    ],
)
evaluation = dict(
    interval=1, metrics=['top_k_accuracy', 'mean_class_accuracy'], gpu_collect=True)

# optimizer
optimizer = dict(type='AdamW', lr=5e-4, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    by_epoch=False,
    min_lr=0,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=10,
)
total_epochs = 150

# runtime settings
checkpoint_config = dict(interval=10, create_symlink=False)
auto_resume=True
work_dir = 'OUTPUT/swin_base_bevt_twostream'
find_unused_parameters = True

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=2,
    grad_clip=dict(max_norm=3.0, norm_type=2),
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)
