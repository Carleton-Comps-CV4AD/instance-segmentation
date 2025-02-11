_base_ = ['./masktrack-rcnn_mask-rcnn_r50_fpn_8xb1-12e_youtubevis2019.py']

# data_root = 'data/youtube_vis_2021/'
# data_root = '//Data/meierj/YoutubeVIS2021/'
data_root = '/Data/video_data/clear_day' 

dataset_version = data_root[-5:-1]

# dataloader
# train_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         dataset_version=dataset_version,
#         ann_file='annotations/youtube_vis_2021_train.json'))
# val_dataloader = dict(
#     dataset=dict(
#         data_root=data_root,
#         dataset_version=dataset_version,
#         ann_file='annotations/youtube_vis_2021_valid.json'))
# test_dataloader = val_dataloader


model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=8), mask_head=dict(num_classes=8))) # set number of classes correctly?? 

# Modify dataset related settings
metainfo = {
    'classes': ('pedestrian', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'), # add classes taken from annotations
    # 'palette': [ # dont worry about this yet
    #     (220, 20, 60),
    # ]
}




train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations.json',
        data_prefix=dict(img='train/rgb/'))) 
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations.json',
        data_prefix=dict(img='val/rgb'))) 
test_dataloader = val_dataloader

# Modify metric related settings
val_evaluator = dict(ann_file=data_root + 'val/annotation_coco.json')
test_evaluator = val_evaluator
