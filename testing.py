from mmdet.apis import DetInferencer

# Initialize the DetInferencer
# inferencer = DetInferencer('rtmdet_tiny_8xb32-300e_coco')
# inferencerYolact = DetInferencer('yolact_r101_1x8_coco')
# # inferencerByteTrack = DetInferencer('bytetrack_yolox_x_8xb4-amp-80e_crowdhuman-mot17halftrain_test-mot17halfval', weights='bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth')

# models = DetInferencer.list_models('mmdet')

# for model in models:
#     if 'bytetrack' in model:
#         print(model)

# # # Perform inference
# # inferencer('demo/demo.jpg', out_dir='outputs/', no_save_pred=False)
# inferencerYolact('demo/large_image.jpg', out_dir='outputs/', no_save_pred=False)

# # inferencerByteTrack('demo/demo.jpg', out_dir='outputBT/', no_save_pred=False)




inferencer = DetInferencer(model='work_dirs/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day.py', weights='work_dirs/masktrack-rcnn_carla_clear_day/epoch_12.pth')


inferencer('/Data/januaryData/1_17_clear_day/_outRaw/0.png', out_dir='outputs/', no_save_pred=False)