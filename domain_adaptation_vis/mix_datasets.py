import os
import random
import math
import shutil
import json

path_to_weathers = '/Data/video_data'

def mix_data(splits):
    '''
    the idea here is that we shall go t the paths,
    list out all of the videos in rgb and shuffle the list, pick the first x, where x
    is the split corresponding to that weather type

    we shall then copy all the videos to the new folder as well as their annotations
    (when moving annotations keep in mind both the videos, images, and instances[annoations])

    once all of the above is made, we should have a dataset reasdy to be trained on.

    '''

    # assert(len(weathers) == len(splits))

    weathers = splits.keys()
    partitions = splits.values()
    total = 0 # total partition
    cur_videos = {
        'train' : [],
        'val': []
    }

    for partition in partitions:
        total += partition

    new_folder = str.join('_', weathers)

    for split in partitions:
        new_folder += ('_' + str(split))

    new_path = os.path.join(path_to_weathers, new_folder)

    os.makedirs(new_path, exist_ok=True)

    # weather_folders = os.listdir(path_to_weathers)

    for weather, partition in splits.items():
        move_data(weather, partition, total, new_path, cur_videos)


def move_data(weather, partition, total, new_path, cur_videos):

    weather_path = os.path.join(path_to_weathers, weather)

    splits = ['train', 'val']

    for split in splits:
        # videos_present = cur_videos[split]
        
        videos_list = []
        path = os.path.join(weather_path, split, 'rgb')

        # now we want to move both the rgb and the annoatations
        videos = os.listdir(path)
        random.shuffle(videos)

        index = math.ceil((partition/total) * len(videos))
        videos_to_be_added = []

        # make sure that videos added are not the same that are already present
        j = 0
        for i in range(index+1):
            try:
                while videos[j] in cur_videos[split]:
                    j += 1
                videos_to_be_added.append(videos[j])
                cur_videos[split].append(videos[j])
            except IndexError:
                pass

        # print(j)
        
        new_videos_folder = os.path.join(new_path, split)
        os.makedirs(new_videos_folder, exist_ok=True)

        print(videos_to_be_added)

        for video in videos_to_be_added:
            source = os.path.join(path_to_weathers, weather, split, 'rgb', video)
            destination = os.path.join(new_videos_folder, 'rgb', video)

            shutil.copytree(source, destination)

            videos_list.append(video)


        # move annotations that are part of the video
        ann_folder = os.path.join(path_to_weathers, weather, split, 'annotations.json')
        new_ann_folder = os.path.join(new_videos_folder, 'annotations.json')


        # time to do annotations

        # load from original version
        with open(ann_folder) as file:
            ann = json.load(file)


        # create/load new annotations dictionary for the given split
        if os.path.exists(new_ann_folder):
            with open(new_ann_folder) as file:
                annotations = json.load(file)
        else:
            # os.makedirs(new_ann_folder)
            annotations = {
            'categories' : [],
            'videos' : [],
            'images' : [],
            'annotations' : [],
            }
            annotations['categories'] = ann['categories']

        video_ann = ann['videos']
        imgs_ann = ann['images']
        annotations_ann = ann['annotations']

        video_ids = []

        for vid in video_ann:
            if vid['name'] in videos_list:
                annotations['videos'].append(vid)
                video_ids.append(vid['id'])

        for img in imgs_ann:
            if img['video_id'] in video_ids:
                annotations['images'].append(img)

        for annotation in annotations_ann:
            if annotation['video_id'] in video_ids:
                annotations['annotations'].append(annotation)

        with open(new_ann_folder, "w") as file:
            json.dump(annotations, file, indent = 4)

        # cur_videos[split].extend(videos_list)
            

if __name__ == '__main__':

    splits = {
        # weather : percentage
        'clear_day' : 70,
        'foggy_day' : 30
    }

    mix_data(splits)