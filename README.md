# Instance Segmentation on CARLA Data for 2025 Carleton College Comps Project

Forked from the [MMDetection](https://github.com/open-mmlab/mmdetection) repository, which is an open source object detection toolbox that provides access to many different models. 


We chose to adapt the implementation of [MaskTrackRCNN](./configs/masktrack_rcnn/README.md), which is a video instance segmentation model, that was originally trained on the [YoutubeVIS2021](https://youtube-vos.org/dataset/vis/) dataset, to be trained on a custom made dataset of image sequences from the simulator [CARLA](https://github.com/carla-simulator/carla) to investigate how different weather situations impacted the performance of the model and how we could mitigate those changes in performance.

## Contents

- [Environment Setup](#environment-setup)
- [Training](#training)
- [Visualization/Inference](#visualizationinference)


## Environment Setup

If desired, you can use the [original documentation](./docs/en/get_started.md) from OpenMMLab to go through the installation process. Otherwise, you can follow these simplified directions.

MMDetection works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.8+. We also had 1 GPU, though more would likely improve training and inference. MMDetection says that you can run their models on CPU only, but we did not try this.

Step 1: Install Miniconda

Step 2: Create and activate conda environment

```bash
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```

Step 3: Install dependencies

```bash
conda install pytorch torchvision -c pytorch
conda install fsspec 
pip install -U openmim
mim install mmengine
conda install cuda -c nvidia 
mim install mmcv=="2.1.0"
```

```bash
cd mmdetection
pip install -v -e .
pip install seaborn
```

## Training

To train the model, first you must have the correct video data. Go to our [CARLA data repo](https://github.com/Carleton-Comps-CV4AD/cv4ad_CARLA_code/tree/main) to produce that or email ___@gmail.com. 

Once you have the data, you must create the proper annotations for it...


Once your annotations are complete, go into the config of the model 
for the type of data that you want to train on, and change the data_root
variable at near the top of the file to point to the location of the
weather type within the video_data directory. For example, if I want
to train on clear day sequences, then set <code>data_root = 'path_to_data/video_data/clear_day'</code>.
 For training purposes, the <code>dataset_to_be_tested</code> variable inside each config does 
 not matter, so use any of the config files within that weather's directory 
 to train. 

Then, you can run <code>python tools/train.py [path to config]</code> 
and it will create a work directory where logs and checkpoints at the 
end of each epoch will be saved. With 1 GPU our models took about 5 
hours each to train, so it is likely best to set up a [screen]() 
session (if on linux) first so nothing will interrupt the training. 

Here is an example of what you would run if you wanted to train your 
model on clear day sequences:

```bash
python tools/train.py ./masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_clear_day.py
```

To train all on all 4 weather types, make sure this train command is 
repeated for clear_day, clear_night, rainy_day, and foggy_day by 
pointing to a config that corresponds to that weather type.

## Evaluation

To evaluate the performance of the model on a particular dataset, we have the configs set up to calculate the [COCO metrics](https://cocodataset.org/#detection-eval) for Average Precision and Average Recall. We primarily used Mean Average Precision to compare our models. 

Run <code>python tools/test_tracking.py [path to config file] --checkpoint  [path to checkpoint]</code> to calculate these metrics. It should take 3-5 minutes and the output will be put to the terminal unless a text file is specified. It is also important to note that the config now matters because each config for each weather has the <code>dataset_to_be_tested</code> variable set individually. The naming of each config tells you which dataset that config is set up to test on: masktrack-rcnn_carla_x_on_y.py where x is the training set and y is the test set. 

Here is an example of testing a model trained on clear day on foggy day scenes with the output to a text file:

```bash
python tools/test_tracking.py masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_foggy_day.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth > clear_day_on_foggy_day.txt
```

If you have all 4 model checkpoints trained for each weather and in the correct location (as specified in the script), you can also run this script to run all 16 tests (4 models on all 4 weather types):

```bash
./tools/masktrack-rcnn_test.sh
```


## Visualization/Inference

Step 1: create the mp4s

To visualize the output of each different model on the weather situations you need to have an mp4 (video) to give as input to the model instead of the sequences of 18 images that create a 3 second clip at 6 fps. We found the easiest way to do this was to use [ffmpeg](https://www.ffmpeg.org/documentation.html) to create an mp4 from a video sequence. Run <code> ffmpeg -framerate 6 -start_number [number of first filename in video sequence] -i <path_to_data>/video_data/<weather>/val/rgb/<video_number>/%04d.png -c:v mpeg4 -q:v 5 <output_file>.mp4 </code>

Here's an example of this command being run:

```bash
ffmpeg -framerate 6 -start_number 7993 -i /Data/video_data/rainy_day/val/rgb/video_149/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_149_rainy_day.mp4
```

We used this script to generate mp4s of 6 different videos for all 4 weathers. If you wish to use it, make sure the paths to the images and output paths are correct to your use case.

```bash
./tools/visualizations/video_conversion.sh
```

Step 2: run your model on the mp4

Run <code>  python demo/mot_demo.py [path to mp4]  [path to config] --checkpoint [path to checkpoint] --score-thr 0.5 --out [output file].mp4 </code> to output another mp4 with the instance segmentation masks and bounding boxes from the model visualized on the video.

An example of this full command of the model trained on clear day being visualized on video_49_clear_day is:

```bash
python demo/mot_demo.py /Data/video_data/videos/video_49_clear_day.mp4  masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day.py  --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth --score-thr 0.5 --out /Data/video_data/video_results/clear_day_on_video_49_clear_day.mp4
```

We used this script to create visualizations of all 4 models on all 6 videos in all 4 weather situations. Feel free to modify it to your file paths as necessary:

```bash
./tools/visualizations/model_vis.sh
```

Step 3 (optional): convert visualization mp4s into gifs

Run <code>ffmpeg -i [path to mp4] -vf fps=15,scale=1080:-1:flags=lanczos,split[s0][s1];[s0]palettegen=stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=1:diff_mode=rectangle [path to output file].gif
</code> to create a gif from the mp4.

If you are looking to convert many mp4s to gifs, you can use this script to generate the file path names for all the videos by modifying it to look in the correct directory of your videos:

```bash
./tools/video_filenames.sh
```

Then copy the list of video filenames into this script and run it to create all the gifs:

```bash
./tools/gif_conversion.sh
```










## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citations

See the paper for the object detection toolbox/frame work we utilized: [MMDetection](https://arxiv.org/pdf/1906.07155)

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

See the paper for the model we chose: [MaskTrack-RCNN](https://arxiv.org/pdf/1905.04804)

```
@inproceedings{yang2019video,
  title={Video instance segmentation},
  author={Yang, Linjie and Fan, Yuchen and Xu, Ning},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5188--5197},
  year={2019}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
