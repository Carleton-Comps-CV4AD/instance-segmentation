# Instance Segmentation on CARLA Data for 2025 Carleton College Comps Project

Forked from the [MMDetection](https://github.com/open-mmlab/mmdetection) repository, which is an open source object detection toolbox that provides access to many different models. 


We chose to adapt the implementation of [MaskTrackRCNN](./configs/masktrack_rcnn/README.md), which is a video instance segmentation model, that was originally trained on the [YoutubeVIS2021](https://youtube-vos.org/dataset/vis/) dataset, to be trained on a custom made dataset of image sequences from the simulator [CARLA](https://github.com/carla-simulator/carla) to investigate how different weather situations impacted the performance of the model and how we could mitigate those changes in performance.



## Environment Setup

If desired, you can use the [original documentation](./docs/en/get_started.md) from OpenMMLab to go through the installation process. Otherwise, you can follow these simplified directions.

MMDetection works on Linux, Windows, and macOS. It requires Python 3.7+, CUDA 9.2+, and PyTorch 1.8+. We also had 1 GPU, though more would likely improve training and inference. MMDetection says that you can run their models on CPU, but we did not try this.

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

To train the model, first you must have the correct video data. Go to our [CARLA data repo]() to produce that or email ___@gmail.com. 

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


```bash
python tools/test_tracking.py [path to config file] --checkpoint  [path to checkpoint]
```


## Visualization/Inference











## Acknowledgement

MMDetection is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and develop their own new detectors.

## Citations

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
