#!/bin/bash

# Video files
video_files=(
  "/Data/video_data/videos/video_149_clear_day.mp4"
  "/Data/video_data/videos/video_149_clear_night.mp4"
  "/Data/video_data/videos/video_149_foggy_day.mp4"
  "/Data/video_data/videos/video_149_rainy_day.mp4"
  "/Data/video_data/videos/video_222_clear_day.mp4"
  "/Data/video_data/videos/video_222_clear_night.mp4"
  "/Data/video_data/videos/video_222_foggy_day.mp4"
  "/Data/video_data/videos/video_222_rainy_day.mp4"
  "/Data/video_data/videos/video_49_clear_day.mp4"
  "/Data/video_data/videos/video_49_clear_night.mp4"
  "/Data/video_data/videos/video_49_foggy_day.mp4"
  "/Data/video_data/videos/video_49_rainy_day.mp4"
  "/Data/video_data/videos/video_77_clear_day.mp4"
  "/Data/video_data/videos/video_77_clear_night.mp4"
  "/Data/video_data/videos/video_77_foggy_day.mp4"
  "/Data/video_data/videos/video_77_rainy_day.mp4"
  "/Data/video_data/videos/video_331_clear_day.mp4"
  "/Data/video_data/videos/video_331_clear_night.mp4"
  "/Data/video_data/videos/video_331_foggy_day.mp4"
  "/Data/video_data/videos/video_331_rainy_day.mp4"
  "/Data/video_data/videos/video_416_clear_day.mp4"
  "/Data/video_data/videos/video_416_clear_night.mp4"
  "/Data/video_data/videos/video_416_foggy_day.mp4"
  "/Data/video_data/videos/video_416_rainy_day.mp4"
)

# Model configurations and checkpoints

# If creating visualizations of base models
model_configs=(
  "clear_day:masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_clear_day.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth"
  "clear_night:masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/masktrack-rcnn_carla_clear_night_on_clear_day.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth"
  "foggy_day:masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/masktrack-rcnn_carla_foggy_day_on_clear_day.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth"
  "rainy_day:masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/masktrack-rcnn_carla_rainy_day_on_clear_day.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth"
)

# If creating visualizations of domain adaptation models
# model_configs=(
#   "7030foggy:masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30foggy/masktrack-rcnn_carla_70clear_30foggy_on_foggy.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30foggy/epoch_12.pth"
#   "7030rainy:masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30rainy/masktrack-rcnn_carla_70clear_30rainy_on_rainy.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30rainy/epoch_12.pth"
#   "7030night:masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30night/masktrack-rcnn_carla_70clear_30night_on_night.py --checkpoint masktrack-rcnn_models/masktrack-rcnn_carla_70clear_30night/epoch_12.pth"
# )

# Run the commands
for video_file in "${video_files[@]}"; do
  video_name=$(basename "$video_file")
  video_name="${video_name%.*}" # Remove .mp4 extension

  for model_config in "${model_configs[@]}"; do
    model_name=$(echo "$model_config" | cut -d':' -f1)
    config_checkpoint=$(echo "$model_config" | cut -d':' -f2)

    # Modified output filename convention
    output_file="/Data/video_data/visualizations/video_results/DA_with_thresh/${model_name}_on_${video_name}.mp4"

    if ! python demo/mot_demo.py "$video_file" $config_checkpoint --score-thr 0.5 --out "$output_file" 2> error.log; then
      echo "Error processing ${model_name} on ${video_name}:"
      cat error.log
      rm error.log # Clean up the error log
      exit 1 # Exit the script on error
    fi
  done
done

echo "Video processing complete."