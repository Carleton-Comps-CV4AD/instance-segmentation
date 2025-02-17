#!/bin/bash

# Models and their command-line arguments (as strings)
declare -A model_args
model_args=(
  ["tools/test_tracking.py"]="masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_clear_day.py --checkpoint  masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth"
  ["tools/test_tracking.py"]="masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_clear_night.py --checkpoint  masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth"
  ["tools/test_tracking.py"]="masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_foggy_day.py --checkpoint  masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth"
  ["tools/test_tracking.py"]="masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_rainy_day.py --checkpoint  masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth"
)

# Dataroots to use
dataroots=("/Data/video_data/clear_day/" "/Data/video_data/clear_night/" "/Data/video_data/foggy_day/" "/Data/video_data/rainy_day/")

# Loop through each model
for model in "${!model_args[@]}"; do
  # Loop through each dataroot

  for dataroot in "${dataroots[@]}"; do
    echo "Running model: $model on dataset: $dataroot"

    # Modify the dataroot variable in the Python file
    modify_python_variable "$model" "dataset_to_be_tested" "$dataroot"

    # Get the arguments for the current model
    args="${model_args[$model]}"
    model_name=$(echo "$args" | grep -oE 'carla_[^.]*' | cut -d'_' -f2-)



    # Construct the output filename
    output_file="${model_name}_${dataroot}.txt"  

    # Run the Python script with its arguments and redirect output
    python "$model" $args > "$output_file" 2>&1  # Redirect stdout and stderr

    # Check the exit status
    if [[ $? -eq 0 ]]; then
      echo "$model on $dataroot completed successfully. Output saved to $output_file"
    else
      echo "ERROR: $model on $dataroot failed to run. Check $output_file for details."
      exit 1
    fi
  done
done

echo "All model/dataset combinations completed."

modify_python_variable() {
  python_file="$1"
  variable_name="$2"
  new_value="$3"

    # Escape special characters in the new value for sed
    escaped_value=$(printf '%s' "$new_value" | sed -e 's/[]\/$*.^[]/\\&/g')

  if [[ "$new_value" =~ ^[0-9]+$ ]]; then # Integer
      sed -i "s/\($variable_name = \).*/\1$new_value/" "$python_file"
  elif [[ "$new_value" == "True" || "$new_value" == "False" ]]; then # Boolean
       sed -i "s/\($variable_name = \).*/\1$new_value/" "$python_file"
  elif [[ "$new_value" =~ ^[0-9.]+$ ]]; then # Float
      sed -i "s/\($variable_name = \).*/\1$new_value/" "$python_file"
  else # String (add quotes and escaped value)
      sed -i "s/\($variable_name = \).*/\1\"$escaped_value\"/" "$python_file"
  fi
}