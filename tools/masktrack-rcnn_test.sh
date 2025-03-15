#!/bin/bash

# Models and their command-line arguments
# declare -A model_args
# model_args=(
#     ["clear_day"]=("masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_clear_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth")
#     ["clear_night"]=("masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_clear_night.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth")
#     ["foggy_day"]=("masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_foggy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth")
#     ["rainy_day"]=("masktrack-rcnn_models/masktrack-rcnn_carla_day/masktrack-rcnn_carla_rainy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth")
# )

create_args_array() {  # Helper function
    local model_name="$1"
    shift  # Shift off the model name
    local args=("$@") # all remaining arguments
    echo "${args[@]}" #Return the arguments
}

declare -A model_args

# Define the models and their arguments (using the helper function)
model_args["clear_day_on_clear_day"]=$(create_args_array "clear_day_on_clear_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_clear_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth")
model_args["clear_day_on_clear_night"]=$(create_args_array "clear_day_on_clear_night" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_clear_night.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth")
model_args["clear_day_on_foggy_day"]=$(create_args_array "clear_day_on_foggy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_foggy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth")
model_args["clear_day_on_rainy_day"]=$(create_args_array "clear_day_on_rainy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/masktrack-rcnn_carla_clear_day_on_rainy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_day/epoch_12.pth")



model_args["clear_night_on_clear_day"]=$(create_args_array "clear_night_on_clear_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/masktrack-rcnn_carla_clear_night_on_clear_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth")
model_args["clear_night_on_clear_night"]=$(create_args_array "clear_night_on_clear_night" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/masktrack-rcnn_carla_clear_night_on_clear_night.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth")
model_args["clear_night_on_foggy_day"]=$(create_args_array "clear_night_on_foggy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/masktrack-rcnn_carla_clear_night_on_foggy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth")
model_args["clear_night_on_rainy_day"]=$(create_args_array "clear_night_on_rainy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/masktrack-rcnn_carla_clear_night_on_rainy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_clear_night/epoch_12.pth")


model_args["foggy_day_on_clear_day"]=$(create_args_array "foggy_day_on_clear_day" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/masktrack-rcnn_carla_foggy_day_on_clear_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth")
model_args["foggy_day_on_clear_night"]=$(create_args_array "foggy_day_on_clear_night" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/masktrack-rcnn_carla_foggy_day_on_clear_night.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth")
model_args["foggy_day_on_foggy_day"]=$(create_args_array "foggy_day_on_foggy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/masktrack-rcnn_carla_foggy_day_on_foggy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth")
model_args["foggy_day_on_rainy_day"]=$(create_args_array "foggy_day_on_rainy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/masktrack-rcnn_carla_foggy_day_on_rainy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_foggy_day/epoch_12.pth")



model_args["rainy_day_on_clear_day"]=$(create_args_array "rainy_day_on_clear_day" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/masktrack-rcnn_carla_rainy_day_on_clear_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth")
model_args["rainy_day_on_clear_night"]=$(create_args_array "rainy_day_on_clear_night" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/masktrack-rcnn_carla_rainy_day_on_clear_night.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth")
model_args["rainy_day_on_foggy_day"]=$(create_args_array "rainy_day_on_foggy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/masktrack-rcnn_carla_rainy_day_on_foggy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth")
model_args["rainy_day_on_rainy_day"]=$(create_args_array "rainy_day_on_rainy_day" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/masktrack-rcnn_carla_rainy_day_on_rainy_day.py" "--checkpoint" "masktrack-rcnn_models/masktrack-rcnn_carla_rainy_day/epoch_12.pth")


# dataset_to_be_tested='rainy_day'


# Loop through each model
for model_name in "${!model_args[@]}"; do
    echo "Running model: $model_name"  # Simplified message

    args=("tools/test_tracking.py" ${model_args[$model_name]})    # output_dir=$(dirname "${model_name}.txt") # Simplified output filename
    # mkdir -p "$output_dir"
    output_file="${model_name}.txt" # Simplified output filename

    python "${args[@]}" > "./masktrack-rcnn_results/with_classwise/$output_file" 2>&1

    if [[ $? -eq 0 ]]; then
        echo "$model_name completed successfully. Output saved to $output_file" # Simplified message
    else
        echo "ERROR: $model_name failed to run. Check $output_file for details."
        exit 1
    fi
done

echo "All models completed."