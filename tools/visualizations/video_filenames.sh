# {
#   echo "video_files=("
#   find /Data/video_data/visualizations/video_results -type f -name "*.mp4" | awk '{print "  \"" $0 "\""}'
#   echo ")"
# } > video_results_list.sh

# {
#   echo "video_files=("
#   find /Data/video_data/visualizations/video_results/with_thresh -type f -name "*.mp4" | awk '{print "  \"" $0 "\""}'
#   echo ")"
# } > video_results_thresh_list.sh



{
  echo "video_files=("
  find /Data/video_data/visualizations/video_results/DA_with_thresh -type f -name "*.mp4" | awk '{print "  \"" $0 "\""}'
  echo ")"
} > video_results_7030_list.sh




# {
#   echo "video_files=("
#   find /Data/video_data/videos -type f -name "*.mp4" | awk '{print "  \"" $0 "\""}'
#   echo ")"
# } > video_list.sh