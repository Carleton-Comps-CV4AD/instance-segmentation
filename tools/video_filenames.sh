{
  echo "video_files=("
  find /Data/video_data/visualizations/video_results -type f -name "*.mp4" | awk '{print "  \"" $0 "\""}'
  echo ")"
} > video_list.sh