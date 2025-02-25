#!/bin/bash

# Clear Day
ffmpeg -framerate 6 -start_number 7993 -i /Data/video_data/clear_day/val/rgb/video_149/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_149_clear_day.mp4
ffmpeg -framerate 6 -start_number 11935 -i /Data/video_data/clear_day/val/rgb/video_222/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_222_clear_day.mp4
ffmpeg -framerate 6 -start_number 2593 -i /Data/video_data/clear_day/val/rgb/video_49/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_49_clear_day.mp4
ffmpeg -framerate 6 -start_number 4105 -i /Data/video_data/clear_day/val/rgb/video_77/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_77_clear_day.mp4
ffmpeg -framerate 6 -start_number 17821 -i /Data/video_data/clear_day/val/rgb/video_331/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_331_clear_day.mp4
ffmpeg -framerate 6 -start_number 22411 -i /Data/video_data/clear_day/val/rgb/video_416/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_416_clear_day.mp4

# Clear Night
ffmpeg -framerate 6 -start_number 7993 -i /Data/video_data/clear_night/val/rgb/video_149/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_149_clear_night.mp4
ffmpeg -framerate 6 -start_number 11935 -i /Data/video_data/clear_night/val/rgb/video_222/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_222_clear_night.mp4
ffmpeg -framerate 6 -start_number 2593 -i /Data/video_data/clear_night/val/rgb/video_49/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_49_clear_night.mp4
ffmpeg -framerate 6 -start_number 4105 -i /Data/video_data/clear_night/val/rgb/video_77/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_77_clear_night.mp4
ffmpeg -framerate 6 -start_number 17821 -i /Data/video_data/clear_night/val/rgb/video_331/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_331_clear_night.mp4
ffmpeg -framerate 6 -start_number 22411 -i /Data/video_data/clear_night/val/rgb/video_416/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_416_clear_night.mp4

# Foggy Day
ffmpeg -framerate 6 -start_number 7993 -i /Data/video_data/foggy_day/val/rgb/video_149/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_149_foggy_day.mp4
ffmpeg -framerate 6 -start_number 11935 -i /Data/video_data/foggy_day/val/rgb/video_222/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_222_foggy_day.mp4
ffmpeg -framerate 6 -start_number 2593 -i /Data/video_data/foggy_day/val/rgb/video_49/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_49_foggy_day.mp4
ffmpeg -framerate 6 -start_number 4105 -i /Data/video_data/foggy_day/val/rgb/video_77/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_77_foggy_day.mp4
ffmpeg -framerate 6 -start_number 17821 -i /Data/video_data/foggy_day/val/rgb/video_331/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_331_foggy_day.mp4
ffmpeg -framerate 6 -start_number 22411 -i /Data/video_data/foggy_day/val/rgb/video_416/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_416_foggy_day.mp4

# Rainy Day
ffmpeg -framerate 6 -start_number 7993 -i /Data/video_data/rainy_day/val/rgb/video_149/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_149_rainy_day.mp4
ffmpeg -framerate 6 -start_number 11935 -i /Data/video_data/rainy_day/val/rgb/video_222/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_222_rainy_day.mp4
ffmpeg -framerate 6 -start_number 2593 -i /Data/video_data/rainy_day/val/rgb/video_49/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_49_rainy_day.mp4
ffmpeg -framerate 6 -start_number 4105 -i /Data/video_data/rainy_day/val/rgb/video_77/%04d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_77_rainy_day.mp4
ffmpeg -framerate 6 -start_number 17821 -i /Data/video_data/rainy_day/val/rgb/video_331/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_331_rainy_day.mp4
ffmpeg -framerate 6 -start_number 22411 -i /Data/video_data/rainy_day/val/rgb/video_416/%05d.png -c:v mpeg4 -q:v 5 /Data/video_data/videos/video_416_rainy_day.mp4

echo "Video conversion complete."