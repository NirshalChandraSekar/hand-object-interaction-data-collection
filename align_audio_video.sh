ffmpeg -i output.mp4 -i dataset/task1/audio/1755276910.3663642.wav -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 synced_output.mp4
