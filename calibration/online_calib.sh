#!/bin/bash

export ID1=037522250789 # change this
export ID0=213522250729 # change this

$HOME/calibration/arpg/releases/bin/vicalib \
 -grid_preset small \
 -frame_skip 4 \
 -num_vicalib_frames 64 \
 -output $HOME/code/hand-object-interaction-data-collection/calibration/$ID0-$ID1.xml \
 -cam convert://realsense2:[id0=$ID0,id1=$ID1,size=1280x720,depth=0]// \
 -nocalibrate_intrinsics \
 -model_files $HOME/code/hand-object-interaction-data-collection/calibration/$ID0.xml,$HOME/code/hand-object-interaction-data-collection/calibration/$ID1.xml