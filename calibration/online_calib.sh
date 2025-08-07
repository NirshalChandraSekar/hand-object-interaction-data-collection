#!/bin/bash


CALIB_BASE=${CALIB_BASE:-"$HOME/calibration"}  # Use CALIB_BASE env var or fallback to $HOME/calibration
CALIB_FILES=${CALIB_FILES:-"./calibration"}

export ID0=213522250729 # change this
export ID1=037522250789 # change this

$CALIB_BASE/arpg/releases/bin/vicalib \
 -grid_preset small \
 -frame_skip 4 \
 -num_vicalib_frames 64 \
 -output $CALIB_BASE/$ID0-$ID1.xml \
 -cam convert://realsense2:[id0=$ID0,id1=$ID1,size=1280x720,depth=0]// \
 -nocalibrate_intrinsics \
 -model_files $CALIB_FILES/$ID0.xml,$CALIB_FILES/$ID1.xml



