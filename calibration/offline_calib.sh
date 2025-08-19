#!/bin/bash

CALIB_BASE=${CALIB_BASE:-"$HOME/calibration"}  # Use CALIB_BASE env var or fallback to $HOME/calibration
CALIB_FILES=${CALIB_FILES:-"./calibration"}

export ID0=037522250789 # change this
export ID1=213622251272 # change this

$CALIB_BASE/arpg/releases/bin/vicalib \
 -grid_preset small \
 -frame_skip 1 \
 -num_vicalib_frames 64\
 -output $CALIB_FILES/$ID0-$ID1.xml \
 -cam file:[loop=0,startframe=0]//$CALIB_FILES/calib_images/[cam0,cam1]*.png \
 -nocalibrate_intrinsics \
 -model_files $CALIB_FILES/$ID0.xml,$CALIB_FILES/$ID1.xml

