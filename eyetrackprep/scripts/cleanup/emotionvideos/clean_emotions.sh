#!/bin/bash

BIDS_DIR="/data/neuromod/projects/eyetracking_bids/bids_repos/emotion-videos"
DERIV_DIR="/data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos.eyetrackprep"
FILE_PATH="${DERIV_DIR}/code/eyetrackprep/eyetrackprep/scripts/cleanup_scripts/emotionsvideos_relabelruns.tsv"

python clean_emotionvideos.py "${BIDS_DIR}" "${DERIV_DIR}" "${FILE_PATH}" 