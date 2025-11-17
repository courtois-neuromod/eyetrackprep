import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd


def get_onset_time(
    log_path: str,
    task_name: str,
    fnum: str,
    ip_path: str,
    gz_ts: float,
) -> float:
    """
    Returns run onset time on the same clock as the gaze timestamps, 
    to reset gaze timestamps to 0 at the start of the run.

    Parameters
    ----------
    log_path : str
        Path to the session's log file that matches a run's identifying number ('YYYYMMDD-HHMMSS')
    task_name : str
        Task tag used to label a distinct run in the raw output (e.g., task-friends-s7e18a, 
        task-thingsmemory_run-1). 
    fnum: str
        Automated output numerical identifier based on date and time of acquisition. At least one per session (more if task interruptions)
    ip_path : str
        Path to the run's info.player.json file
    gz_ts : float
        Timestamp of a run's 10th recorded gaze

    Returns
    -------
    o_time: float
        Run's onset time on the same clock as the gaze timestamps
    """
    
    onset_time_dict = {}
    TTL_0 = -1
    has_lines = True
    """
    Parse through a session's log file to build a dict of TTL 0 (run trigger) times
    assigned to each run in the log file.

    Note: interrupted sessions have multiple log files, but the file that contains
    a run's logs shares the run's identifying number ('YYYYMMDD-HHMMSS')
    """
    if not Path(log_path).exists():
        has_lines = False
    else:
        with open(log_path) as f:
            lines = f.readlines()
            if len(lines) == 0:
                has_lines = False
            for line in lines:
                if "fMRI TTL 0" in line:
                    TTL_0 = line.split('\t')[0]
                #elif "saved wide-format data to /scratch/neuromod/data" in line:
                elif "saved wide-format data to /scratch/neur" in line:  # catches typos at console
                    run_id = line.split('\t')[-1].split(f'{fnum}_')[-1].split('_events.tsv')[0]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.localizers.FLoc'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.mutemusic.Playlist'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.video.SingleVideo'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)
                elif "class 'src.tasks.narratives" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = float(TTL_0)

    """
    In cases where the log file is not empty, verify if the 
    logged TTL_0 time is on the same clock as the gaze timestamps. 

    If not, convert TTL_0 to the same clock as the gaze
    """
    if has_lines and Path(ip_path).exists():
        # get run's logged TTL 0 time
        o_time = onset_time_dict[task_name]

        # for each run, the info.player.json file contains the same
        # start time logged on two different clocks: synced and system  
        with open(ip_path, 'r') as f:
            iplayer = json.load(f)
        sync_ts = iplayer['start_time_synced_s']
        syst_ts = iplayer['start_time_system_s']

        # check if gaze timestamp is on synced time
        is_sync_gz = (gz_ts-sync_ts)**2 < (gz_ts-syst_ts)**2
        # check if TTL 0 is on synced time
        is_sync_ot = (o_time-sync_ts)**2 < (o_time-syst_ts)**2
        # If gaze and TTL 0 are on different clocks, 
        # convert TTL 0 to the same clock as the gaze timestamps
        if is_sync_gz != is_sync_ot:
            if is_sync_ot:
                # convert TTL 0 from synced to system clock
                o_time += (syst_ts - sync_ts)
            else:
                # convert TTL 0 from system to synced clock                
                o_time += (sync_ts - syst_ts)
    
    else:
        """
        If the log file is empty, estimate the trigger time based on the run's
        10th gaze timestamp (estimated based on observed lags between eye-tracker
        and task logs)
        """        
        print('empty log file, onset time estimated from gaze timestamp')
        o_time = gz_ts

    return o_time


def parse_task_name(
    pupfile_path,
    task_root,
) -> tuple[str]:
    """."""
    pupil_str, task_str = pupfile_path.split('/')[-3:-1]
    sub, ses, fnum = pupil_str.split('.')[0].split('_')

    if task_root == "friends":
        task_type = f'task-{task_str.split("-")[-1]}'
    elif task_root == "ood":
        task_type = task_str
    elif task_root == "narratives":
        task_type = task_str.split("_run-")[0].replace("_", "")

    return sub, ses, fnum, task_type


def create_event_path(row, file_path, log=False):
    '''
    for each run, create path to events.tsv or PsychoPy log file
    '''
    s = row['subject']
    ses = row['session']
    if log:
        return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}.log'
    else:
        if row['task'] in ['task-bar', 'task-rings', 'task-wedges', 'task-flocdef', 'task-flocalt']:
            return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}_{row["task"]}_events.tsv'
        else:
            return f'{file_path}/{s}/{ses}/{s}_{ses}_{row["file_number"]}_{row["task"]}_{row["run"]}_events.tsv'
