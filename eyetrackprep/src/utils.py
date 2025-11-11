import json
import numpy as np
import pandas as pd


RUN2TASK_MAPPING = {
    'retino': {
        'task-wedges': 'run-01',
        'task-rings': 'run-02',
        'task-bar': 'run-03'
    },
    'floc': {
        'task-flocdef': 'run-01',
        'task-flocalt': 'run-02'
    }
}
"""
Assigns BIDS 'run-XX' numbers to runs labeled only by 'task-subtask' for
retino and fLoc tasks, based on their order of administration.
https://github.com/courtois-neuromod/task_stimuli/blob/main/src/sessions/ses-retino.py
https://github.com/courtois-neuromod/task_stimuli/blob/main/src/sessions/ses-floc.py
"""


def get_onset_time(
    log_path: str,
    run_num: str,
    ip_path: str,
    gz_ts: float,
) -> float:
    """."""
    
    onset_time_dict = {}
    TTL_0 = -1
    has_lines = True

    with open(log_path) as f:
        lines = f.readlines()
        if len(lines) == 0:
            has_lines = False
        for line in lines:
            if "fMRI TTL 0" in line:
                TTL_0 = line.split('\t')[0]
            elif "saved wide-format data to /scratch/neuromod/data" in line:
                rnum = line.split('\t')[-1].split('_')[-2]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                rnum = line.split(': ')[-2].split('_')[-1]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.localizers.FLoc'" in line:
                rnum = RUN2TASK_MAPPING['floc'][line.split(': ')[-2]]
                onset_time_dict[rnum] = float(TTL_0)
            elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                rnum = RUN2TASK_MAPPING['retino'][line.split(': ')[-2]]
                onset_time_dict[rnum] = float(TTL_0)

    if has_lines:
        o_time = onset_time_dict[run_num]

        with open(ip_path, 'r') as f:
            iplayer = json.load(f)
        sync_ts = iplayer['start_time_synced_s']
        syst_ts = iplayer['start_time_system_s']

        is_sync_gz = (gz_ts-sync_ts)**2 < (gz_ts-syst_ts)**2
        is_sync_ot = (o_time-sync_ts)**2 < (o_time-syst_ts)**2
        if is_sync_gz != is_sync_ot:
            if is_sync_ot:
                o_time += (syst_ts - sync_ts)
            else:
                o_time += (sync_ts - syst_ts)
    else:
        print('empty log file, onset time estimated from gaze timestamp')
        o_time = gz_ts

    return o_time




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
