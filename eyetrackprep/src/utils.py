import os, json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import types
import msgpack
import collections


def log_qc(
    log_message,
    qc_path
) -> None:
    """."""
    print(log_message)    
    with open(qc_path, 'a') as qc_report:
        qc_report.write(f"{log_message}\n")    
        

def get_onset_time(
    log_path: str,
    task_name: str,
    fnum: str,
    ip_path: str,
    gz_ts: float,
    qc_path: str,
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
    qc_path: str
        Path to qc_report_{task}.txt report to log parsing issues

    Returns
    -------
    o_time: float
        Run's onset time on the same clock as the gaze timestamps
    """
    
    onset_time_dict = {}
    TTL_0 = -1
    has_logs = True
    """
    Parse through a session's log file to build a dict of TTL 0 (run trigger) times
    assigned to each run in the log file.

    Note: interrupted sessions have multiple log files, but the file that contains
    a run's logs shares the run's identifying number ('YYYYMMDD-HHMMSS')
    """
    if not Path(log_path).exists():
        log_qc(f"Warning: no .log file found for {fnum}", qc_path)
        has_logs = False
    else:
        with open(log_path) as f:
            lines = f.readlines()
            if len(lines) == 0:
                log_qc(f"Warning: empty .log file for {fnum}", qc_path)
                has_logs = False
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

    if task_name not in onset_time_dict:
        log_qc(f"Run name not found during .log text parsing for {fnum}", qc_path)
        has_logs = False

    if not Path(ip_path).exists():
        log_qc(f"No info.player.json file found for {fnum}", qc_path)
        has_logs = False

    """
    In cases where the log file was parsed successfully, verify if the 
    logged TTL_0 time is on the same clock as the gaze timestamps. 

    If not, convert TTL_0 to the same clock as the gaze
    """
    if has_logs:
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
    
        return o_time
    
    else:
        """
        If the log file cannot be parsed, estimate the scanner onset time 
        from the run's 12th gaze timestamp (estimated from differences between 
        gaze timestamps and logged TTL 0 for 14 CNeuroMod tasks with eye-tracking data)
        """        
        log_qc(f"Warning: failed log parsing, onset time estimated from gaze timestamp for {fnum}", qc_path)
        return gz_ts


def extract_gaze(
    seri_gaze,
    onset_time,
    export_plots,
) -> tuple[list]:
    """
    Convert nested gaze dictionary to two lists of arrays (one for BIDS, one for plotting)
    """
    bids_gaze_list = []
    gaze_2plot_list = []

    for gaze in seri_gaze:
        gaze_timestamp = gaze['timestamp'] - onset_time
        if gaze_timestamp > 0.0:
            gaze_x, gaze_y = gaze['norm_pos']
            gaze_conf = gaze['confidence']
            pupil_data = gaze['base_data'][0]
            pupil_x, pupil_y = pupil_data['norm_pos']
            pupil_diameter = pupil_data['diameter']
            ellipse_data = pupil_data['ellipse']
            ellipse_axe_a, ellipse_axe_b = ellipse_data['axes']
            ellipse_angle = ellipse_data['angle']
            ellipse_center_x, ellipse_center_y = ellipse_data['center']

            bids_gaze_list.append([
                gaze_timestamp, gaze_x, gaze_y, gaze_conf, 
                pupil_x, pupil_y, pupil_diameter, 
                ellipse_axe_a, ellipse_axe_b, ellipse_angle, 
                ellipse_center_x, ellipse_center_y,
            ]) 
            if export_plots:
                gaze_2plot_list.append([
                    gaze_x, gaze_y, gaze_timestamp, gaze_conf,
                ])
                
    return bids_gaze_list, gaze_2plot_list


def get_metadata(
    start_time: float,
) -> dict :
    """."""
    col_names = [
        'timestamp', 'x_coordinate', 'y_coordinate', 'confidence', 
        'pupil_x_coordinate', 'pupil_y_coordinate', 'pupil_diameter',
        'pupil_ellipse_axe_a', 'pupil_ellipse_axe_b', 'pupil_ellipse_angle',
        'pupil_ellipse_center_x', 'pupil_ellipse_center_y'
    ]

    return {
        "StartTime": start_time,
        "Columns": col_names,
        "SamplingFrequency": 250.0,
    }


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



def unpacking_object_hook(obj):
    if isinstance(obj, dict):
        return types.MappingProxyType(obj)
    return obj

def unpacking_ext_hook(code, data):
    if code == MSGPACK_EXT_CODE:
        return msgpack.unpackb(
            data,
            use_list=False,
            object_hook=unpacking_object_hook,
            ext_hook=unpacking_ext_hook,
            strict_map_key=False,
        )
    return msgpack.ExtType(code, data)

def load_pldata_file(directory, topic):
    """
    Deserialize pldata
    """
    msgpack_file = os.path.join(directory, topic + ".pldata")
    is_deserialized = True

    try:
        data = collections.deque()

        with open(msgpack_file, "rb") as stream:
            unpacker = msgpack.Unpacker(
                stream,
                use_list=False,
                strict_map_key=False
            )

            for _, to_unpack in unpacker:
                unpacked = msgpack.unpackb(
                    to_unpack, 
                    use_list=False, 
                    object_hook=unpacking_object_hook, 
                    ext_hook=unpacking_ext_hook, 
                    strict_map_key=False
                )
                data.append(unpacked)

    except FileNotFoundError as err:
        print(f"Couldn't read or unpack: {err}")
        data = []
        is_deserialized = False

    return data, is_deserialized