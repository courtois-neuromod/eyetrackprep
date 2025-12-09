import os, json
import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import types
import msgpack
import collections


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


def log_qc(
    log_message,
    qc_path
) -> None:
    """."""
    print(log_message)    
    with open(qc_path, 'a') as qc_report:
        qc_report.write(f"{log_message}\n")    


def init_log(
    log_dir: str,
    task_root: str,
    job_tag: str,
) -> None: 
    """."""
    log_path = f"{log_dir}/code/QC_gaze/{job_tag}_report_{task_root}.txt"
    if Path(log_path).exists():
        log_qc(f"\n---------------------------\n{datetime.datetime.now()}\n", log_path)
    else:
        Path(f"{log_dir}/code/QC_gaze").mkdir(parents=True, exist_ok=True)
        Path(log_path).touch()


def init_logs(
    task_root: str,
    correct_drift: bool,
    export_plots: bool,
    out_dir: str,
    deriv_dir: Union[str, None],
) -> None: 
    """."""
    init_log(out_dir, task_root, 'qc')
    if correct_drift:
        init_log(deriv_dir, task_root, 'qc')
        if export_plots:
            init_log(deriv_dir, task_root, 'plot')
    elif export_plots:
        init_log(out_dir, task_root, 'plot')
        

def get_onset_time(
    log_path: str,
    task_name: str,
    fnum: str,
    ip_path: str,
    gz0_ts: float,
    gzn_ts: float,
    qc_path: str,
) -> tuple[float, float]:
    """
    Returns run onset and offset time on the same clock as the gaze timestamps, 
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
    gz0_ts : float
        Timestamp of a run's 12th recorded gaze
    gzn_ts : float
        Timestamp of a run's last recorded gaze
    qc_path: str
        Path to qc_report_{task}.txt report to log parsing issues

    Returns
    -------
    (on_time, off_time) : tuple[float, float]
        Run's onset and offset time on the same clock as the gaze timestamps
    """
    
    onset_time_dict = {}
    TTL_0 = -1
    stop_et  = -1
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
                elif "stopping eyetracking recording" in line:
                    stop_et = line.split('\t')[0]
                #elif "saved wide-format data to /scratch/neuromod/data" in line:
                elif "saved wide-format data to /scratch/neur" in line:  # catches typos at console
                    run_id = line.split('\t')[-1].split(f'{fnum}_')[-1].split('_events.tsv')[0]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.localizers.FLoc'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.mutemusic.Playlist'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.video.SingleVideo'" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))
                elif "class 'src.tasks.narratives" in line:
                    run_id = line.split(': ')[-2]
                    onset_time_dict[run_id] = (float(TTL_0), float(stop_et))

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
        on_time, off_time = onset_time_dict[task_name]

        # for each run, the info.player.json file contains the same
        # start time logged on two different clocks: synced and system  
        with open(ip_path, 'r') as f:
            iplayer = json.load(f)
        sync_ts = iplayer['start_time_synced_s']
        syst_ts = iplayer['start_time_system_s']

        # check if gaze timestamp is on synced time
        is_sync_gz = (gz0_ts-sync_ts)**2 < (gz0_ts-syst_ts)**2
        # check if TTL 0 is on synced time
        is_sync_ot = (on_time-sync_ts)**2 < (on_time-syst_ts)**2
        # If gaze and TTL 0 are on different clocks, 
        # convert TTL 0 to the same clock as the gaze timestamps
        if is_sync_gz != is_sync_ot:
            if is_sync_ot:
                # convert TTL 0 from synced to system clock
                on_time += (syst_ts - sync_ts)
                off_time += (syst_ts - sync_ts)
            else:
                # convert TTL 0 from system to synced clock                
                on_time += (sync_ts - syst_ts)
                off_time += (sync_ts - syst_ts)
    
        return on_time, off_time
    
    else:
        """
        If the log file cannot be parsed, estimate the scanner onset time 
        from the run's 12th gaze timestamp (estimated from differences between 
        gaze timestamps and logged TTL 0 for 14 CNeuroMod tasks with eye-tracking data)
        """        
        log_qc(f"Warning: failed log parsing, run onset and offset time estimated from gaze timestamps for {fnum}", qc_path)
        return gz0_ts, gzn_ts


def extract_gaze(
    seri_gaze,
    onset_time,
) -> list:
    """
    Converts nested gaze dictionary to two lists of arrays (one for BIDS, one for plotting)
    """
    bids_gaze_list = []

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
                
    return bids_gaze_list


def detect_freezes(
    gaze: np.array,
    out_file: str,
    run_duration: float,
) -> None:
    """
    Identifies temporal gaps in eye-tracking data due to camera freezes, 
    and exports them as physioevents.tsv file.

    Detects instances of inter-sample interval > 0.005s (indicating a dropped frame 
    or camera freeze, based on a 250Hz pupil sampling rate). 

    Args:
        gaze (np.array): A 2D array of gaze data with timestamps in col 0 (in s). 
        out_file (str): The output file name for a particular run.
        run_duration (float): The estimated run duration (in s).

    If camera freezes are detected, two files are exported:
    - `physioevents.tsv.gz`: A compressed tab-separated file with 'onset' and 
        'duration' for the identified camera freezes.
    - `events.json`: A metadata file describing the columns in the TSV.
    """
    ts_arr = np.stack((
            gaze[:-1, 0], gaze[1:, 0] - gaze[:-1, 0],
        ), axis=1,
    )
    ts_arr = ts_arr[ts_arr[:, 1] > 0.005]

    if gaze[0, 0] > 0.04:  # ~ 10 skipped frames
        ts_arr = np.concat([
            np.array([[0.0, gaze[0, 0]]]),
            ts_arr,
        ], axis=0)
        
    if run_duration - gaze[-1, 0] > 0.04:  # ~ 10 skipped frames
        ts_arr = np.concat([
            ts_arr,
            np.array([[gaze[-1, 0], run_duration-gaze[-1, 0]]]),
        ], axis=0)
        
    if ts_arr.shape[0] > 0:
        pd.DataFrame(ts_arr).to_csv(
            f'{out_file}events.tsv.gz', sep='\t', header=False, index=False, compression='gzip',
        )
        with open(f'{out_file}events.json', 'w') as metadata_file:
            json.dump({
                    "Columns": ['onset', 'duration'],
                    "Description": "Eye-tracking camera freezes.",
                    "OnsetSource": "timestamp",
                }, metadata_file, indent=4,
            )


def get_metadata(
    start_time: float,
    col_names: list[str],
) -> dict :
    """."""
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


def get_event_path(
    pupil_path: str,
) -> str:
    '''
    Parses gaze.pldata parent directory's path, 
    returns path to corresponding run's events.tsv file.

    Works for tasks with events.tsv files: 
        emotionsvideos, floc, friends_fix, langloc, mario, 
        mariostars, mario3, movie10_fix, multfs, mutemusic, 
        narratives (recency sub-task only), retino, things, 
        and triplets
    '''
    parent_path = str(Path(pupil_path).parents[2])
    pupil_str, task_str = pupil_path.split('/')[-3:-1]
    
    return f"{parent_path}/{pupil_str.split('.')[0]}_{task_str}_events.tsv"


def get_conf_thresh(
    task_root: str,
    ev_path: str,
) -> float:
    '''
    Return pupil detection confidence threshold for a given run or subject, 
    per task.
    '''
    with open(f'./config/{task_root}.json', 'r') as conf_file:
        conf_dict = json.load(conf_file)

    sub, ses, fnum = ev_path.split('_')[:3]
    run = ev_path.split('task-')[-1].replace('_events.tsv', '')

    return conf_dict.get(sub, {}).get(ses, {}).get(fnum, {}).get(run, conf_dict[sub]['default'])
