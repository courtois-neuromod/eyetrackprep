import os, glob, sys, json, re
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from bids import BIDSLayout

from src.utils import (
    parse_task_name, 
    load_pldata_file, 
    log_qc,
    get_device_name,
)


BIDS_COL_NAMES = [
    'timestamp', 'x_coordinate', 'y_coordinate', 'confidence', 
    'pupil_x_coordinate', 'pupil_y_coordinate', 'pupil_diameter',
    'pupil_ellipse_axe_a', 'pupil_ellipse_axe_b', 'pupil_ellipse_angle',
    'pupil_ellipse_center_x', 'pupil_ellipse_center_y'
]


def sort_tasks(
    pup_dir: str, 
    task_root: str,
    fnum: str,
) -> tuple[list[str], bool]:
    """."""
    # parse log file corresponding to fnum.pupil output dir to get run temporal order
    try:
        task_list = []
        with open(pup_dir.replace(".pupil", ".log")) as f:
            lines = f.readlines()
            for line in lines:
                if "saved wide-format data to /scratch/neur" in line:
                    task_list.append(
                        line.split('\t')[-1].split(f'{fnum}_')[-1].split('_events.tsv')[0]
                    )
                elif "class 'src.tasks.localizers.FLoc'" in line:
                    task_list.append(line.split(': ')[-2])
                elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                    task_list.append(line.split(': ')[-2])
                elif "class 'src.tasks.mutemusic.Playlist'" in line:
                    task_list.append(line.split(': ')[-2])
                elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                    task_list.append(line.split(': ')[-2])
                elif "class 'src.tasks.video.SingleVideo'" in line:
                    task_list.append(line.split(': ')[-2])
                elif "class 'src.tasks.narratives" in line:
                    task_name = line.split(': ')[-2]
                    if task_name != task_list[-1]:
                        task_list.append(task_name)
    except:
        task_list = []
        
    if len(task_list) > 0:
        return [f"{pup_dir}/{t}" for t in task_list], True 
    elif task_root in ['emotionsvideos', 'friends', 'friends_fix', 'mario', 'mario3', 'mariostars', 'things', 'triplets']:
        return sorted(glob.glob(f"{pup_dir}/*task-*")), True
    elif task_root == 'mutemusic':
        return [re.sub(r'(run-)(\d+)', lambda m: f"{m.group(1)}{int(m.group(2))}", tsk_pad) for tsk_pad in sorted(
            [re.sub(r'(run-)(\d+)', lambda m: f"{m.group(1)}{m.group(2).zfill(3)}", tsk) for tsk in glob.glob(
                f"{pup_dir}/*task-*")])], True
    else:
        return glob.glob(f"{pup_dir}/*task-*"), False


def build_calibration_dict(
    task_root: str,
    in_path: str,
) -> dict:
    """
    Returns a nested dictionary of eye-tracking calibration coordinates, per run
    """
    calib_dict = {}
    for pup_dir in sorted(glob.glob(f"{in_path}/sub-*/ses*/*.pupil")):
        sub, ses, fnum = os.path.basename(pup_dir).split("_")[:3]
        fnum = fnum.split(".")[0]
        if sub not in calib_dict:
            calib_dict[sub] = {}
        if ses not in calib_dict[sub]:
            calib_dict[sub][ses] = {}
        if fnum not in calib_dict[sub][ses]:
            calib_dict[sub][ses][fnum] = {}
            if task_root == 'mario':
                cal_list = [re.sub(r'(calibration-)(\d+)', lambda m: f"{m.group(1)}{int(m.group(2))}", cal_pad) for cal_pad in sorted(
                    [re.sub(r'(calibration-)(\d+)', lambda m: f"{m.group(1)}{m.group(2).zfill(3)}", cal) for cal in glob.glob(
                        f"{in_path}/{sub}/{ses}/*{fnum}*alibration*events*.tsv")]
                )]
            else:
                cal_list = sorted(glob.glob(f"{in_path}/{sub}/{ses}/*{fnum}*alibration*events*.tsv"))
            task_list, task_is_sorted = sort_tasks(pup_dir, task_root, fnum)
            for i, task in enumerate(task_list):
                if Path(f"{task}/000/pupil.pldata").exists():
                    tname = os.path.basename(task)
                    try:
                        if task_is_sorted:
                            cal_df = pd.read_csv(cal_list[i], sep="\t")
                            calib_dict[sub][ses][fnum][tname] = {
                                'hv': f"HV{str(cal_df.shape[0])}",
                                'coord': [[int(cal_df.iloc[j, 0] + 1280/2), int(cal_df.iloc[j, 1] + 1024/2)] for j in range(cal_df.shape[0])],
                            }
                        else:
                            cal_df = pd.read_csv(cal_list[0], sep="\t")
                            calib_dict[sub][ses][fnum][tname] = {
                                'hv': f"HV{str(cal_df.shape[0])}",
                                'coord': [],
                            }
                            print(f"cannot sort calibration coordinates for {sub}, {ses}, {fnum} {tname}")
                    except:
                        print(f"no calibration coordinates found for {sub}, {ses}, {fnum} {tname}")
    
    return calib_dict


def parse_ev_dset(
    df_files: pd.DataFrame,
    task_root: str,
    sub_list: list,
    ses_list: list,
) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Processes a BIDS-like raw dataset structure to compile 
    a DataFrame overview of available eye-tracking files.

    This function relies on parsing run-wise events.tsv files, 
    is incompatible with tasks without logged "events" 
    (e.g., tasks like friends, movie10, ood and narratives).

    Parameters
    ----------
    df_files : pandas.DataFrame
        The cumulative DataFrame to append run data to. Must have columns = [
        'subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 
        'has_gaze', 'has_eyemovie', 'has_log', 'empty_log'].
    task_root : str
        The root task name (e.g., 'mario', 'retino', 'floc'). 
    ses_list : list of str
        A list of paths to fMRI session directories (e.g., 'path/to/sub-XX/ses-YY').

    Returns
    -------
    tuple
        - df_files : pandas.DataFrame
            The updated DataFrame containing eyetracking file availability 
            info for each run.
        - pupil_file_paths : list of tuple
            A list of tuples: (pupil_data_directory_path, metadata_tuple)
            for all runs with a 'pupil.pldata' file.
    """
    has_run_num = False if task_root in ['retino', 'floc', 'langloc', 'friends_fix', 'movie10fix'] else True

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]

        events_list = sorted(glob.glob(f'{ses_path}/*task*events.tsv'))
        for event in events_list:
            ev_file = os.path.basename(event)

            try:
                if has_run_num:
                    sub, ses, fnum, task_type, run_num, appendix = ev_file.split('_')
                else:
                    sub, ses, fnum, task_type = ev_file.split('_')[:4]
                    if task_type == 'task-alice':
                        task_type = f'{task_type}_{ev_file.split("_")[4]}'
                    run_num = None 

                if sub in sub_list:
                    assert sub == sub_num
                    assert ses_num == ses

                    log_list = glob.glob(
                        f'{ses_path}/{sub_num}_{ses_num}_{fnum}.log'
                    )
                    has_log = len(log_list) == 1
                    if has_log:
                        with open(log_list[0]) as f:
                            lines = f.readlines()
                            empty_log = len(lines) == 0
                    else:
                        empty_log = True

                    if has_run_num:
                        pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}_{run_num}/000'
                    else:
                        pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}/000'

                    list_pupil = glob.glob(f'{pupil_path}/pupil.pldata')
                    has_pupil = len(list_pupil) == 1
                    if has_pupil:
                        pupil_file_paths.append(
                            (
                                os.path.dirname(list_pupil[0]),
                                (sub, ses, run_num, task_type, fnum),
                            )
                        )

                    has_eyemv = len(glob.glob(f'{pupil_path}/eye0.mp4')) == 1
                    has_gaze = len(glob.glob(f'{pupil_path}/gaze.pldata')) == 1

                    run_data = [
                        sub_num, ses_num, run_num, task_type, fnum,
                        has_pupil, has_gaze, has_eyemv, has_log, empty_log,
                    ]
                    df_files = pd.concat(
                        [
                            df_files,
                            pd.DataFrame(
                                np.array(run_data).reshape(1, -1),
                                columns=df_files.columns,
                            ),
                        ],
                        ignore_index=True,
                    )

            except:
                print(f'[parse_ev_dset] cannot process {ev_file}')

    return df_files, pupil_file_paths


def parse_noev_dset(
    df_files: pd.DataFrame,
    task_root: str,
    sub_list: list,
    ses_list: list,
) -> tuple[pd.DataFrame, list[tuple]]:
    """
    Processes a BIDS-like raw dataset structure to compile 
    a DataFrame overview of available eye-tracking files.

    This task does not rely on events.tsv files, it parses 
    run details directly from the pupil data directory path.
    Compatible with tasks without logged events (e.g., friends, 
    ood, movie10, narratives).

    Parameters
    ----------
    df_files : pandas.DataFrame
        The cumulative DataFrame to append run data to. Must have columns = [
        'subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 
        'has_gaze', 'has_eyemovie', 'has_log', 'empty_log'].
    task_root : str
        The root task name (e.g., 'friends', 'ood', 'narratives'). 
    ses_list : list of str
        A list of paths to session directories (e.g., 'path/to/sub-XX/ses-YY').

    Returns
    -------
    tuple
        - df_files : pandas.DataFrame
            The updated DataFrame containing eyetracking file availability 
            info for each run.
        - pupil_file_paths : list of tuple
            A list of tuples: (pupil_data_directory_path, metadata_tuple)
            for all runs with a 'pupil.pldata' file.
    """

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]

        epfile_list = sorted(glob.glob(
                f'{ses_path}/sub-*.pupil/task-*/000'
            )
        )

        for epfile in epfile_list:
            sub, ses, fnum, task_type = parse_task_name(
                epfile, task_root,
            )
            run_num = None

            if sub not in sub_list:
                print(f"Subject {sub} data not in subject list")
            if sub != sub_num:
                print(f"Subject {sub} data under {sub_num} directory")
            if ses_num != ses:
                print(f"{ses} data under {ses_num} directory")

            log_list = glob.glob(
                f'{ses_path}/{sub_num}_{ses_num}_{fnum}.log'
            )
            has_log = len(log_list) == 1
            if has_log:
                with open(log_list[0]) as f:
                    lines = f.readlines()
                    empty_log = len(lines) == 0
            else:
                empty_log = True

            list_pupil = glob.glob(f'{epfile}/pupil.pldata')
            has_pupil = len(list_pupil) == 1
            if has_pupil:
                pupil_file_paths.append(
                    (os.path.dirname(
                        list_pupil[0]),
                        (sub, ses, run_num, task_type, fnum))
                )

            has_eyemv = len(glob.glob(f'{epfile}/eye0.mp4')) == 1
            has_gaze = len(glob.glob(f'{epfile}/gaze.pldata')) == 1

            run_data = [
                sub_num, ses_num, run_num, task_type, fnum,
                has_pupil, has_gaze, has_eyemv, has_log, empty_log,
            ]
            df_files = pd.concat(
                [
                    df_files,
                    pd.DataFrame(
                        np.array(run_data).reshape(1, -1),
                        columns=df_files.columns,
                    )
                ],
                ignore_index=True,
            )

    return df_files, pupil_file_paths


def compile_rawfile_list(
    in_path: str,
    out_path: BIDSLayout,
) -> tuple[tuple[pd.DataFrame, list[tuple]], str, dict]:
    """
    Compiles an overview of all available eye-tracking files
    for a given CNeuroMod dataset and exports it as a .tsv file 
    to support quality control (QC).

    The availability of key output files (from Pupil: pupil.pldata, 
    gaze.pldata, eye0.mp4; from PsychoPy: log file) is checked 
    for every run in the dset.

    Parameters
    ----------
    in_path : str
        The root directory of the raw dataset 
        (e.g., '/unf/eyetracker/neuromod/retino/sourcedata').
    out_path : str
        The output directory where the .tsv file listing available data will be saved.

    Returns
    -------
    tuple
        - df_files : pandas.DataFrame
            A DataFrame containing file availability info for all processed runs.
        - pupil_file_paths : list of tuple
            A list of tuples: (pupil_data_directory_path, metadata_tuple)
            for all runs that contain a 'pupil.pldata' file.
    task_root: str
        dset task name
    calib_coordinates: dict
        dictionary with calibration marker configuration (hv9 or hv10), per run, 
        and list of  fixation marker coordiates in their temporal order of appearance
    """
    # Find all the fMRI session directories with numeric session numbers for all subjects
    # e.g., (on elm, in_path = '/unf/eyetracker/neuromod/triplets/sourcedata')
    sub_list = [f'sub-{sub_id}' for sub_id in out_path.get_subjects()]
    
    ses_list = [
        x for x in sorted(glob.glob(
            f'{in_path}/sub-*/ses-*')
        ) if x.split('-')[-1].isnumeric()
    ]

    col_names = [
        'subject', 'session', 'run', 'task', 'file_number', 
        'has_pupil', 'has_gaze', 'has_eyemovie',
        'has_log', 'empty_log', 
    ]
    df_files = pd.DataFrame(columns=col_names)

    task_root = in_path.split('/')[-2]

    # Parse dataset files (using events.tsv files if available) 
    if task_root in ['friends', 'movie_10', 'ood', 'narratives']:
        df_files, pupil_file_paths = parse_noev_dset(
            df_files, task_root, sub_list, ses_list
        )
    else:
        df_files, pupil_file_paths = parse_ev_dset(
            df_files, task_root, sub_list, ses_list
        )

    # Extract calibration coordinates 
    calib_coordinates = build_calibration_dict(task_root, in_path)

    # Export file list spreadsheet to support QCing
    Path(f"{out_path.root}/code/QC_gaze").mkdir(parents=True, exist_ok=True)

    df_files.to_csv(
        f"{out_path.root}/code/QC_gaze/{task_root}_QCflist.tsv",
        sep='\t', header=True, index=False,
    )

    return pupil_file_paths, task_root, calib_coordinates


def get_onset_time(
    log_path: str,
    task_name: str,
    fnum: str,
    ip_path: str,
    gz0_ts: float,
    gzn_ts: float,
    qc_path: str,
) -> tuple[float, float, str]:
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
    (on_time, off_time, rsv) : tuple[float, float, str]
        Run's onset and offset time on the same clock as the gaze timestamps, 
        and Pupil Core software version 
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
        rsv = f"Pupil Capture version {iplayer['recording_software_version']}"

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
    
        return on_time, off_time, rsv
    
    else:
        """
        If the log file cannot be parsed, estimate the scanner onset time 
        from the run's 12th gaze timestamp (estimated from differences between 
        gaze timestamps and logged TTL 0 for 14 CNeuroMod tasks with eye-tracking data)
        """        
        log_qc(f"Warning: failed log parsing, run onset and offset time estimated from gaze timestamps for {fnum}", qc_path)
        return gz0_ts, gzn_ts, "n/a"


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
) -> float:
    """
    Identifies temporal gaps in eye-tracking data due to camera freezes, 
    and exports them as physioevents.tsv file.

    Detects instances of inter-sample interval > 0.25s (indicating a  
    camera freeze, based on a 250Hz pupil sampling rate). 

    Args:
        gaze (np.array): A 2D array of gaze data with timestamps in col 0 (in s). 
        out_file (str): The output file name for a particular run.
        run_duration (float): The estimated run duration (in s).

    Returns:
        int: number of camera freezes

    If camera freezes are detected, two files are exported:
    - `physioevents.tsv.gz`: A compressed tab-separated file with 'onset' and 
        'duration' for the identified camera freezes.
    - `events.json`: A metadata file describing the columns in the TSV.
    """
    ts_arr = np.stack((
            gaze[:-1, 0], gaze[1:, 0] - gaze[:-1, 0],
        ), axis=1,
    )
    #ts_arr = ts_arr[ts_arr[:, 1] > 0.005]  # too sensitive: detects any skipped frame...
    ts_arr = ts_arr[ts_arr[:, 1] > 0.25]  # > 0.5s freezes only

    if gaze[0, 0] > 1.0:  
        """
        1s buffer between the first captured gaze and the logged
        run onset time (TTL 0), to account for potential delays
        """
        ts_arr = np.concat([
            np.array([[0.0, gaze[0, 0]]]),
            ts_arr,
        ], axis=0)
        
    if run_duration - gaze[-1, 0] > 2.0:  
        """
        2s buffer between the last captured gaze and the logged
        eyetracker offset time to account for potential logging delay
        """
        ts_arr = np.concat([
            ts_arr,
            np.array([[gaze[-1, 0], run_duration-gaze[-1, 0]]]),
        ], axis=0)
        
    if ts_arr.shape[0] > 0:
        pd.DataFrame(ts_arr).to_csv(
            f'{out_file}events.tsv.gz', sep='\t', header=False, index=False, compression='gzip',
        )

    return ts_arr.shape[0]


def format_dset_metadata(
    out_dir: str,
)-> None:
    """."""
    dset_name = os.path.basename(out_dir)
    with open(f'{out_dir}/task-{dset_name}_recording-eye0_physioevents.json', 'w') as metadata_file:
        json.dump({
                "Columns": ['onset', 'duration'],
                "Description": "Eye-tracking camera freezes.",
                "OnsetSource": "timestamp",
                "onset": {
                    "Description": "Onset of the camera freeze event.",
                    "Units": "seconds",
                },
                "duration": {
                    "Description": "Duration of the camera freeze event.",
                    "Units": "seconds",
                }
            }, metadata_file, indent=4,
        )

    if Path(f'{out_dir}/task-{dset_name}_events.json').exists():
        with open(f'{out_dir}/task-{dset_name}_events.json', 'r') as metadata_file:
            events_data = json.load(metadata_file)
    else:
        events_data = {}

    events_data["StimulusPresentation"] = {
        "ScreenDistance": 1.8,
        "ScreenDistanceUnits": "meters",
        "ScreenSize": [0.55, 0.44],
        "ScreenSizeUnits": "meters",
        "ScreenOrigin": ["bottom", "left"],
        "ScreenResolution": [1280, 1024],
        "ScreenResolutionUnits": pixels,
    }
    with open(f'{out_dir}/task-{dset_name}_events.json', 'w') as metadata_file:
        json.dump(events_data, metadata_file, indent=4)


def format_runwise_metadata(
    start_time: float,
    duration: float,
    col_names: list[str],
    freeze_count: int,
    pupil_version: str,
    device_name: str,
    calib_vals: dict,
) -> dict :
    """."""
    return {
        "DeviceSerialNumber": device_name,
        "Columns": col_names,
        "Manufacturer": "MRC",
        "ManufacturersModelName": "MRC-HighSpeed",
        "PhysioType": "eyetrack",
        "RecordedEye": "right",
        "SamplingFrequency": 250.0,
        "SampleCoordinateSystem": "gaze-on-screen",
        "SoftwareVersion": pupil_version,
        "PupilFitMethod": "ellipse",
        "CalibrationType": calib_vals['hv'],
        "CalibrationPosition": calib_vals['coord'],
        "CalibrationUnit": "pixel",
        "EyeCameraSettings": {
            "exposure_time": 4000,
            "global_gain": 1,
        },
        "EyeTrackerDistance": 0.1,
        "EyeTrackingMethod": "pupil-labs/pupil-detectors:2d",
        "StartTime": start_time,
        "Duration": duration,
        "CameraFreezeCount": freeze_count,
        "timestamp": {
            "Description": "A continuously increasing identifier of the sampling time registered by the device",
            "Units": "seconds",
            "Origin": "run trigger onset",
        },
        "x_coordinate": {
            "LongName": "Gaze position (x)",
            "Description": "Gaze position x-coordinate of the recorded eye, normalized as a proportion of the screen width. Bound = [0, 1], where 0 = left.",
            "Units": "arbitrary",
        },
        "y_coordinate": {
            "LongName": "Gaze position (y)",
            "Description": "Gaze position y-coordinate of the recorded eye, normalized as a proportion of the screen height. Bound = [0, 1], where 0 = bottom.",
            "Units": "arbitrary",
        },
        "confidence": {
            "Description": "Quality assessment of the pupil detection ranging from 0 to 1. A value of 0 indicates that the pupil could not be detected, whereas a value of 1 indicates a very high pupil detection certainty.",
            "Units": "ratio",
        },
        "pupil_x_coordinate": {
            "LongName": "Pupil position (x)",
            "Description": "Pupil position x-coordinate normalized as a proportion of the width of the eye video frame. A value of 0 indicates the left of the frame.",
            "Units": "pixel",
        },
        "pupil_y_coordinate": {
            "LongName": "Pupil position (y)",
            "Description": "Pupil position y-coordinate normalized as a proportion of the height of the eye video frame. Values are ranging from 0 to 1. A value of 0 indicates the bottom of the frame.",
            "Units": "pixel",
        },
        "pupil_diameter": {
            "Description": "Diameter of the pupil as observed in the eye image frame (not corrected for perspective).",
            "Units": "pixel", 
        },
        "pupil_ellipse_axe_a": {
            "Description": "Length of the first axis of the 2D fitted ellipse used to model of the pupil.",
            "Units": "pixel",
        },
        "pupil_ellipse_axe_b": {
            "Description": "Length of the second axis of the 2D fitted ellipse used to model of the pupil.",
            "Units": "pixel",
        },
        "pupil_ellipse_angle": {
            "Description": "Orientation of the 2D fitted ellipse used to model the pupil.",
            "Units": "degrees",
        },
        "pupil_ellipse_center_x": {
            "Description": "x-coordinate of the center of the 2D fitted ellipse used to model the pupil.",
            "Units": "pixel",
        },
        "pupil_ellipse_center_y": {
            "Description": "y-coordinate of the center of the 2D fitted ellipse used to model the pupil",
            "Units": "pixel",
        }
    }


def get_calib_vals(
    p_path: str, 
    calib_dict: dict,
) -> dict:
    """."""
    sub, ses, pup_dir, task = p_path.split("/")[-5:-1]
    fnum = pup_dir.split("_")[2].split(".")[0]

    run_dict = calib_dict.get(sub, {}).get(ses, {}).get(fnum, {}).get(task, None)

    if run_dict is None or run_dict['hv'] not in ['HV9', 'HV10']:
        return {'hv': 'N/A', 'coord': []}
    else:
        return run_dict


def export_bids(    
    pupil_path: tuple,
    in_path: str,
    calib_coordinates: dict,
    out_path: str,
) -> np.array:
    '''
    Function extracts a single run's gaze and pupil metrics from .pldata (Pupil's) format,
    and exports them to BIDS (BEP20, .tsv.gz). 
    
    Returns bids-compatible gaze & pupil data, and timestamped gaze position with detection 
    confidence scores to plot (for manual QCing)

    Parameters
    ----------
    pupil_path : str
        Path to the directory that contains one functional runs' eye-tracking data files
    in_path : str
        The root directory of the dataset (e.g., '.../neuromod/retino/sourcedata').
    calib_coordinates : dict
        Nested dictionary of marker coordinates for gaze calibration, per run 
    out_path : str
        The output directory where the BIDS & QC files will be saved.

    Returns
    -------
    Numpy array : the run's gaze and pupil data exported to bids, to support drift correction and 
        produce plots to support QCing
    '''
    sub, ses, run, task, fnum = pupil_path[1]

    task_root = in_path.split('/')[-2]
    # early mario3 runs accidentally labelled task-mariostars...
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task.replace("-fixations", "").replace("-friends", "")
    qc_path = f"{out_path}/code/QC_gaze/qc_report_{task_root}.txt"
    
    """
    TODO: write a clean-up script that will scrub 'fnum' from the final file names
    AFTER the entire dset has been Qced (fnum crucial to tell appart repeated runs...)
    """
    if run is None:
        bids_path = f'{out_path}/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{fnum}_recording-eye0_physio'
    else:
        bids_path = f'{out_path}/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{run}_{fnum}_recording-eye0_physio'

    if Path(f'{bids_path}.tsv.gz').exists():
        return np.loadtxt(
            f'{bids_path}.tsv.gz', 
            delimiter='\t',
        )
    else:
        Path(os.path.dirname(bids_path)).mkdir(parents=True, exist_ok=True)

        # gaze data includes pupil metrics from which each gaze was derived
        seri_gaze, is_deserialized = load_pldata_file(pupil_path[0], 'gaze')
        log_qc(f"\n{sub} {ses} {run} {pseudo_task} {task} {fnum} {len(seri_gaze)}", qc_path)

        if len(seri_gaze) < 13:
            if not is_deserialized:
                log_qc(f"\nRun fail: unsuccessful deserialization of .pldata for {fnum}", qc_path)
            else:
                log_qc(f"\nRun fail: {len(seri_gaze)} pupils found for {fnum}", qc_path)

            return np.array([])       
        else:
            # Get run onset time
            if task_root in ['floc', 'retino', 'langloc', 'ood', 'friends_fix', 'movie10fix']:
                tname = task
            elif task_root == 'friends':
                tname = task.replace("task-", "task-friends-")
            elif task_root == 'narratives':
                tname = f'{task.replace("part", "_part")}_run-01'
            else:
                tname = f'{task}_{run}'     
            
            onset_time, offset_time, pupil_version = get_onset_time(
                f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.log',
                tname, fnum,
                f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{tname}/000/info.player.json',
                seri_gaze[12]['timestamp'],  # updated from 10th to 12th gaze, most precise estimation in dset
                seri_gaze[-1]['timestamp'],
                qc_path,
            )

            # Extract gaze from dict to arrays 
            bids_gaze_list = extract_gaze(seri_gaze, onset_time)
            bids_gaze = np.array(bids_gaze_list)
            
            if len(bids_gaze_list) > 0:
                # Save camera freeze events (onset and duration)
                freeze_count = detect_freezes(bids_gaze, bids_path, offset_time-onset_time)

                device_name = get_device_name(
                    f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/pupil.log',
                )
                calib_vals = get_calib_vals(pupil_path[0], calib_coordinates)

                # Save timeseries and their metadata
                pd.DataFrame(bids_gaze).to_csv(
                    f'{bids_path}.tsv.gz', sep='\t', header=False, index=False, compression='gzip',
                )
                with open(f'{bids_path}.json', 'w') as metadata_file:
                    json.dump(
                        format_runwise_metadata(
                            bids_gaze_list[0][0], 
                            bids_gaze_list[-1][0] - bids_gaze_list[0][0], 
                            BIDS_COL_NAMES,
                            freeze_count, 
                            pupil_version,
                            device_name,
                            calib_vals,
                        ), metadata_file, indent=4,
                    )
            else:
                log_qc(f"Run fail: no pupils timestamped after run onset for {fnum}", qc_path)

            return bids_gaze
