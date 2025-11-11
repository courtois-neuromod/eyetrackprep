import os, glob, sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from utils import get_onset_time, parse_task_name

# TODO: fix this by importing pupil as library 
sys.path.append(
    os.path.join(
        "/home/mariestl/cneuromod/ds_prep/eyetracking",
        "pupil",
        "pupil_src",
        "shared_modules",
    )
)

from file_methods import load_pldata_file


SUB_LIST = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06']
"""
List of valid CNeuroMod subject IDs used to filter out test files.
"""

def parse_ev_dset(
    df_files: pd.DataFrame,
    task_root: str,
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
    has_run_num = False if task_root in ['retino', 'floc', 'langloc'] else True

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

                if sub in SUB_LIST:
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
                print(f'cannot process {ev_file}')

    return df_files, pupil_file_paths


def parse_noev_dset(
    df_files: pd.DataFrame,
    task_root: str,
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
        """
        task-friends-s6e8b, task-wot2, 

        Note: run number is a fluke, does not reflect the repetition... (they're all run-01...)
        task-TunnelRecency_run-01, task-PrettymouthRecency_run-01, task-PrettymouthRecall_run-01, task-SlumlordRecency_run-01, task-SlumlordRecall_run-01
        task-Tunnel_part2Story_run-01, task-Tunnel_part2Recall_run-01, task-PrettymouthStory_run-01, task-SlumlordStory_run-01

        """
        for epfile in epfile_list:
            sub, ses, fnum, task_type = parse_task_name(
                epfile, task_root,
            )
            run_num = None

            if sub not in SUB_LIST:
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
    out_path: str,
) -> tuple[pd.DataFrame, list[tuple]]:
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
    """
    # Find all the fMRI session directories with numeric session numbers for all subjects
    # e.g., (on elm, in_path = '/unf/eyetracker/neuromod/triplets/sourcedata')
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
            df_files, task_root, ses_list)
    else:
        df_files, pupil_file_paths = parse_ev_dset(
            df_files, task_root, ses_list,
        )

    # Export file list spreadsheet to support QCing
    Path(f"{out_path}/QC_gaze").mkdir(parents=True, exist_ok=True)
    df_files.to_csv(
        f"{out_path}/QC_gaze/{task_root}_QCflist.tsv",
        sep='\t', header=True, index=False,
    )

    return pupil_file_paths


def export_bids(    
    pupil_path: tuple,
    in_path: str,
    out_path: str,
    export_plots: bool,
) -> Union[np.array, None]:
    '''
    Function extracts a single run's gaze and pupil metrics from .pldata (Pupil's) format,
    and exports them to BIDS (BEP20, .tsv.gz). 
    
    Returns timestamped gaze position with detection confidence scores to plot (for manual QCing)

    Parameters
    ----------
    pupil_path : str
        Path to the directory that contains one functional runs' eye-tracking data files
    in_path : str
        The root directory of the dataset (e.g., '.../neuromod/retino/sourcedata').
    out_path : str
        The output directory where the BIDS & QC files will be saved.
    export_plots : bool
        If True, returns metrics to plot gaze confidence and position over time

    Returns
    -------
    Numpy array : run's gaze position in x and y over time, with confidence scores
    '''
    sub, ses, run, task, fnum = pupil_path[1]

    task_root = in_path.split('/')[-2]
    # early mario3 runs accidentally labelled task-mariostars...
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task

    # TODO: write a clean-up script that will scrub 'fnum' from the final file name, AFTER Qcing (fnum crucial to tell appart repeated runs...)
    if run is None:
        bids_path = f'{out_path}/bids/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{fnum}_recording-eye1_physio.tsv.gz'
    else:
        bids_path = f'{out_path}/bids/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{run}_{fnum}_recording-eye1_physio.tsv.gz'

    if not os.path.exists(bids_path):
        Path(bids_path).mkdir(parents=True, exist_ok=True)

        # gaze data includes pupil metrics from which each gaze was derived
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')[0]
        print(sub, ses, run, pseudo_task, len(seri_gaze))

        # Get run onset time
        if task_root in ['floc', 'retino', 'langloc', 'ood']:
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{task}/000/info.player.json'
        elif task_root == 'friends':
            tname = task.replace("task-", "task-friends-")
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{tname}/000/info.player.json'
        elif task_root == 'narratives':
            tname = f'{task.replace("part", "_part")}_run-01'
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{tname}/000/info.player.json'
        else:
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{task}_{run}/000/info.player.json'                
        
        onset_time = get_onset_time(
            f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.log',
            task_root, task, run,
            infoplayer_path,
            seri_gaze[10]['timestamp'],
        )

        # Convert serialized file to lists of arrays (one for BIDS, one for plotting)
        bids_gaze_list = []
        gaze_2plot_list = []

        for gaze in seri_gaze:
            # TODO: add metrics to .json sidecar
            #['timestamp', 'x_coordinate', 'y_coordinate', 'confidence',
            #'pupil.xcoordinate', 'pupil.ycoordinate', 'pupil_diameter', 
            #'pupil.elipse_axe_a', 'pupil.elipse_axe_b', 
            # 'pupil.ellipse_angle', 'pupil.ellipse_center_x', 'pupil.ellipse_center_y']
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
                    ellipse_axe_0, ellipse_axe_1, ellipse_angle, 
                    ellipse_center_x, ellipse_center_y,
                ]) 
                gaze_2plot_list.append(np.array(
                    [gaze_x, gaze_y, gaze_timestamp, gaze_conf])
                )

        bids_gaze = np.array(bids_gaze_list)
        pd.DataFrame(bids_gaze).to_csv(
            bids_path, sep='\t', header=False, index=False,
        )
        if export_plots:
            return bids_gaze, np.stack(gaze_2plot_list, axis=0)
        else:
            return bids_gaze, None


