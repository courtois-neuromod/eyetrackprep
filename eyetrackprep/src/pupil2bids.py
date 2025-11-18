import os, glob, sys, json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from bids import BIDSLayout

from src.utils import (
    get_onset_time, 
    parse_task_name, 
    load_pldata_file, 
    extract_gaze,
    get_metadata, 
    log_qc,
)


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

    # Export file list spreadsheet to support QCing
    Path(f"{out_path.root}/code/QC_gaze").mkdir(parents=True, exist_ok=True)

    df_files.to_csv(
        f"{out_path.root}/code/QC_gaze/{task_root}_QCflist.tsv",
        sep='\t', header=True, index=False,
    )
    Path(f"{out_path.root}/code/QC_gaze/qc_report_{task_root}.txt").touch()

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
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task.replace("-fixations", "").replace("-friends", "")
    qc_path = f"{out_path}/code/QC_gaze/qc_report_{task_root}.txt"
    
    """
    TODO: write a clean-up script that will scrub 'fnum' from the final file names
    AFTER the entire dset has been Qced (fnum crucial to tell appart repeated runs...)
    """
    if run is None:
        bids_path = f'{out_path}/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{fnum}_recording-eye_physio'
    else:
        bids_path = f'{out_path}/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{run}_{fnum}_recording-eye_physio'

    if not Path(bids_path).exists():
        Path(os.path.dirname(bids_path)).mkdir(parents=True, exist_ok=True)

        # gaze data includes pupil metrics from which each gaze was derived
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')
        log_qc(f"\n{sub} {ses} {run} {pseudo_task} {task} {len(seri_gaze)}", qc_path)

        if len(seri_gaze) < 13:
            log_qc(f"Run fail: {len(seri_gaze)} pupils found for {fnum}", qc_path)

            return np.array([]), np.array([])        
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
            
            onset_time = get_onset_time(
                f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.log',
                tname, fnum,
                f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{tname}/000/info.player.json',
                seri_gaze[12]['timestamp'],  # updated from 10th to 12th gaze, most precise estimation in dset
                qc_path,
            )

            # Extract gaze from dict to arrays 
            bids_gaze_list, gaze_2plot_list = extract_gaze(
                seri_gaze, onset_time, export_plots)
            
            bids_gaze = np.array(bids_gaze_list)
            
            if len(bids_gaze_list) > 0:
                # Save timeseries and their metadata
                pd.DataFrame(bids_gaze).to_csv(
                    f'{bids_path}.tsv.gz', sep='\t', header=False, index=False,
                )

                with open(f'{bids_path}.json', 'w') as metadata_file:
                    json.dump(
                        get_metadata(bids_gaze_list[0][0]), metadata_file, indent=4,
                    )
            else:
                log_qc(f"Run fail: no pupils timestamped after run onset for {fnum}", qc_path)

            return bids_gaze, np.array(gaze_2plot_list)
