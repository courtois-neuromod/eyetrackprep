import os, glob, sys
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from utils import get_onset_time

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
    (e.g., movie  tasks like friends, movie10, ood).

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

    pupil_file_paths = []

    for ses_path in ses_list:
        [sub_num, ses_num] = ses_path.split('/')[-2:]

        events_list = sorted(glob.glob(f'{ses_path}/*task*events.tsv'))
        for event in events_list:
            ev_file = os.path.basename(event)

            try:
                if task_root in ['retino', 'floc']:
                    skip_run_num = True
                    sub, ses, fnum, task_type, appendix = ev_file.split('_')
                    run_num = RUN2TASK_MAPPING[task_root][task_type]
                else:
                    skip_run_num = False
                    sub, ses, fnum, task_type, run_num, appendix = ev_file.split('_')

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

                    if skip_run_num:
                        pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}/000'
                    else:
                        pupil_path = f'{ses_path}/{sub_num}_{ses_num}_{fnum}.pupil/{task_type}_{run_num}/000'

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
                        sub_num,
                        ses_num,
                        run_num,
                        task_type,
                        fnum,
                        has_pupil,
                        has_gaze,
                        has_eyemv,
                        has_log,
                        empty_log,
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
    ses_list: list,
) -> tuple[pd.DataFrame, list[tuple]]:
"""
    Processes the raw 'friends' dataset to compile a DataFrame overview
    of available eye-tracking files.

    This task does not rely on events.tsv files and requires parsing 
    run details from the pupil data directory path.

    Parameters
    ----------
    df_files : pandas.DataFrame
        The cumulative DataFrame to append run data to. Must have columns = [
        'subject', 'session', 'run', 'task', 'file_number', 'has_pupil', 
        'has_gaze', 'has_eyemovie', 'has_log', 'empty_log'].
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
                f'{ses_path}/sub-*.pupil/task-friends-*/000'
            )
        )

        for epfile in epfile_list:
            f1, f2 = epfile.split('/')[-3:-1]
            t1, t2, run_num = f2.split('-')
            task_type = f'{t1}-{t2}'
            sub, ses = f1.split('.')[0].split('_')[:2]

            fnum = f1.split('.')[0].split('_')[-1]

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
                sub_num,
                ses_num,
                run_num,
                task_type,
                fnum,
                has_pupil,
                has_gaze,
                has_eyemv,
                has_log,
                empty_log,
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
    for a given CNeuroMod dataset and exports it as a TSV file 
    for quality control (QC).

    The availability of key output files (Pupil: pupil.pldata, gaze.pldata, eye0.mp4;
    PsychoPy: log file) is checked for every identified run.

    Parameters
    ----------
    in_path : str
        The root directory of the dataset (e.g., '.../neuromod/retino/sourcedata').
    out_path : str
        The output directory where the QC file and plots will be saved.

    Returns
    -------
    tuple
        - df_files : pandas.DataFrame
            The DataFrame containing file availability info for all processed runs.
        - pupil_file_paths : list of tuple
            A list of tuples: (pupil_data_directory_path, metadata_tuple)
            for all runs that contain a 'pupil.pldata' file.
    """
    col_names = [
        'subject', 'session', 'run', 'task', 'file_number', 
        'has_pupil', 'has_gaze', 'has_eyemovie',
        'has_log', 'empty_log', 
    ]
    df_files = pd.DataFrame(columns=col_names)

    task_root = in_path.split('/')[-2]

    # Find all subject/session directories with numeric session numbers
    # e.g., (on elm, in_path = '/unf/eyetracker/neuromod/triplets/sourcedata')
    ses_list = [
        x for x in sorted(glob.glob(
            f'{in_path}/sub-*/ses-*')
        ) if x.split('-')[-1].isnumeric()
    ]

    # Friends has no events files, files are parsed differently 
    # TODO: expand script to newer tasks w/o events like narratives, multfs... (check structure)
    if task_root == 'friends':
        df_files, pupil_file_paths = parse_noev_dset(
            df_files, ses_list)
    else:
        df_files, pupil_file_paths = parse_ev_dset(
            df_files, task_root, ses_list,
        )

    # Export file list spreadsheet for QCing
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

    task_root = out_path.split('/')[-1]
    # early mario3 runs accidentally labelled task-mariostars
    pseudo_task = 'task-mario3' if task_root == 'mario3' else task

    bids_path = f'{out_path}/bids/{sub}/{ses}/func/{sub}_{ses}_{pseudo_task}_{run}_{fnum}_recording-eye1_physio.tsv.gz'
    Path(bids_path).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(bids_path):
        # gaze data includes pupil metrics from which each gaze was derived
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')[0]
        print(sub, ses, run, pseudo_task, len(seri_gaze))

        # Get run's onset time
        if task in ['task-bar', 'task-rings', 'task-wedges', 'task-flocdef', 'task-flocalt']:
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{task}/000/info.player.json'
        else:
            infoplayer_path = f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{task}_{run}/000/info.player.json'                
        
        onset_time = get_onset_time(
            f'{in_path}/{sub}/{ses}/{sub}_{ses}_{fnum}.log',
            run,
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
            #'pupil.elipse_axes', 'pupil.ellipse_angle', 'pupil.ellipse_center']
            gaze_timestamp = gaze['timestamp'] - onset_time
            if gaze_timestamp > 0.0:
                gaze_x, gaze_y = gaze['norm_pos']
                gaze_conf = gaze['confidence']
                #gaze_model = gaze['topic']
                pupil_data = gaze['base_data'][0]
                pupil_x, pupil_y = pupil_data['norm_pos']
                pupil_diameter = pupil_data['diameter']
                ellipse_data = pupil_data['ellipse']
                ellipse_axe_0, ellipse_axe_1 = ellipse_data['axes']
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

        pd.DataFrame(np.array(bids_gaze_list)).to_csv(
            bids_path, sep='\t', header=False, index=False,
        )
        if export_plots:
            return np.stack(gaze_2plot_list, axis=0)
        else:
            return None


