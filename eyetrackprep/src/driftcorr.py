import os, json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.utils import log_qc, get_event_path, get_metadata
from src.pupil2bids import BIDS_COL_NAMES


DERIV_COL_NAMES = [
    'timestamp', 'driftcorr_x_coordinate', 'driftcorr_y_coordinate', 
    'confidence', 'x_coordinate', 'y_coordinate', 
    'pupil_x_coordinate', 'pupil_y_coordinate', 'pupil_diameter',
    'pupil_ellipse_axe_a', 'pupil_ellipse_axe_b', 'pupil_ellipse_angle',
    'pupil_ellipse_center_x', 'pupil_ellipse_center_y'
]

def filter_gaze(
    bids_gaze: np.array,
    gaze_threshold: float,
    distance: bool=True,
) -> np.array:
    """
    Filters a run's gaze above specified pupil detection confidence threshold. 

    Parameters
    ----------
    bids_gaze : np.array
        Array of gaze and pupil data to export to/import from bids-compliant dataset.
        bids_gaze columns 0-3 are [gaze_timestamp, gaze_x, gaze_y, gaze_conf] 
    gaze_threshold : float
        Pupil detection confidence threshold to filter out run's low confidence gaze
    distance: bool
        If False, returns the gaze's normalized position (expressed as a proportion 
        of the screen). 
        If True, returns the normalized distance between the gaze and the point 
        of central fixation.

    Returns
    -------
    Numpy array
        Array of high-confidence timestamped gaze coordinates (or distances to 
        central fixation) and their confidence score. 
        Columns 0-3 are [gaze_timestamp, gaze_x, gaze_y, gaze_conf] 

    """    
    filtered_gaze = bids_gaze[bids_gaze[:, 3] > gaze_threshold][:, :4]

    if distance:
        filtered_gaze[:, 1:3] = filtered_gaze[:, 1:3] - 0.5

    return filtered_gaze
    

def get_fixations(
    df_ev: pd.DataFrame,
    task: str,
    gaze_arr: np.array,
) -> tuple[np.array, int, int]:
    """
    Identifies gaze data corresponding to fixation periods and calculates their 
    median position. 

    Filters gaze within windows of fixation defined by task structure events.tsv files, 
    (applying a +0.8s start  buffer and -0.1s end buffer), and computes the median 
    gaze coordinates in x and y. The timestamp assigned to each fixation corresponds 
    to the first gaze reccorded within a fixation window.

    Args:
        df_ev (pd.DataFrame): Run's event.tsv dataframe with timing and trial info. 
            Required columns depend on `task`:
            - 'emotionsvideos': 'onset_fixation_flip', 'onset_video_flip'
            - 'langloc': 'trial_type', 'onset', 'duration'
            - 'mario*': 'trial_type', 'onset', 'duration'
            - 'triplets': 'onset', 'duration'
        task (str): The experimental task name. Supported tasks: 'emotionsvideos', 
            'langloc', 'mariostars', 'mario3', 'triplets'. 
            (Note: 'multfs' and 'mutemusic' are planned but not implemented).
        gaze_arr (np.array): A numpy array where column 0 is the timestamp, 
            columns 1 and 2 are the X and Y coordinates, and column 4 is 
            the pupil detection confidence (dropped in output).

    Returns:
        np.array(fix_data): A (N, 3) array containing the [onset_time, median gaze x, median gaze y] 
            for each period of fixation with reccorded gaze. Returns an empty array if 
            no fixations are found.
        total_fix (int): 
            Total number of fixation periods in the run's task
        good_fix (int): 
            Total number of fixation periods with recorded high-confidence gaze 

    """
    fix_data = []  # [time, x, y]
    total_fix = 0
    good_fix = 0

    for i in range(df_ev.shape[0]):
        row_has_fix = True

        if task == "emotionsvideos":
            fix_onset = df_ev['onset_fixation_flip'][i]
            fix_offset = df_ev['onset_video_flip'][i]

        elif task == 'langloc':
            if df_ev['trial_type'][i] != 'fix':
                row_has_fix = False
            else:
                fix_onset = df_ev['onset'][i]
                fix_offset = fix_onset + df_ev['duration'][i]

        elif 'mario' in task:
            if df_ev['trial_type'][i] != 'fixation_dot':
                row_has_fix = False
            else:
                fix_onset = df_ev['onset'][i]
                fix_offset = fix_onset + df_ev['duration'][i]

        elif task == 'triplets':
            fix_onset = df_ev['onset'][i] - 3.0 if i == 0 else df_ev['onset'][i-1] + df_ev['duration'][i-1]
            fix_offset = df_ev['onset'][i]

        # TODO: add fix onset and offset for multfs and mutemusic, drive from task_stimuli library and events.tsv files

        if row_has_fix:
            """
            Select gaze from pre-trial fixation period
            """
            total_fix += 1
            trial_gaze = gaze_arr[np.logical_and(
                gaze_arr[:, 0] > (fix_onset + 0.8),   # + capture from 0.8s (800ms) after fixation onset to account for saccade
                gaze_arr[:, 0] < (fix_offset - 0.1),  # drop last 0.1s of fix
            )]
            if trial_gaze.shape[0] > 0:
                # TODO: consider setting a min number of gaze >1 to estimate fixation
                good_fix += 1
                trial_gaze[:, 0] = trial_gaze[0, 0]
                fix_data.append(np.median(trial_gaze[:, :3], axis=0))

    return np.array(fix_data), total_fix, good_fix


def driftcorr_fromlast(
    fix_data: np.array,
    bids_gaze: np.array,
) -> np.array:
    """
    Corrects for gaze drift from last period of known central fixation. 

    For each gaze, correct drift based on the median gaze position for a 
    reference period of central fixation (ideally the latest fixation that 
    precedes the trial, with some robustness built in for periods of missing
    or low quality gaze).

    Subtract the difference between 
    by subtracting the difference between 
    """
    gaze_aligned = []
    j = 0

    for i in range(len(bids_gaze)):
        while j < len(fix_data)-1 and bids_gaze[i, 0] > fix_data[j+1, 0]:
            j += 1
        gaze_aligned.append(bids_gaze[i, 1:3] - fix_data[j, 1:3])

    return np.array(gaze_aligned)


def export_dcgaze(
    bids_gaze: np.array,
    driftcorr_gaze: np.array,
    deriv_path: str,
) -> np.array:
    """."""

    deriv_gaze_df = pd.DataFrame(bids_gaze, columns=BIDS_COL_NAMES)
    deriv_gaze_df.insert(loc=1, column='driftcorr_x_coordinate', value=driftcorr_gaze[:, 0])
    deriv_gaze_df.insert(loc=2, column='driftcorr_y_coordinate', value=driftcorr_gaze[:, 1])
    deriv_gaze_df = deriv_gaze_df[DERIV_COL_NAMES]
    
    deriv_gaze_df.to_csv(
        f'{deriv_path}.tsv.gz', sep='\t', header=False, index=False, compression='gzip',
    )

    with open(f'{deriv_path}.json', 'w') as metadata_file:
        json.dump(
            get_metadata(bids_gaze[0, 0], DERIV_COL_NAMES), metadata_file, indent=4,
        )

    return deriv_gaze_df.to_numpy()


def dc_knownfix(
    bids_gaze: np.array,
    task_root: str,
    fnum: str,
    events_path: str, 
    deriv_path: str,
    qc_path: str,
) -> tuple[np.array, Union[np.array, None], Union[np.array, None]]:
    """
    Tasks with known periods of central fixation
    Gaze drift corrected based on the latest period of central fixation between trials, levels, videos, etc.

    Each run has one event.tsv file with logged fixation onset and duration in bids_dir
    """
    if not Path(events_path).exists():
        log_qc(f"Drift correction fail: no events.tsv file found for {fnum}", qc_path)
        return bids_gaze, None, None
    else:
        run_event = pd.read_csv(events_path, sep = '\t', header=0)
    
        """
        Filter-out gaze from pupils detected below confidence threshold
        Returns normalized distance to central fixation point
        """
        # TODO: add argument to specify conf thresholds per subject or (better) per run...
        gaze_threshold = 0.85  # Note: used 0.75 for sub-01 for THINGS task, 0.9 for other subjects...
        clean_gaze = filter_gaze(bids_gaze, gaze_threshold, distance=True)
        log_qc(f"{len(clean_gaze)} out of {len(bids_gaze)} ({(100*len(clean_gaze))/len(bids_gaze):.2f}%) of gaze above {gaze_threshold} confidence for {fnum}", qc_path)

        if len(clean_gaze) == 0:
            log_qc(f"Drift correction fail: no gaze found above confidence threshold for {fnum}", qc_path)
            return bids_gaze, None, None
        else:
            """
            Computes median position in x and y for known periods of central fixation;
            fix_data columns = [timestamp, median gaze x, median gaze y]
            """
            fix_data, total_fix, valid_fix = get_fixations(
                run_event,
                task_root,
                clean_gaze,
            )
            log_qc(f"{valid_fix} out of {total_fix} fixations valid for {fnum}", qc_path)

            if valid_fix == 0:
                log_qc(f"Drift correction fail: no fixation periods with high-confidence gaze for {fnum}", qc_path)
                return bids_gaze, None, None

            else:
                """
                Use median gaze position from latest period of central fixation with 
                gaze to drift-correct every gaze in the run. 
                """
                driftcorr_gaze = driftcorr_fromlast(
                    fix_data,
                    bids_gaze,
                )

                return export_dcgaze(
                    bids_gaze, driftcorr_gaze, deriv_path,
                ), clean_gaze, fix_data


def driftcorr_run(
    bids_gaze: np.array,
    task_root: str,
    meta_data: tuple[str, tuple], 
    deriv_dir: str,   
) -> tuple[np.array, Union[np.array, None], Union[np.array, None]]:
    """
    Assigns gaze to proper drift-correction strategy based on task
    """
    sub, ses, run, task, fnum = meta_data[1]
    task = 'task-mario3' if task_root == 'mario3' else task.replace("-fixations", "").replace("-friends", "")

    if run is None:
        deriv_path = f'{deriv_dir}/{sub}/{ses}/func/{sub}_{ses}_{task}_{fnum}_recording-eye0_desc-driftcorr_physio'
    else:
        deriv_path = f'{deriv_dir}/{sub}/{ses}/func/{sub}_{ses}_{task}_{run}_{fnum}_recording-eye0_desc-driftcorr_physio'

    if Path(f'{deriv_path}.tsv.gz').exists():
        return np.loadtxt(
            f'{deriv_path}.tsv.gz', delimiter='\t',
        ), None, None
    else:
        Path(os.path.dirname(deriv_path)).mkdir(parents=True, exist_ok=True)
        dcqc_path = f"{deriv_dir}/code/QC_gaze/qc_report_{task_root}.txt"
        log_qc(f"\n{sub} {ses} {run} {task} {fnum}", dcqc_path)

        if task_root in ['emotionsvideos', 'langloc', 'mariostars', 'mario3', 'multfs', 'mutemusic', 'triplets']:
            """
            Tasks with known periods of central fixation
            Gaze drift corrected based on the latest period of central fixation between trials, levels, videos, etc.
            """    
            return dc_knownfix(
                bids_gaze, task_root, fnum,
                get_event_path(meta_data[0]), 
                deriv_path, dcqc_path,
            )

        elif task_root in ['floc', 'retino', 'things']:
            """
            Tasks with central fixation sustained throughout the run
            Gaze drift corrected based on arbitrarily defined periods of fixation before the run / sequence onset.
            """        
            # TODO: implement dc_stablefix
            return bids_gaze, None, None

        elif task_root in ['friends', 'ood']:
            """
            Movie-watching tasks
            Gaze drift corrected based on distribution of gaze prediction from deepgaze
            """        
            # TODO: implement dc_deepgaze
            return bids_gaze, None, None

        elif task_root in ['friends_fix', 'movie10_fix', 'mario', 'narratives']:
            """
            Tasks for which there is no current approach to drift correction
            """        
            return bids_gaze, None, None   
