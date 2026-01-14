import os, json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from src.utils import log_qc, get_event_path
from src.pupil2bids import BIDS_COL_NAMES


DERIV_COL_NAMES = [
    'timestamp', 'driftcorr_x_coordinate', 'driftcorr_y_coordinate', 
    'reference_fixation_idx', 'confidence', 'x_coordinate', 'y_coordinate', 
    'pupil_x_coordinate', 'pupil_y_coordinate', 'pupil_diameter',
    'pupil_ellipse_axe_a', 'pupil_ellipse_axe_b', 'pupil_ellipse_angle',
    'pupil_ellipse_center_x', 'pupil_ellipse_center_y'
]


def get_conf_thresh(
    task_root: str,
    ev_path: str,
) -> float:
    '''
    Return fixation threshold metrics for a given run or subject, 
    specified in task-specific config file.
    
    Returns:
        [float, float, float]: [
            pupil detection confidence threshold,
            minimum ratio of above-threshold gaze recorded over fixation period,
            maximal variation in gaze position throughout fixation (derived from stdev in x and y),
        ] 
    '''
    with open(f'./config/{task_root}.json', 'r') as conf_file:
        conf_dict = json.load(conf_file)

    sub, ses, fnum = ev_path.split('_')[:3]
    run = ev_path.split('task-')[-1].replace('_events.tsv', '')

    return conf_dict.get(sub, {}).get(ses, {}).get(fnum, {}).get(run, conf_dict[sub]['default'])


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
        Columns 0-3 are [gaze_timestamp, gaze_dist-x, gaze_dist-y, gaze_conf] 

    """    
    filtered_gaze = bids_gaze[bids_gaze[:, 3] > gaze_threshold][:, :4]

    if distance:
        filtered_gaze[:, 1:3] = filtered_gaze[:, 1:3] - 0.5

    return filtered_gaze
    

def get_fixations(
    df_ev: pd.DataFrame,
    task: str,
    gaze_arr: np.array,
    gaze_ratio: float,
    dist_cutoff: float,
) -> tuple[np.array, int]:
    """
    Identifies gaze data corresponding to fixation periods and calculates their 
    timing and median distance to the central fixation marker, in x and y. 

    Filters gaze within windows of fixation defined by task structure events.tsv files, 
    (applying a +0.7s start  buffer), and computes the median 
    gaze distance to center in x and y. The onset time assigned to each fixation 
    corresponds to the first gaze reccorded within a fixation window.

    Args:
        df_ev (pd.DataFrame): Run's event.tsv dataframe with timing and trial info. 
            Required columns depend on `task`:
            - 'emotionsvideos': 'onset_fixation_flip', 'onset_video_flip'
            - 'langloc': 'trial_type', 'onset', 'duration'
            - 'mario*': 'trial_type', 'onset', 'duration'
            - 'triplets': 'onset', 'duration'
        task (str): The experimental task name. Supported tasks: 'emotionsvideos', 
            'langloc', 'mariostars', 'mario3', 'multfs', 'mutemusic', 'triplets'. 
        gaze_arr (np.array): A numpy array where column 0 is the timestamp, 
            columns 1 and 2 are the distance to center in X and Y, and column 4 is 
            the pupil detection confidence.
        gaze_ratio : float
            Minimal ratio of above-threshold gaze (out of total expected gaze points)
            recorded over fixation period. To filter out unreliable fixations.
        dist_cutoff : float
            Maximal variability in position between gaze captured throughout fixation. 
            Composite metric estimated from stdev in x and y. Used to filter out 
            unreliable fixations.

    Returns:
        np.array(fix_data): A (N, 7) array containing the following columns
            for each period of fixation with reccorded gaze: [onset, median gaze dist-x, 
            median gaze dist-y, duration, pupil_counts,  stdev gaze x, stdev gaze y]. 
            Returns an empty array if no high-confidence fixations are identified.
        total_fix (int): 
            Total number of fixation periods in the run (with or without high-confidence gaze points).
    """
    fix_data = []  # [onset, x, y, duration, count, stdev x, stdev y]
    total_fix = 0

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

        elif task == 'mutemusic':
            fix_offset = df_ev['onset'][i]
            fix_onset = fix_offset - 6.0 if i == 0 else fix_offset - 2.0

        elif task == 'multfs':
            fix_offset = df_ev['stimulus_0_onset'][i]
            fix_onset = fix_offset - 6.0 if i == 0 else fix_offset - 3.98

        if row_has_fix:
            """
            Select gaze from pre-trial fixation period
            """
            fix_min = int(250*(fix_offset - (0.7 + fix_onset))*gaze_ratio)  # default : 0.2. 20% of pupils expected to be captured during fix period
            total_fix += 1
            trial_gaze = gaze_arr[np.logical_and(
                gaze_arr[:, 0] > (fix_onset + 0.7),   # + capture from 0.7s (700ms) after fixation onset to account for saccade
                gaze_arr[:, 0] < fix_offset,  # gaze_arr[:, 0] < (fix_offset - 0.1),  # drop last 0.1s of fix
            )]
            if trial_gaze.shape[0] > fix_min:
                if np.sqrt(np.std(trial_gaze[:, 1])**2 + np.std(trial_gaze[:, 2])**2) < dist_cutoff:
                    fix_data.append([
                        trial_gaze[0, 0],
                        np.median(trial_gaze[:, 1]),
                        np.median(trial_gaze[:, 2]),                    
                        trial_gaze[-1, 0] - trial_gaze[0, 0],
                        trial_gaze.shape[0],
                        np.std(trial_gaze[:, 1]),
                        np.std(trial_gaze[:, 2]),                    
                    ])

    return np.array(fix_data), total_fix


def driftcorr_fromlast(
    fix_data: np.array,
    bids_gaze: np.array,
) -> np.array:
    """
    Corrects for gaze drift from last period of known central fixation. 

    For each gaze, correct drift based on the median gaze distance to center for a 
    reference period of central fixation (ideally the latest fixation that 
    precedes the trial, with some robustness built in for periods of missing
    or low quality gaze).

    Subtract the distance between the fixation coordinates and the screen center 
    (drift estimation) from the mapped gaze position in x and y. 
    Also returns the reference fixation index.
    """
    gaze_aligned = []
    j = 0

    for i in range(len(bids_gaze)):
        while j < len(fix_data)-1 and bids_gaze[i, 0] > fix_data[j+1, 0]:
            j += 1
        gaze_aligned.append([
            bids_gaze[i, 1] - fix_data[j, 1],  # driftcorr gaze x_coord 
            bids_gaze[i, 2] - fix_data[j, 2],  # driftcorr gaze y_coord
            j,   # ref fixation index
        ])

    return np.array(gaze_aligned)


def format_dset_metadata(
    deriv_dir: str,
)-> None:
    """."""
    dset_name = os.path.basename(deriv_dir).split(".")[0]
    with open(f'{deriv_dir}/task-{dset_name}_recording-eye0_physioevents.json', 'w') as metadata_file:
        json.dump({
                "Columns": ['onset', 'median_distance_x', 'median_distance_y', 'duration', 'pupil_count', 'stdev_distance_x', 'stdev_distance_y'],
                "Description": "Known periods of fixations used to correct drift in gaze mapping.",
                "OnsetSource": "timestamp",
                "onset": {
                    "Description": "Onset of the reference fixation period.",
                    "Units": "seconds",
                },
                "median_distance_x": {
                    "LongName": "Median distance to center (x)",
                    "Description": "Median gaze distance in x to the central fixation marker during the reference fixation period, normalized as a proportion of the screen width. Bound = [-0.5, 0.5], where -0.5 = left edge of the screen.",
                    "Units": "arbitrary",
                },
                "median_distance_y": {
                    "LongName": "Median distance to center (y)",
                    "Description": "Median gaze distance in y to the central fixation marker during the reference fixation period, normalized as a proportion of the screen height. Bound = [-0.5, 0.5], where -0.5 = bottom edge of the screen.",
                    "Units": "arbitrary",
                },
                "duration": {
                    "Description": "Difference between the timestamps of the last and the first high confidence gaze sampled during the reference fixation period.",
                    "Units": "seconds",
                },
                "pupil_count": {
                    "Description": "Number of gaze points derived from pupils detected with above-threshold confidence used to calculate median gaze coordinates during the reference fixation period.",
                    "Units": "count",
                },
                "stdev_distance_x": {
                    "LongName": "Standard deviation of distance to center (x)",
                    "Description": "Standard deviation of gaze distance in x to the central fixation marker during the reference fixation period, normalized as a proportion of the screen width. Bound = [-0.5, 0.5], where -0.5 = left edge of the screen.",
                    "Units": "arbitrary",
                },
                "stdev_distance_y": {
                    "LongName": "Standard deviation of distance to center (y)",
                    "Description": "Standard deviation of gaze distance in y to the central fixation marker during the reference fixation period, normalized as a proportion of the screen height. Bound = [-0.5, 0.5], where -0.5 = bottom edge of the screen.",
                    "Units": "arbitrary",
                }
            }, metadata_file, indent=4,
        )


def format_runwise_metadata(
    start_time: float,
    col_names: list[str],
    gaze_threshold: float,
    gaze_ratio: float,
    dist_cutoff: float,
    hcgaze_count: int,
    gaze_count: int,
    hcfix_count: int,
    fix_count: int,
) -> None :
    """."""
    return {
        "StartTime": start_time,
        "Columns": col_names,
        "PhysioType": "eyetrack",
        "RecordedEye": "right",
        "SamplingFrequency": 250.0,
        "SampleCoordinateSystem": "gaze-on-screen",
        "DriftCorrectionMethod": "Latest Fixation",
        "PupilConfidenceThreshold": gaze_threshold,
        "MinProportionGazePerFix": gaze_ratio,
        "MaxGazeFixVariability": dist_cutoff,
        "HighConfidenceGazeCount": hcgaze_count,
        "TotalGazeCount": gaze_count,
        "HighConfidenceFixationCount": hcfix_count,
        "TotalFixationCount": fix_count,
        "timestamp": {
            "Description": "A continuously increasing identifier of the sampling time registered by the device",
            "Units": "seconds",
            "Origin": "run trigger onset",
        },
        "driftcorr_x_coordinate": {
            "LongName": "Drift-corrected gaze position (x)",
            "Description": "Gaze position x-coordinate of the recorded eye, normalized as a proportion of the screen width and corrected for drift based on the lastest period of central fixation. Bound = [0, 1], where 0 = left.",
            "Units": "arbitrary",
        },
        "driftcorr_y_coordinate": {
            "LongName": "Drift-corrected gaze position (y)",
            "Description": "Gaze position y-coordinate of the recorded eye, normalized as a proportion of the screen height and corrected for drift based on the lastest period of central fixation. Bound = [0, 1], where 0 = bottom.",
            "Units": "arbitrary",
        },
        "reference_fixation_idx": {
            "Description": "Row index in the run's physioevents.tsv.gz file corresponding to the fixation period used to drift correct gaze x and y coordinates.",
            "Units": "row number",
        },
        "confidence": {
            "Description": "Quality assessment of the pupil detection ranging from 0 to 1. A value of 0 indicates that the pupil could not be detected, whereas a value of 1 indicates a very high pupil detection certainty.",
            "Units": "ratio",
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


def export_dcgaze(
    bids_gaze: np.array,
    driftcorr_gaze: np.array,
    deriv_path: str,
) -> np.array:
    """."""

    deriv_gaze_df = pd.DataFrame(bids_gaze, columns=BIDS_COL_NAMES)
    deriv_gaze_df.insert(loc=1, column='driftcorr_x_coordinate', value=driftcorr_gaze[:, 0])
    deriv_gaze_df.insert(loc=2, column='driftcorr_y_coordinate', value=driftcorr_gaze[:, 1])
    deriv_gaze_df.insert(loc=3, column='reference_fixation_idx', value=driftcorr_gaze[:, 2])
    deriv_gaze_df = deriv_gaze_df[DERIV_COL_NAMES]
    
    deriv_gaze_df.to_csv(
        f'{deriv_path}.tsv.gz', sep='\t', header=False, index=False, compression='gzip',
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
        gaze_threshold, gaze_ratio, dist_cutoff = get_conf_thresh(
            task_root,
            os.path.basename(events_path),
        )
        clean_gaze = filter_gaze(bids_gaze, gaze_threshold, distance=True)
        log_qc(f"{len(clean_gaze)} out of {len(bids_gaze)} ({(100*len(clean_gaze))/len(bids_gaze):.2f}%) of gaze above {gaze_threshold} confidence for {fnum}", qc_path)

        if len(clean_gaze) == 0:
            log_qc(f"Drift correction fail: no gaze found above confidence threshold for {fnum}", qc_path)
            return bids_gaze, None, None
        else:
            """
            Computes median distance to center in x and y for known periods of central fixation;
            fix_data columns = [timestamp, median gaze dist-x, median gaze dist-y,
                                duration, pupil_counts,  stdev gaze x, stdev gaze y]
            """
            fix_data, total_fix = get_fixations(
                run_event,
                task_root,
                clean_gaze,
                gaze_ratio,
                dist_cutoff,
            )
            log_qc(f"{fix_data.shape[0]} out of {total_fix} fixations valid for {fnum}", qc_path)

            if fix_data.shape[0] == 0:
                log_qc(f"Drift correction fail: no fixation periods with high-confidence gaze for {fnum}", qc_path)
                return bids_gaze, None, None

            else:
                """
                Export fixation metrics
                """
                pd.DataFrame(fix_data).to_csv(
                    f'{deriv_path}events.tsv.gz', sep='\t', 
                    header=False, index=False, compression='gzip',
                )

                """
                Use median gaze distance to center from latest period of 
                central fixation (with enough high-confidence gaze points) 
                to drift-correct every gaze in the run. 
                """
                driftcorr_gaze = driftcorr_fromlast(
                    fix_data, bids_gaze,
                )
                
                """
                Export gaze metadata 
                """
                with open(f'{deriv_path}.json', 'w') as metadata_file:
                    json.dump(format_runwise_metadata(
                        bids_gaze[0, 0], DERIV_COL_NAMES, 
                        gaze_threshold, gaze_ratio, dist_cutoff,
                        len(clean_gaze), len(bids_gaze),
                        fix_data.shape[0], total_fix,                          
                    ), metadata_file, indent=4)

                """
                Export and return corrected gaze 
                """
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

        if bids_gaze.shape[0] == 0:
            log_qc(f"Drift correction fail: no gaze found for {fnum}", dcqc_path)
            return bids_gaze, None, None

        elif task_root in ['emotionsvideos', 'langloc', 'mariostars', 'mario3', 'multfs', 'mutemusic', 'triplets']:
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
