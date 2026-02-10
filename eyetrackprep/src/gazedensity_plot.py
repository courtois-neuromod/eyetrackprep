import click
import os, glob, sys
from pathlib import Path

import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial.distance import pdist

from src.utils import parse_file_name

"""

numpy version:  1.24.4
pandas version:  1.3.5
matplotlib version:  3.7.5
seaborn version:  0.11.2

THINGS: analyze fixations
https://github.com/courtois-neuromod/things.behaviour/blob/b84bf6d5c18e53e78c6278bfcb5d4e0e6afff214/code/analyze_fixations.py

notebook: gaze density plot
https://github.com/courtois-neuromod/cneuromod-things/blob/main/datapaper/notebooks/fixation_compliance.ipynb

function: export gaze during trial, skip ISI period (used as reference)

params: plot per subject, plot per run (one subject), plot per trial (one run)...
how much to downsample? 


Compile tsv of concat data per subject and save (use caching...?)
"""


def get_event_path(
    et_path: str,
    bids_dir: str,
) -> str:
    """
    Find eyetracking file's corresponding events file.
    """
    eb = parse_file_name(os.path.basename(et_path))
    # TODO: add robustness to ses and run number padding... possibly task names...
    if 'run' in eb:
        ev_paths = glob.glob(
            f'{bids_dir}/sub-{eb["sub"]}/ses-{eb["ses"]}/sub-{eb["sub"]}_ses-{eb["ses"]}_'
            f'{eb["fnum"]}_task-{eb["task"]}_run-{eb["run"]}_events.tsv'
        )
    else:
        ev_paths = glob.glob(
            f'{bids_dir}/sub-{eb["sub"]}/ses-{eb["ses"]}/sub-{eb["sub"]}_ses-{eb["ses"]}_'
            f'{eb["fnum"]}_task-{eb["task"]}_events.tsv'
        )

    if len(ev_paths) == 1:
        return ev_paths[0]

    return None


def get_degrees(x, y):
    '''
    converts normalized coordinates x and y into degrees of visual angle, 
    and calculate gaze distance from central fixation point
    '''
    assert len(x) == len(y)

    dist_in_pix = 4164 # in pixels
    m_vecpos = np.array([0., 0., dist_in_pix])

    all_pos = np.stack((x, y), axis=1)
    gaze_in_deg = (all_pos - 0.5)*(17.5, 14.0)

    gaze = (all_pos - 0.5)*(1280, 1024)
    gaze_vecpos = np.concatenate((gaze, np.repeat(dist_in_pix, len(gaze)).reshape((-1, 1))), axis=1)

    all_distances = []
    for gz_vec in gaze_vecpos:
        vectors = np.stack((m_vecpos, gz_vec), axis=0)
        distance = np.rad2deg(np.arccos(1.0 - pdist(vectors, metric='cosine')))[0]
        all_distances.append(distance)

    return gaze_in_deg[:, 0].tolist(), gaze_in_deg[:, 1].tolist(), all_distances


def get_trial_times(
    df_ev: pd.DataFrame,
    ev_path: str,
    i: int,
) -> tuple[float, float]:
    """
    Retrieve trial onset and offset times (from run onset) from run's events.tsv file
    """

    if 'emotionvideos' in ev_path:
        trial_onset = df_ev['onset_video_flip'][i] 
        trial_offset = trial_onset + df_ev['total_duration'][i]
    elif 'multfs' in ev_path:
        trial_onset = df_ev['stimulus_0_onset'][i] 
        trial_offset = df_ev[sorted([
            x for x in df_ev.columns() if '_offset' in x
        ])[-1]][i]
    else:
        # THINGS, langloc, things, triplets... # not yet fLoc, retino
        trial_onset = df_ev['onset'][i]
        if 'mario' in ev_path:
            trial_offset = df_ev['onset'][i + 1]
        elif 'mutemusic' in ev_path:
            trial_offset = trial_onset + df_ev['total_duration'][i]
        else:
            trial_offset = trial_onset + df_ev['duration'][i]

    return trial_onset, trial_offset


def format_gaze_data(
    df: pd.DataFrame,
    eb: dict,
    conf_thresh: float,
    trial_num=None,
) -> pd.DataFrame:
    """."""

    # filter out gaze below confidence threshold; # col_4 = timestamp
    df = df[df.iloc[:, 4].to_numpy() > conf_thresh]

    # convert gaze positions to degrees of visual angle (dist from center screen)
    x_deg, y_deg, dist_deg = get_degrees(
        df.iloc[:, 1].tolist(),  # col_1 = driftcorr_x_coordinate
        df.iloc[:, 2].tolist(),  # col_2 = driftcorr_y_coordinate
    )

    df_2_concat = pd.DataFrame(
        {
            "timestamp": df.iloc[:, 0],
            "x_norm": df.iloc[:, 1],
            "y_norm": df.iloc[:, 2],
            "x_deg": x_deg,
            "y_deg": y_deg,
            "confidence": df.iloc[:, 4],
        }
    )

    df_2_concat.insert(loc=0, column="subject_id", value=f'sub-{eb["sub"]}', allow_duplicates=True)
    df_2_concat.insert(loc=1, column="session_id", value=f'ses-{eb["ses"]}', allow_duplicates=True)
    
    run_val = f'task-{eb["task"]}' if 'run' not in eb else f'task-{eb["task"]}_run-{eb["run"]}'
    df_2_concat.insert(loc=2, column="run_id", value=run_val, allow_duplicates=True)

    if trial_num is not None:
        df_2_concat.insert(loc=3, column="trial_id", value=trial_num, allow_duplicates=True)

    return df_2_concat


def compile_gaze_df(
    gaze_dir: str,
    sub_num: str,
    sampling: int,
    conf_thresh: float,
    bids_dir=None,
) -> pd.DataFrame:
    """
    Concatenate gaze data across a subject's runs into one dataframe
    """
    et_file_list = sorted(glob.glob(
        f'{gaze_dir}/sub-{sub_num}/ses-*/func/sub-{sub_num}*'
        '_recording-eye0_desc-driftcorr_physio.tsv.gz',
    ))

    if bids_dir is None:
        """
        Without events files (no trials). 
        """
        gaze_df = pd.DataFrame(columns=[
            'subject_id','session_id', 'run_id', 'timestamp', 
            'x_norm', 'y_norm', 'x_deg', 'y_deg', 'confidence',
        ])

        """
        Extract gaze from each run (no trials).
        Include all run's gaze.
        """
        for et_path in et_file_list:

            df_et = pd.read_csv(et_path, sep= '\t')
            eb = parse_file_name(os.path.basename(et_path))

            df_2_concat = format_gaze_data(
                df_et, eb, conf_thresh, trial_num=None,
            )

            gaze_df = pd.concat((gaze_df, df_2_concat), ignore_index=True)

    else:
        """
        With events files (has trials)
        """
        gaze_df = pd.DataFrame(columns=[
            'subject_id','session_id', 'run_id', 'trial_id', 
            'timestamp', 'x_norm', 'y_norm', 'x_deg', 'y_deg', 
            'confidence',
        ])
        """
        Extract gaze from each run. 
        Include trial gaze, exclude ISI gaze
        """
        for et_path in et_file_list:

            df_et = pd.read_csv(et_path, sep= '\t')
            eb = parse_file_name(os.path.basename(et_path))
        
            ev_path = get_event_path(et_path, bids_dir)
            if ev_path is not None:
                df_ev = pd.read_csv(ev_path, sep= '\t')

                trial_count = 0
                for i in range(df_ev.shape[0]):
                    if 'trial_type' in df_ev.columns() and 'trial_type' in ['fix', 'fixation_dot']:
                        continue

                    trial_onset, trial_offset = get_trial_times(df_ev, ev_path, i)

                    # TODO: skip trials with no behav response (button press)?

                    # filter trial's gaze and downsample (one every "sampling" gaze points)
                    df_trial = df_et[np.logical_and(
                        df_et.iloc[:, 0].to_numpy() > trial_onset,  # col_0 = timestamp
                        df_et.iloc[:, 0].to_numpy() < trial_offset  # col_0 = timestamp
                    )][::sampling]  # sample 1 every x; if sampling==5, sample 1 every 5 gaze

                    trial_count += 1
                    trial_num = df_ev['TrialNumber'][i] if 'TrialNumber' in df_ev.columns() else trial_count

                    df_2_concat = format_gaze_data(
                        df_trial, eb, conf_thresh, trial_num=trial_num,
                    )

                    gaze_df = pd.concat((gaze_df, df_2_concat), ignore_index=True)

    return gaze_df


@click.command()
@click.argument(
    "--gaze_dir",
    type=click.Path(),
    help='path to dset repo with drift corrected gaze'    
)
@click.argument(
    '--subject', 
    help='The participant number. E.g., 01.',
)
@click.option(
    "--bids_dir",
    type=click.Path(),
    help='Path to BIDS dset repo with events.tsv files. '
    'E.g., /unf/eyetracker/neuromod/emotionsvideos/sourcedata. '
    'Specify only for tasks with trials to exclude gaze captured during ISI.'    
)
@click.option(
    "--per_run",
    is_flag=True,
    help='If True, plot gaze density per run',
)
@click.option(
    '--run',
    help='If a run identifier is specified, plot gaze density '
    'for just that run.',
)
@click.option(
    '--trial',
    help='If a trial number is specified, plot gaze density '
    'for just that trial. The bids_dir argument must be specified.'
)
@click.option(
    '--sampling',
    type=click.IntRange(1, 250),
    default=5,
    help='Gaze sampling rate. E.g., at default sampling=5, the gaze is sampled every 5 frames.'
    'sampling=1 means no downsampling. Gaze is acquired at 250 fps.',
)
@click.option(
    '--conf_thresh',
    type=click.FloatRange(0, 1),
    default=0.9,
    help="Pupil detection confidence threshold for gaze. "
    "Value between 0.0 and 1.0, inclusive."
)
@click.option(
    "--use_cache",
    is_flag=True,
    help='If True, use cached tsv to generate the plot (if exists), '
    'else overwrite any cached file.',
)
def main(
    gaze_dir,
    subject,
    bids_dir,
    per_run,
    run,
    trial,
    sampling,
    conf_thresh,
    use_cache,
):
    """Gaze density plotting.

    Lists, formats and applies drift correction to raw eyetracking data acquired during fMRI, 
    extracts fixation metrics, and generates quality reports.
    \b

    Parameters

    ----------

    gaze_dir : str or pathlib.Path

        Absolute path to the derivative repository with drift-corrected gaze data, 
        events data (e.g., fixation metrics per trial), etc. 
        Figures and cached files are saved in this repository.
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos.eyetrackprep

    subject : str

        The CNeuroMod participant identifier. E.g., 01 for sub-01.

    bids_dir : str or pathlib.Path

        Absolute path to the cloned BIDS repository with events.tsv files per run.
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/bids_repos/emotion-videos

    per_run : bool, optional

        If specified, gaze density is plotted per run rather than across runs for the specified subject

    run : str, optional

        If a run identifier is passed as an argument, gaze density is ploted only for that run.

    trial : int, optional

        If a trial number is passed as an argument, gaze density is plotted only for that trial.

    sampling: int, optional

        Determines the gaze sampling rate to produce plots. At the default sampling=5, 
        the gaze is sampled every 5 frames. At sampling=1, there is no downsampling. 
        The gaze was acquired at a frequency of 250 fps.',

    conf_thresh: float, optional

        Determines the pupil detection confidence threshold to exclude gaze estimated from pupils 
        captured with low confidence.

    use_cache: bool, optional

        If True, generate plots from a cached .tsv file with gaze concatenated across runs,
        as long as the file exists. If False (default), overwrite any cached file.
    """


    """
    Step 1. compile a dataframe of concatenated gaze points. 
    Compile a single file with data from every run for 
    the specified subject.
    """
    gaze_df_path = f'{gaze_dir}/sub-{subject}/cache/sub-{subject}_recording-eye0_physio.tsv.gz'  # TODO: add task name?
    
    if Path(gaze_df_path).exists() and use_cache:
        gaze_df = pd.read_csv(gaze_df_path, sep= '\t')

    else: 
        gaze_df = compile_gaze_df(
            gaze_dir,
            sub_num,
            sampling,
            conf_thresh,
            bids_dir,
        )
        Path(os.path.dirname(gaze_df_path)).mkdir(parents=True, exist_ok=True)
        gaze_df.to_csv(
            gaze_df_path, sep='\t', header=True, index=False, compression='gzip',
        )

    """
    Step 2. Generate gaze density figure(s) 
    """
    # TODO



if __name__ == "__main__":
    main()
