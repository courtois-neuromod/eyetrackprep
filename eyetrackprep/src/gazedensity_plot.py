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

THINGS: analyze fixations
https://github.com/courtois-neuromod/things.behaviour/blob/b84bf6d5c18e53e78c6278bfcb5d4e0e6afff214/code/analyze_fixations.py

notebook: gaze density plot
https://github.com/courtois-neuromod/cneuromod-things/blob/main/datapaper/notebooks/fixation_compliance.ipynb

numpy version:  1.24.4
pandas version:  1.3.5
matplotlib version:  3.7.5
seaborn version:  0.11.2


"""


def get_event_path(
    et_path: str,
    raw_dir: str,
    eb: dict,
) -> str:
    """
    Find eyetracking file's corresponding events file.
    """
    # TODO: add robustness to ses and run number padding... possibly task name variations too...
    if 'run' in eb:
        ev_paths = glob.glob(
            f'{raw_dir}/sub-{eb["sub"]}/ses-{eb["ses"]}/sub-{eb["sub"]}_ses-{eb["ses"]}_'
            f'{eb["fnum"]}_task-{eb["task"]}_run-{eb["run"]}_events.tsv',
        )
    else:
        ev_paths = glob.glob(
            f'{raw_dir}/sub-{eb["sub"]}/ses-{eb["ses"]}/sub-{eb["sub"]}_ses-{eb["ses"]}_'
            f'{eb["fnum"]}_task-{eb["task"]}_events.tsv',
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
            x for x in df_ev.columns if '_offset' in x
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
    #x_deg, y_deg, dist_deg = get_degrees(
    #    df.iloc[:, 1].tolist(),  # col_1 = driftcorr_x_coordinate
    #    df.iloc[:, 2].tolist(),  # col_2 = driftcorr_y_coordinate
    #)

    df_2_concat = pd.DataFrame(
        {
            "timestamp": df.iloc[:, 0],
            "x_norm": df.iloc[:, 1],
            "y_norm": df.iloc[:, 2],
            #"x_deg": x_deg,
            #"y_deg": y_deg,
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
    raw_dir=None,
) -> pd.DataFrame:
    """
    Concatenate gaze data across a subject's runs into one dataframe
    """
    et_file_list = sorted(glob.glob(
        f'{gaze_dir}/sub-{sub_num}/ses-*/func/sub-{sub_num}*'
        '_recording-eye0_desc-driftcorr_physio.tsv.gz',
    ))

    if raw_dir is None:
        """
        Without events files (no trials). 
        """
        gaze_df = pd.DataFrame(columns=[
            'subject_id','session_id', 'run_id', 'timestamp', 
            #'x_norm', 'y_norm', 'x_deg', 'y_deg', 'confidence',
            'x_norm', 'y_norm', 'confidence',
        ])

        """
        Extract gaze from each run (no trials).
        Include all run's gaze.
        """
        for et_path in et_file_list:

            df_et = pd.read_csv(et_path, sep= '\t')[::sampling]
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
            #'timestamp', 'x_norm', 'y_norm', 'x_deg', 'y_deg', 
            #'confidence',
            'timestamp', 'x_norm', 'y_norm', 'confidence',
        ])
        """
        Extract gaze from each run. 
        Include trial gaze, exclude ISI gaze
        """
        for et_path in et_file_list:

            df_et = pd.read_csv(et_path, sep= '\t', header=None)
            eb = parse_file_name(os.path.basename(et_path))
        
            ev_path = get_event_path(et_path, raw_dir, eb)
            if ev_path is not None:
                df_ev = pd.read_csv(ev_path, sep= '\t')

                trial_count = 0
                for i in range(df_ev.shape[0]):
                    if 'trial_type' in df_ev.columns and df_ev['trial_type'][i] in ['fix', 'fixation_dot']:
                        continue

                    trial_onset, trial_offset = get_trial_times(df_ev, ev_path, i)

                    # TODO: skip trials with no behav response (button press)?
                    """
                    Filter trial's gaze and downsample.
                    Sample gaze 1 every x; e.g., if sampling==5, sample 1 every 5 gaze.
                    """
                    df_trial = df_et[np.logical_and(
                        df_et.iloc[:, 0].to_numpy() > trial_onset,  # col_0 = timestamp
                        df_et.iloc[:, 0].to_numpy() < trial_offset  # col_0 = timestamp
                    )][::sampling]

                    trial_count += 1
                    trial_num = df_ev['TrialNumber'][i] if 'TrialNumber' in df_ev.columns else trial_count

                    df_2_concat = format_gaze_data(
                        df_trial, eb, conf_thresh, trial_num=trial_num,
                    )

                    gaze_df = pd.concat((gaze_df, df_2_concat), ignore_index=True)

    return gaze_df


def gaussian(x, sx, y=None, sy=None):
    """
    Gaussian function from PyGazeAnalyser's gazeplotter.py script
    source:
    https://github.com/esdalmaijer/PyGazeAnalyser/blob/de44913fb60876134c5b942690d586e5dab40476/pygazeanalyser/gazeplotter.py#L423
    
    Returns an array of numpy arrays (a matrix) containing values between
    1 and 0 in a 2D Gaussian distribution
    
    arguments
    x : width in pixels
    sx: width standard deviation
    
    keyword arguments
    y: height in pixels (default = x)
    sx: heigh standard deviation (default = sx)
    """
    
    # square Gaussian if only x values are passed
    if y == None:
        y = x  
    if sy == None:
        sy = sx
        
    # centers
    xo = x/2
    yo = y/2
    # matrix of zeros
    M = np.zeros([y,x],dtype=float)
    # gaussian matrix
    for i in range(x):
        for j in range(y):
            
            M[j,i] = np.exp(-1.0 * (((float(i)-xo)**2/(2*sx*sx)) + ((float(j)-yo)**2/(2*sy*sy)) ) )
    
    return M


def plot_gaze(
    df_gaze: pd.DataFrame,
    fig_path: str,
    contour: bool,
) -> None:
    """
    Generate gaze density plot(s)
    """

    """
    Filter gaze outside the screen (1280, 1024)
    """
    df_fig = df_gaze[np.logical_and(
        df_gaze["x_norm"].to_numpy() > 0,
        df_gaze["x_norm"].to_numpy() < 1
    )]
    df_fig = df_fig[np.logical_and(
        df_fig["y_norm"].to_numpy() > 0,
        df_fig["y_norm"].to_numpy() < 1
    )]

    """
    Convert gaze coordinates to pixels
    """
    x_pix = (1280*df_fig["x_norm"].to_numpy()).astype(int)
    y_pix = (1024*df_fig["y_norm"].to_numpy()).astype(int)
    screen_dim = (1280, 1024)  # (int(1280*(10.0/17.5)), int(1024*(10.0/14.0)))

    """
    Create gaze density heatmap
    """
    gwh = 50
    gsdwh = gwh/6
    gaus = gaussian(gwh,gsdwh)
    strt = int(gwh/2)

    heatmapsize = int(screen_dim[1] + 2*strt), int(screen_dim[0] + 2*strt)
    heatmap = np.zeros(heatmapsize, dtype=float)

    """
    Add Gaussian to the current heatmap
    """
    for i in range(len(x_pix)):
        x = int(x_pix[i])
        y = int(y_pix[i])
        heatmap[y:y+gwh,x:x+gwh] += gaus

    """
    Resize heatmap (trim added padding)
    """
    heatmap = heatmap[strt:screen_dim[1]+strt,strt:screen_dim[0]+strt]
    
    """
    Normalize heatmap values to between 0 and 1
    """
    min_val = np.min(heatmap)
    heatmap -= min_val
    max_val = np.max(heatmap)
    heatmap = heatmap/max_val    
    
    """
    Plot heatmap
    origin='lower' -> array index 0, 0 plotted in bottom left corner 
    """
    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap='turbo', alpha=1.0, origin='lower')

    """
    Add contours to heatmap
    """
    if add_contour:
        # make dataframe of gaze in pixels for seaborn
        gazepix_df = pd.DataFrame({
            "x": x_pix,
            "y": y_pix,
        })
        sns.kdeplot(
            data=gazepix_df,
            x="x",
            y="y",
            color="xkcd:white",
            bw_adjust=2,
            levels=[0.25, 0.5, 0.75]
        )

    """
    Make ticks. screen dim (1280, 1024) = (17.5, 14.0) deg of visual angle.
    146 pixels ~ 2 deg visual angles [1280 * (2/17.5), or 1024 * (2/14.0)].
    Make tick mark every 2 deg visual of angle from screen center. 
    """
    #plt.xticks(np.arange(56, 1279, 146), ["", "", "", "", "", "", "", "", ""])
    #plt.yticks(np.arange(74, 1023, 146), ["", "", "", "", "", "", ""])
    plt.xticks(np.arange(56, 1279, 146), [-8, -6, -4, -2, 0, 2, 4, 6, 8])  # deg visual angle from center
    plt.yticks(np.arange(74, 1023, 146), [-6, -4, -2, 0, 2, 4, 6])

    plt.xlim([0, 1280])
    plt.ylim([0, 1024])

    ax = plt.gca()
    ax.tick_params(width=3, size=8)

    plt.savefig(f'{fig_path}gazedensity.jpg', dpi=300) 

    plt.close()


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
    "--raw_dir",
    type=click.Path(),
    help='Path to raw dset repo with events.tsv files. '
    'E.g., /unf/eyetracker/neuromod/emotionsvideos/sourcedata. '
    'Specify only for tasks with trials to exclude gaze captured during ISI.'    
)
@click.option(
    "--per_run",
    is_flag=True,
    help='If True, plot gaze density per run. This flag is ignored if session '
    '(with/without run or trial) is specified.',
)
@click.option(
    '--session',
    help='If a session identifier is specified, plot gaze density '
    'for just that session (unless run and/or trial are also specified).',
)
@click.option(
    '--run',
    help='If a run identifier is specified, plot gaze density '
    'for just that run (unless trial is also specified). The session '
    'argument needs to be specified.',
)
@click.option(
    '--trial',
    help='If a trial number is specified, plot gaze density '
    'for just that trial. The raw_dir, session and run arguments '
    'must be specified.',
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
    "--contour",
    is_flag=True,
    help='If True, adds contour lines that represent 25, 50 and 75 percent'
    ' of the gaze density to the gaze density plots.',
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
    raw_dir,
    per_run,
    session,
    run,
    trial,
    sampling,
    conf_thresh,
    contour,
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

    raw_dir : str or pathlib.Path

        Absolute path to the raw files repository with events.tsv files per run.
        e.g., on elm: /unf/eyetracker/neuromod/emotionsvideos/sourcedata

    per_run : bool, optional

        If specified, gaze density is plotted per run rather than across runs for the specified subject. 
        This flag is ignored if session (with/without run or trial) is specified.

    session : str, optional

        If a session identifier is passed as an argument, gaze density is ploted only for that session, 
        unless run and/or trial are also specified, in which case gaze is plotted per run or trial (whichever
        is most specific).

    run : str, optional

        If a run identifier is passed as an argument, gaze density is ploted only for that run, unless trial
        is also specified (in which case gaze is only plotted for that run's trial). Note that the session 
        number also needs to be specified.

    trial : int, optional

        If a trial number is passed as an argument, gaze density is plotted only for that trial. The raw_dir, 
        session and run numbers also need to be specified.

    sampling: int, optional

        Determines the gaze sampling rate to produce plots. At the default sampling=5, 
        the gaze is sampled every 5 frames. At sampling=1, there is no downsampling. 
        The gaze was acquired at a frequency of 250 fps.',

    conf_thresh: float, optional

        Determines the pupil detection confidence threshold to exclude gaze estimated from pupils 
        captured with low confidence.

    contour: bool, optional

        If True, adds contour lines that represent 25, 50 and 75 percent of the gaze density to 
        the gaze density plots.

    use_cache: bool, optional

        If True, generate plots from a cached .tsv file with gaze concatenated across runs,
        as long as the file exists. If False (default), overwrite any cached file.
    """


    """
    Step 1. compile a dataframe of concatenated gaze points. 
    Compile a single file with data from every run for 
    the specified subject.
    """
    gaze_df_path = f'{gaze_dir}/cache/sub-{subject}_recording-eye0_physio.tsv.gz'  # TODO: add task name?
    
    if Path(gaze_df_path).exists() and use_cache:
        gaze_df = pd.read_csv(gaze_df_path, sep= '\t')

    else: 
        gaze_df = compile_gaze_df(
            gaze_dir,
            subject,
            sampling,
            conf_thresh,
            raw_dir,
        )
        Path(os.path.dirname(gaze_df_path)).mkdir(parents=True, exist_ok=True)
        gaze_df.to_csv(
            gaze_df_path, sep='\t', header=True, index=False, compression='gzip',
        )

    """
    Step 2. Generate gaze density figure(s) 
    """
    plot_path = f'{gaze_dir}/sub-{subject}/figures/sub-{subject}_task-{os.path.basename(gaze_dir).split(".")[0]}_'
    Path(os.path.dirname(plot_path)).mkdir(parents=True, exist_ok=True)
    if session is not None:
        if f'ses-{session}' not in gaze_df['session_id']:
            print(f'No session {session} was found for sub-{subject}')
            plot_path = None
        else:
            gaze_df = gaze_df[gaze_df['session_id'] == f'ses-{session}']
            plot_path += f'ses-{session}_'
            if run is not None:
                if run not in gaze_df['run_id']:
                    print(f'No run {run.split('_')[-1].split('-')[-1]} was found in session {session} for sub-{subject}')
                    plot_path = None
                else:
                    gaze_df = gaze_df[gaze_df['run_id'] == run]
                    plot_path += f'{run.split('_')[-1]}_'
                    if trial is not None:
                        if raw_dir is None:
                            print("This task has no distinct trials")
                            plot_path = None
                        elif trial not in gaze_df['trial_id']:
                            print(f'No trial {trial} was found in run {run.split('_')[-1].split('-')[-1]}, session {session} for sub-{subject}')
                            plot_path = None
                        else:
                            gaze_df = gaze_df[gaze_df['trial_id'] == trial]
                            plot_path += f'trial-{trial}_'
            elif trial is not None:
                print("Please make sure to specify a run identifier")
                plot_path = None
        
        if plot_path is not None:
            plot_gaze(gaze_df, plot_path, contour)

    elif run is not None or trial is not None: 
        print("Please make sure to specify a session number")
    elif per_run:
        for session in np.unique(gaze_df['session_id']):
            ses_df = gaze_df[gaze_df['session_id' == session]]
            for run in np.unique(ses_df['run_id']):
                run_df = ses_df[ses_df['session_id'] == run]
                plot_gaze(run_df, f'{plot_path}{session}_{run}_', contour)
    else:
        plot_gaze(gaze_df, plot_path, contour)


if __name__ == "__main__":
    main()
