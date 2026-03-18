import os, glob
from pathlib import Path

import ffmpeg
import cv2
import numpy as np
import pandas as pd
#from moviepy.editor import *
from moviepy import VideoFileClip
from scipy.interpolate import interp1d

#from utils import parse_file_name

FULL_SCREEN_SIZE = (1280, 1024)  # pix
SCALING_EMOTION_VIDEOS = 900  # pix

"""
TODO: once merged to main, import parse_file_name from utils (on visu branch)
and delete local function below
"""
def parse_file_name(
    file_name: str,
)-> dict:
    """."""
    file_bits = file_name.split("_")
    
    fb_dict = {}
    for fb in file_bits:
        fb_bits = fb.split("-")
        if len(fb_bits) == 2:
            if fb_bits[0].isdigit():
                fb_dict["fnum"] = fb
            else:
                fb_dict[fb_bits[0]] = fb_bits[1]
    return fb_dict


def get_video_specs(
    video_path: str,
)-> dict:
    """
    use ffmpeg to extract video specs
    """
    probe = ffmpeg.probe(video_path)
    video_specs = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'
    ), None)
    
    width = int(video_specs['width'])
    height = int(video_specs['height'])
    
    # FPS is usually stored as a fraction like "60000/1001" or "30/1"
    fps_eval = video_specs['avg_frame_rate'].split('/')
    fps = float(fps_eval[0]) / float(fps_eval[1])
    
    # nb_frames is the most accurate frame count available in the header
    # Fallback to duration * fps if nb_frames is missing
    duration = float(video_specs.get('duration', probe['format']['duration']))
    nb_frames = int(video_specs.get('nb_frames', round(duration * fps)))
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
    }


def norm2pix(
    video_specs: dict,
    df_trial: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    TODO: add doc
    e.g., https://github.com/courtois-neuromod/ds_prep/blob/57e94626fd9ba59259e16850472956f544b91be9/eyetracking/FRIENDS_Gaze_on_Film.py#L51    
    """
    video_width = video_specs['width']
    video_height = video_specs['height']

    x_pix = df_trial['x_coord'].to_numpy() * FULL_SCREEN_SIZE[0]
    y_pix = (1.0 - df_trial['y_coord'].to_numpy()) * FULL_SCREEN_SIZE[1]   

    if video_width > video_height:
        """
        video full width is projected onto screen's central 900 pixels 
        (full width is 1280, with 190 pixels of padding on either side).
        Gaze position is clipped to remain on the movie frame
        """
        x_pix = np.floor((x_pix - 190) * (video_width/SCALING_EMOTION_VIDEOS)).astype(int)
        x_pix = np.clip(x_pix, 0, int(video_width-1))

        """
        height is scaled proortionally to width
        """
        projected_height = (video_height / video_width) * SCALING_EMOTION_VIDEOS
        edge_y = int((FULL_SCREEN_SIZE[1] - projected_height)/2)
        y_pix = np.floor((y_pix - edge_y) * (video_height/projected_height)).astype(int)
        y_pix = np.clip(y_pix, 0, int(video_height-1))

    else:
        """
        video full height is projected onto screen's central 900 pixels
        (full height is 1024, with 62 pixels of padding on either side).
        Gaze position is clipped to remain on the movie frame
        """
        y_pix = np.floor((y_pix - 62) * (video_height/SCALING_EMOTION_VIDEOS)).astype(int)
        y_pix = np.clip(y_pix, 0, int(video_height-1))

        """
        width is scaled proortionally to height
        """
        projected_width = (video_width / video_height) * SCALING_EMOTION_VIDEOS
        edge_x = int((FULL_SCREEN_SIZE[0] - projected_width)/2)
        x_pix = np.floor((x_pix - edge_x) * (video_width/projected_width)).astype(int)
        x_pix = np.clip(x_pix, 0, int(video_width-1))

    df_trial.insert(loc=1, column="x_pix", value=x_pix, allow_duplicates=True)
    df_trial.insert(loc=2, column="y_pix", value=y_pix, allow_duplicates=True)

    return df_trial


def drawgaze(clip,fx,fy,r_zone):
    """
    Adapted from: https://zulko.github.io/moviepy/examples/headblur.html
    Returns a filter that will add a moving circle that corresponds to the gaze mapped onto
    the frames. The position of the circle at time t is
    defined by (fx(t), fy(t)), and the radius of the circle
    by ``r_zone``.
    Requires OpenCV for the circling.
    Automatically deals with the case where part of the image goes
    offscreen.
    """    
    
    def gaze2frame(get_frame,t):
        im_orig = get_frame(t)
        im = np.copy(im_orig)

        #im.setflags(write=1)
        #x,y = int(fx(t)),int(fy(t))
        try:
            x,y = int(fx(t)),int(fy(t))        
        except ValueError:
            # If t is outside interpolation range, return original frame
            return im

        # OpenCV circle automatically handles coordinates outside image bounds.
        #cv2.circle(im, (x, y), r_zone, (255, 0, 0), -1, lineType=cv2.LINE_AA)

        h,w,d = im.shape
        x1,x2 = max(0,x-r_zone),min(x+r_zone,w)
        y1,y2 = max(0,y-r_zone),min(y+r_zone,h)
        region_size = y2-y1,x2-x1

        orig = im[y1:y2, x1:x2]
        circled = cv2.circle(orig, (r_zone, r_zone), r_zone, (155, 0, 155), -1,
                             lineType=cv2.CV_AA)

        im[y1:y2, x1:x2] = circled
        
        return im

    return clip.transform(gaze2frame)    


def project_gaze2emotionvideos(
    events_path: str,
    stimuli_path: str,
    gaze_path: str,
    out_path: str,
    conf_thresh: float = 0.9,
    trial_num = None,
) -> None:
    """
    adapt from:
    https://github.com/courtois-neuromod/ds_prep/blob/eyetrack_dev_local/eyetracking/FRIENDS_Gaze_on_Film.py
    
    video scaling: 
    https://github.com/courtois-neuromod/task_stimuli/blob/main/src/tasks/emotionvideos.py#L64

    widest side (length or height) is scaled to 900 pix, the other side 
    is stretched proportionally
    debug = True
    if debug:
        events_path = '/unf/eyetracker/neuromod/emotionsvideos/sourcedata/sub-01/ses-003/sub-01_ses-003_20230526-103636_task-emotionvideos_run-08_events.tsv'
        stimuli_path = '/data/neuromod/projects/eyetracking_bids/figs_repos/repeated_gifs'
        gaze_path = '/data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos.eyetrackprep'
        out_path = '/data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos.eyetrackprep'
        conf_thresh = 0.9
        trial_num = 3

    """

    """
    parse events file name: subject, session, run
    """
    eb = parse_file_name(os.path.basename(events_path))

    """
    for each (repeated) video, output name w placeholder val to replace
    """
    out_dir = f'{out_path}/sub-{eb["sub"]}/videos'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_name = f'{out_dir}/sub-{eb["sub"]}_ses-{eb["ses"]}_task-{eb["task"]}_run-{eb["run"]}_placehold.mp4'
    

    """
    load events file
    """
    df_ev = pd.read_csv(events_path, sep='\t')

    """
    load run's gaze; filter above confidence threshold
    """
    et_path_list = sorted(glob.glob(
        f'{gaze_path}/sub-{eb["sub"]}/ses-{eb["ses"]}/func/'
        f'sub-{eb["sub"]}_ses-{eb["ses"]}_task-{eb["task"]}_'
        f'run-{eb["run"]}_*recording-eye0_desc-driftcorr_physio.tsv.gz'
    ))
    if not len(et_path_list) == 1:
        print(f'{len(et_path_list)} eye-tracking file(s) found for sub-{eb["sub"]}, ses-{eb["ses"]}, run-{eb["run"]}')
    else:
        df_et = pd.read_csv(et_path_list[0], sep= '\t', header=None).iloc[:, :5]
        df_et.columns = ["timestamp", "x_coord", "y_coord", "fix_idx", "confidence"]
        df_et = df_et[df_et["confidence"].to_numpy() > conf_thresh]  # filter gaze with pupil detection confidence threshold

    """
    loop throught events file to export gaze on video frame for each trial 
    """
    for i in range(df_ev.shape[0]):
        """
        If trial_num is specified, project gaze on video for that trial only, 
        otherwise extract them all.
        """
        if trial_num is None or df_ev['TrialNumber'][i] == trial_num:
            
            """
            extract trial gaze data
            TODO: make code below into a separate function
            """
            trial_onset = df_ev['onset_video_flip'][i]
            trial_offset = trial_onset + df_ev['total_duration'][i]
            df_trial = df_et[np.logical_and(
                df_et['timestamp'].to_numpy() > trial_onset - 0.1, 
                df_et['timestamp'].to_numpy() < trial_offset + 0.1 
            )].reset_index(drop=True)
            if df_trial.shape[0] == 0:
                continue
            else:
                df_trial.insert(loc=0, column="trial_timestamp", value=df_trial['timestamp'].to_numpy() - trial_onset)
                """
                TODO: adress boundaries & interpolation...
                if df_trial["trial_timestamp"][0] > 0:
                    df_trial = pd.concat([
                        pd.DataFrame([{
                            "trial_timestamp": 0.0, "timestamp": 0.0, "x_coord": 0.5, "y_coord": 0.5, "fix_idx": 0, "confidence": 0.99,
                        }]), 
                        df_trial], ignore_index=True)
                if df_trial["trial_timestamp"].iloc[-1] < (trial_offset - trial_onset):
                    df_trial = pd.concat([
                        df_trial, 
                        pd.DataFrame([{
                            "trial_timestamp": trial_offset - trial_onset, "timestamp": trial_offset - trial_onset, 
                            "x_coord": 0.5, "y_coord": 0.5, "fix_idx": 0, "confidence": 0.99,
                        }])], 
                        ignore_index=True) 
                """

                """
                get video name, onset and offset (relative to run onset)
                """
                video_num = df_ev['Gif'][i]
                video_path = f'{stimuli_path}/{video_num}'
                """
                stimuli_path /data/neuromod/projects/eyetracking_bids/figs_repos/repeated_gifs
                use ffmpeg to get video dims, fps, frame count and duration
                """
                video_specs = get_video_specs(video_path)

                """
                convert trial's gaze position from normalized (screen) to pixels, based on video dim and projection to screen

                create interpolation function in x and y (per video, or for entire run? do it once?...)
                Interpolation fonctions
                https://docs.scipy.org/doc/scipy/reference/reference/generated/scipy.interpolate.interp1d.html#scipy.interpolate.interp1d
                'linear’, ‘nearest’, ‘nearest-up’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, or ‘next’.
                """
                df_trial = norm2pix(video_specs, df_trial)
                gaze_times = df_trial['trial_timestamp'].to_numpy()
                f_xcoord = interp1d(gaze_times, df_trial['x_pix'].to_numpy(), kind='linear')
                f_ycoord = interp1d(gaze_times, df_trial['y_pix'].to_numpy(), kind='linear')

                """
                load video with moviepy
                draw gaze points on each frame and write video file
                """
                clip = VideoFileClip(video_path)
                clip_gaze = drawgaze(clip, f_xcoord, f_ycoord, 6)

                trial_out_name = out_name.replace('placehold', f"trial-{str(df_ev['TrialNumber'][i]).zfill(2)}_mv-{video_num.split('.')[0]}")
                clip_gaze.write_videofile(trial_out_name)    


