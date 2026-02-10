import os, glob, sys
from typing import Union

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from src.utils import log_qc, get_event_path


def plot_raw_gaze(
    gaze_data: np.ndarray,
    run_metadata: tuple,
    task_root: str,
    plot_dir: str,
    failed_dc: bool=False,
) -> None:
    """
    Plots raw gaze only (no drift corrected gaze)
    Either drift correction was not performed, or it failed for a given run.

    Exports figure with uncorrected gaze position (x-axis) over time (y-axis). 
    Plots gaze position in x and y on separate pannels.
    Color map reflects the pupil detection confidence score    
    """
    qc_path = f"{plot_dir}/code/QC_gaze/plot_report_{task_root}.txt"
    sub, ses, run, task, fnum = run_metadata[1]  # == pupil_path[1]
    # early mario3 runs accidentally labelled task-mariostars...
    task = 'task-mario3' if task_root == 'mario3' else task.replace("-fixations", "").replace("-friends", "")
    
    Path(f'{plot_dir}/{sub}/figures').mkdir(parents=True, exist_ok=True)
    fdc = "_desc-driftcorrfail" if failed_dc else ""
    if run is None:
        fig_path = f'{plot_dir}/{sub}/figures/{sub}_{ses}_{fnum}_{task}{fdc}_qcplot.png'
        run = ""
    else:
        fig_path = f'{plot_dir}/{sub}/figures/{sub}_{ses}_{fnum}_{task}_{run}{fdc}_qcplot.png'

    if not Path(fig_path).exists():
        if gaze_data.shape[0] == 0:
            log_qc(f"Plotting fail: no gaze data for {sub} {ses} {run} {task} {fnum}", qc_path)

        elif gaze_data.shape[1] != 12:
            log_qc(f"Plotting fail: unexpected number of columns in gaze data file for {sub} {ses} {run} {task} {fnum}", qc_path)

        else:
            fig, axes = plt.subplots(1, 2, figsize=(21, 5))
            plot_labels = ['gaze_x', 'gaze_y']
            run_dur = gaze_data[-1][0] + 15  # 15s after offset to space out x-axis

            for i in range(2):
                axes[i].scatter(
                    gaze_data[:, 0],      # timestamp (col 0)
                    gaze_data[:, i+1],    # x (col 1) or y (col 2)
                    c=gaze_data[:, 3],    # confidence (col 3)
                    s=10,
                    cmap='terrain_r',
                    alpha=0.2,
                )
                axes[i].set_ylim(-2, 2)
                axes[i].set_xlim(0, run_dur)  # x-axis limit is last gaze's time stamp + buffer
                axes[i].set_title(f'{sub} {task} {ses} {run} {plot_labels[i]}')

            fig.savefig(fig_path)
            plt.close()


def plot_dc_gaze(
    gaze_data: np.ndarray,
    run_metadata: tuple,
    task_root: str,
    plot_dir: str,
    clean_data: Union[np.ndarray, None],
    fix_data: Union[np.ndarray, None],
) -> None:
    """
    Plots raw and drift corrected gaze
    """
    qc_path = f"{plot_dir}/code/QC_gaze/plot_report_{task_root}.txt"
    sub, ses, run, task, fnum = run_metadata[1]  # == pupil_path[1]
    # early mario3 runs accidentally labelled task-mariostars...
    task = 'task-mario3' if task_root == 'mario3' else task.replace("-fixations", "").replace("-friends", "")

    Path(f'{plot_dir}/{sub}/{ses}/figures').mkdir(parents=True, exist_ok=True)
    if run is None:
        fig_path = f'{plot_dir}/{sub}/figures/{sub}_{ses}_{fnum}_{task}_desc-driftcorr_qcplot.png'
        run = ""
    else:
        fig_path = f'{plot_dir}/{sub}/figures/{sub}_{ses}_{fnum}_{task}_{run}_desc-driftcorr_qcplot.png'

    if not Path(fig_path).exists():
        if gaze_data.shape[0] == 0:
            log_qc(f"Plotting fail: no gaze data for {sub} {ses} {run} {task} {fnum}", qc_path)

        elif gaze_data.shape[1] == 12:
            """
            Expects 12 columns in gaze_data array for bids eyetracking export
            """
            log_qc(f"Plotting raw data only: drift correction failed for {sub} {ses} {run} {task} {fnum}", qc_path)

            plot_raw_gaze(gaze_data, run_metadata, task_root, plot_dir, failed_dc=True)

        elif gaze_data.shape[1] != 15:
            """
            Expects 15 columns in gaze_data array for derivative eyetracking export            
            """
            log_qc(f"Plotting fail: unexpected number of columns in gaze data file for {sub} {ses} {run} {task} {fnum}", qc_path)

        else:
            fig, axes = plt.subplots(3, 2, figsize=(21, 14))
            run_dur = gaze_data[-1][0] + 15  # 15s after offset to space out x-axis
            plot_labels = [
                ['raw gaze_x', 'raw gaze_y'], 
                ['re-aligned gaze_x', 're-aligned gaze_y'],
                ['fix distance x', 'fix distance y'],
            ]

            for i in range(2):
                """ plot raw gaze """            
                axes[0][i].scatter(
                    gaze_data[:, 0],      # timestamp (col 0)
                    gaze_data[:, i+5],    # raw x (col 5) or raw y (col 6)
                    c=gaze_data[:, 4],    # confidence (col 4)
                    s=10,
                    cmap='terrain_r',
                    alpha=0.2,
                )
                axes[0][i].set_ylim(-2, 2)
                axes[0][i].set_xlim(0, run_dur)  # x-axis limit is last gaze's time stamp + buffer
                axes[0][i].set_title(
                    f'{sub} {task} {ses} {run}\n{plot_labels[0][i]}'
                )

                """ plot raw vs corrected gaze """            
                axes[1][i].scatter(
                    gaze_data[:, 0],          # timestamp (col 0)
                    gaze_data[:, i+5],        # raw x (col 5) or raw y (col 6)
                    s=10, 
                    color='xkcd:light grey', 
                    alpha=gaze_data[:, 4],    # confidence (col 4)
                )
                axes[1][i].scatter(
                    gaze_data[:, 0],          # timestamp (col 0) 
                    gaze_data[:, i+1],        # re-aligned x (col 1) or re-aligned y (col 2) 
                    c=gaze_data[:, 4],        # confidence (col 4)
                    s=10, 
                    cmap='terrain_r', 
                    alpha=0.2,
                )
                axes[1][i].set_ylim(-2, 2)
                axes[1][i].set_xlim(0, run_dur)
                axes[1][i].set_title(f'{plot_labels[1][i]}')

                """ plot distance to central fixation """
                if clean_data is not None and fix_data is not None:
                    axes[2][i].scatter(
                        clean_data[:, 0],           # timestamp (col 0) 
                        clean_data[:, i+1],         # dist to center x (col 1) or dist to center y (col 2)  
                        color='xkcd:light blue', 
                        s=20, alpha=0.2,
                    )
                    axes[2][i].scatter(
                        fix_data[:, 0],             # timestamp (col 0)  
                        fix_data[:, i+1],           # dist to center x (col 1) or dist to center y (col 2)  
                        #color='xkcd:orange',       
                        #c=fix_data[:, 4],           # c = num above-threshold high-conf pupils during fix
                        c=np.sqrt(fix_data[:, 5]**2 + fix_data[:, 6]**2),   # c = stdev in x and y converted to avg distance to fix middle
                        cmap='magma',              # color-coded fixations based on confidence (yellow=low, red = high)
                        s=20, alpha=1.0,
                    )
                    if i == 0:
                        axes[2][i].set_ylim(-2, 2)
                    else:
                        lb = np.min(fix_data[:, i+1])-0.1 if np.min(fix_data[:, i+1]) < -2 else -2
                        hb = np.max(fix_data[:, i+1])+0.1 if np.max(fix_data[:, i+1]) > 2 else 2
                        axes[2][i].set_ylim(lb, hb)
                    axes[2][i].set_xlim(0, run_dur)
                    axes[2][i].set_title(f'{plot_labels[2][i]}')

            fig.savefig(fig_path)
            plt.close()