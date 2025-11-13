import os, glob, sys
from typing import Union

import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt


def make_qc_plot(
    run_metadata: tuple,
    task_root: str,
    raw_gaze: np.array,
    dcorr_gaze: Union[np.array, None],
    out_dir: str,
) -> None:
    """."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    sub, ses, run, task, fnum = run_metadata  # == pupil_path[1]
    # early mario3 runs accidentally labelled task-mariostars...
    task = 'task-mario3' if task_root == 'mario3' else task
    if run is None:
        fig_path = f'{out_dir}/code/{sub}_{ses}_{fnum}_{task}_qcplot.png'
        run = ""
    else:
        fig_path = f'{out_dir}/code/{sub}_{ses}_{fnum}_{task}_{run}_qcplot.png'


    if dcorr_gaze is None:
        """
        Exports figure with uncorrected gaze position over time, 
        in x and y (separate pannels).
        Color map is determined by the pupil detection confidence score
        """
        fig, axes = plt.subplots(2, 1, figsize=(7, 7))
        plot_labels = ['gaze_x', 'gaze_y']

        for i in range(2):
            axes[i].scatter(
                raw_gaze[:, 2],
                raw_gaze[:, i],
                c=raw_gaze[:, 3],
                s=10,
                cmap='terrain_r',
                alpha=0.2,
            )
            axes[i].set_ylim(-2, 2)
            axes[i].set_xlim(0, raw_gaze[-1, 2])
            axes[i].set_title(
                f'{sub} {task} {ses} {run} {plot_labels[i]}'
            )
        fig.savefig(fig_path)
        plt.close()

    else:
        fig_path = fig_path.replace("qcplot.png", "desc-driftcorr_qcplot.png")