import click
from pathlib import Path
from bids import BIDSLayout

from src import pupil2bids, driftcorr, qc_plots, utils


@click.command()
@click.argument(
    "raw_et_dir",
    type=click.Path(),
)
@click.argument(
    "out_dir",
    type=click.Path(),
)
@click.option(
    "--deriv_dir",
    type=click.Path(),
)
@click.option(
    "--export_plots",
    is_flag=True,
)
@click.option(
    "--correct_drift",
    is_flag=True,
)
def main(
    raw_et_dir,
    out_dir,
    deriv_dir=None,
    export_plots=False,
    correct_drift=False,
):
    """Eyetrackprep workflow.

    Lists, formats and applies drift correction to raw eyetracking data acquired during fMRI, 
    extracts fixation metrics, and generates quality reports.
    \b

    Parameters

    ----------

    raw_et_dir : str or pathlib.Path

        Absolute path to dataset directory with the raw eyetracking output files, 
        including pupils.pldata, gaze.pldata, eye0.mp4, and PsychoPy log file.
        e.g., (on elm): /unf/eyetracker/neuromod/triplets/sourcedata"

    out_dir : str or pathlib.Path

        Absolute path to the cloned BIDS repository where to save BIDS-like eyetracking data, 
        and QC figures and reports (optional).
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/bids_repos/emotion-videos

    deriv_dir : str or pathlib.Path, optional

        Absolute path to the derivative repository where to export drift-corrected
        gaze data, events data (e.g., fixation metrics per trial), etc.
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos

    export_plots : bool, optional

        If specified, exports plots to support QCing of the dataset's runs.

    correct_drift : bool, optional

        If specified, exports a dataset of drift-corrected gaze as derivatives.

    """

    """
    Compiles an overview of all available files (pupils.pldata, gaze.pldata
    and eye0.mp4 exported by pupil, PsychoPy log file).

    Exports a .tsv listing all files to support manual QCing
    Returns a list of directories that contain raw eye-tracking data to process
    """
    out_dir_layout = BIDSLayout(out_dir)

    pupil_file_paths, task_root = pupil2bids.compile_rawfile_list(
        raw_et_dir, out_dir_layout)

    """
    Initializes process logs
    """
    utils.init_logs(
        task_root, correct_drift, export_plots, out_dir, deriv_dir
    )

    """
    Processes, exports and returns pupil and gaze metrics in BIDS format.
    
    If correct_drift == True, also exports drift corrected gaze as derivatives    
    """
    for pupil_path in pupil_file_paths:
        
        """ exports raw pupils to bids """
        bids_gaze = pupil2bids.export_bids(
            pupil_path, raw_et_dir, out_dir)

        """ corrects gaze for drift and exports as derivatives """
        if correct_drift:
            dc_gaze, filter_gaze, fix_data = driftcorr.driftcorr_run(
                bids_gaze, task_root, 
                pupil_path, deriv_dir,
            )
            if export_plots:
                qc_plots.plot_dc_gaze(
                    dc_gaze, pupil_path, task_root, 
                    deriv_dir, filter_gaze, fix_data,
                )

        elif export_plots:
            qc_plots.plot_raw_gaze(
                bids_gaze, pupil_path, task_root, out_dir,                
            )

    """
    TODO: implement gaze drift correction for tasks w/o known fixations
    """

if __name__ == "__main__":
    main()
