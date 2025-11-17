from pathlib import Path
import click
from src import pupil2bids
from bids import BIDSLayout


@click.command()
@click.argument(
    "raw_et_dir",
    type=click.Path(),
)
@click.argument(
    "out_dir",
    type=click.Path(),
)
@click.argument(
    "deriv_dir",
    type=click.Path(),
)
@click.option(
    "--export_plots",
    is_flag=True,
)
@click.option(
    "--drift_corr",
    is_flag=True,
)
def main(
    raw_et_dir,
    out_dir,
    export_plots=False,
    drift_corr=False,
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

    deriv_dir : str or pathlib.Path

        Absolute path to the derivative repository where to export drift-corrected
        gaze data, events data (e.g., fixation metrics per trial), etc.
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/deriv_repos/emotion-videos

    export_plots : bool, optional

        If specified, exports plots to support QCing of the dataset's runs.

    drift_corr : bool, optional

        If specified, exports a dataset of drift-corrected gaze as derivatives.

    """

    """
    Compiles an overview of all available files (pupils.pldata, gaze.pldata
    and eye0.mp4 exported by pupil, PsychoPy log file).

    Exports a .tsv listing all files to support manual QCing
    Returns a list of directories with eye-tracking data to process
    """
    out_dir_layout = BIDSLayout(out_dir)

    pupil_file_paths = pupil2bids.compile_rawfile_list(
        raw_et_dir, out_dir_layout)
    
    """
    Processes, exports and returns pupil and gaze metrics in BIDS format.
    If export_plots is True, also returns raw (uncorrected) gaze coordinates 
    to plot for manual QCing.
    """
    for pupil_path in pupil_file_paths:
        
        bids_gaze, raw_gaze_2plot = pupil2bids.export_bids(
            pupil_path, raw_et_dir, out_dir, export_plots)


    """
    TODO: implement gaze drift correction for tasks w known fixations
    TODO: import Pupil library
    TODO: generate multi-pannel plot for manual QCing
    """


if __name__ == "__main__":
    main()