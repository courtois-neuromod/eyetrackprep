from pathlib import Path
import click
from src import pupil2bids


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

        Path to the directory where the raw eyetracking data live.
        Absolute path to dataset directory with the raw eyetracking output files, 
        including pupils.pldata, gaze.pldata, eye0.mp4, and PsychoPy log file.
        e.g., (on elm): /unf/eyetracker/neuromod/triplets/sourcedata"

    out_dir : str or pathlib.Path

        Absolute path to the output directory where to export BIDS-like dataset, 
        QC figures and reports (option), and drift-corrected gaze derivatives (option).
        e.g., on elm: /data/neuromod/projects/eyetracking_bids/emotionsvideos

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
    pupil_file_paths = pupil2bids.compile_rawfile_list(
        raw_et_dir, out_dir)

    """
    Processes, exports and returns pupil and gaze metrics in BIDS format.
    If export_plots is True, also returns raw (uncorrected) gaze coordinates 
    to plot for manual QCing.
    """
    for pupil_path in pupil_paths:
        bids_gaze, raw_gaze_2plot = pupil2bids.export_bids(
            pupil_path, raw_et_dir, out_dir, export_plots)

    """
    TODO: implement gaze drift correction for tasks w known fixations
    TODO: import Pupil library
    TODO: generate multi-pannel plot for manual QCing
    """


if __name__ == "__main__":
    main()