from pathlib import Path

import click

from eyetrackprep.src import pupil2bids, utils 


@click.command()
@click.argument(
    "raw_et_dir",
    type=click.Path(),
    help="Absolute path to dataset directory with raw output files, including "
    "pupils.pldata, gaze.pldata, eye0.mp4, and PsychoPy log file. "
    "e.g., (on elm): /unf/eyetracker/neuromod/triplets/sourcedata"
)
@click.argument(
    "out_dir",
    type=click.Path(),
    help="Absolute path to output directory."
)
@click.option(
    "--export_plots",
    is_flag=True,
    help="If specified, plots will be generated to QC the dataset runs.",
)
@click.option(
    "--drift_corr",
    is_flag=True,
    help="If specified, outputs a dataset of drift corrected gaze as derivatives.",
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

        Path to the directory where the raw eyetracking data lives.

    out_dir : str or pathlib.Path

        Path to the output directory where to export BIDS-like dataset, 
        QC figures and reports (option), and drift-corrected gaze derivatives (option).

    drift_corr : bool, optional

        If specified, export a dataset of drift-corrected gaze as derivatives.

    export_plots : bool, optional

        If specified, export plots to support QCing of dataset runs.

    """

    """
    Step 1.1:
    Compiles an overview of all available files (pupils.pldata, gaze.pldata
    and eye0.mp4 exported by pupil, PsychoPy log file).

    Exports a .tsv listing all files to support manual QCing
    """
    # TODO: does script need to return df_pupfiles Dataframe, or just paths?
    df_pupfiles, pupil_file_paths = pupil2bids.compile_rawfile_list(
        raw_et_dir, out_dir)

    """
    Step 1.2: For each run:
    - export gaze files from pupil .pldata format to BIDS-like .tsv.gz format
    ??- plot the raw gaze & pupil data and export chart for QCing
    # TODO: merge charting into single process w drift-correction script?
    """
    for pupil_path in pupil_paths:
        pupil2bids.export_bids(
            pupil_path, raw_et_dir, out_dir)



if __name__ == "__main__":
    main()