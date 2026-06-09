import click
import glob
import json
from pathlib import Path

import pandas as pd 


def _get_vals(df_run):
    sub = df_run['subject']
    ses = df_run['session']
    run = df_run['run']
    new_run = df_run['run_rename']
    fnum = df_run['file_number']

    return sub, ses, run, new_run, fnum


def _update_metadata(jpath, qc_check):
    with open(jpath, 'r') as metadata_file:
        mdata_old = json.load(metadata_file)

    if "DriftCorrection_QualityCheck" not in mdata_old:
        mdata_new = {"DriftCorrection_QualityCheck": qc_check}

        m_data = {**mdata_new, **mdata_old}

        with open(jpath, 'w') as metadata_file:
            json.dump(m_data, metadata_file, indent=4)


def _rename_run_files(
    old_root,
    new_root,
    bids_dir,
    deriv_dir,
):
    bids_list = sorted(glob.glob(
        f"{str(bids_dir)}/{old_root}*")
    )
    deriv_list = sorted(glob.glob(
        f"{str(deriv_dir)}/{old_root}*")
    )
    for et_file in bids_list + deriv_list:
        Path(et_file).rename(
            et_file.replace(old_root, new_root)
        )


@click.command()
@click.argument(
    "bids_dir",
    type=click.Path(),
)
@click.argument(
    "deriv_dir",
    type=click.Path(),
)
@click.argument(
    "label_file",
    type=click.Path(),
)
def main(
    bids_dir,
    deriv_dir,
    label_file,
):
    """."""
    df = pd.read_csv(label_file, sep="\t")

    for i in range(df.shape[0]):
        # Get run metrics
        df_run = df.iloc[i]
        sub, ses, run, new_run, fnum = _get_vals(df_run)
        old_root = (
            f"{sub}/{ses}/func/{sub}_{ses}_task-emotionvideos"
            f"_{run}_{fnum}_recording-eye0"
        )

        # Add Drift Correction Quality Check to run metadata
        jpath = (
            f"{str(deriv_dir)}/{old_root}_desc-driftcorr_physio.json"
        )
        if Path(jpath).exists():
            qc_check = "Pass" if not df_run['exclude'] else "Fail"
            _update_metadata(jpath, qc_check)

        # Rename old files (rm identifiers, re-set run numbers within ses)
        new_root = (
            f"{sub}/{ses}/func/{sub}_{ses}_task-emotion"
            f"_{new_run}_recording-eye0"
        )
        _rename_run_files(
            old_root, new_root, bids_dir, deriv_dir,
        )

        # Rename old figures (rm identifiers, re-set run numbers within ses)
        old_fig_root = (
            f"{sub}/figures/{sub}_{ses}_{fnum}_task-emotionvideos_{run}"
        )
        new_fig_root = (
            f"{sub}/figures/{sub}_{ses}_task-emotion_{new_run}"
        )
        _rename_run_files(
            old_fig_root, new_fig_root, bids_dir, deriv_dir,
        )


if __name__ == "__main__":
    main()