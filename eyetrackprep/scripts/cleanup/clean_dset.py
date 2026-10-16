import click
import glob
import json
from pathlib import Path

import pandas as pd 


def _get_vals(df_run):
    sub = df_run['subject']
    ses = df_run['session']
    task = df_run['task']
    run = df_run['run']
    fnum = df_run['file_number']
    qc_fail = df_run['exclude']

    return sub, ses, task, run, fnum, qc_fail


def _update_metadata(jpath, qc_check):
    with open(jpath, 'r') as metadata_file:
        mdata_old = json.load(metadata_file)

    if "DriftCorrection_QualityCheck" not in mdata_old:
        mdata_new = {"DriftCorrection_QualityCheck": {
            "Value": qc_check,
            "Description": "Whether drift-corrected gaze passed manual inspection (pass/fail).",
        }}

        m_data = {**mdata_new, **mdata_old}

        with open(jpath, 'w') as metadata_file:
            json.dump(m_data, metadata_file, indent=4)


def _rename_run_files(
    old_root,
    new_root,
    bids_dir,
    deriv_dir,
    qc_fail,
):
    bids_list = sorted(glob.glob(
        f"{str(bids_dir)}/{old_root}*")
    )
    deriv_list = sorted(glob.glob(
        f"{str(deriv_dir)}/{old_root}*")
    )
    qc_tag = 'desc-FAIL' if qc_fail else 'desc-driftcorr'
    for et_file in bids_list + deriv_list:
        Path(et_file).rename(
            et_file.replace(old_root, new_root).replace(
                'desc-driftcorr', qc_tag)
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
        sub, ses, task, run, fnum, qc_fail = _get_vals(df_run)
        old_root = (
            f"{sub}/{ses}/func/{sub}_{ses}_{task}"
            f"_{run}_{fnum}_recording-eye0"
        )

        # Add Drift Correction Quality Check to run metadata
        jpath = (
            f"{str(deriv_dir)}/{old_root}_desc-driftcorr_qcmetrics.json"
        )
        if Path(jpath).exists():
            qc_check = "Fail" if qc_fail else "Pass"
            _update_metadata(jpath, qc_check)

        # Rename old files (rm identifiers)
        new_root = (
            f"{sub}/{ses}/func/{sub}_{ses}_{task}"
            f"_{run}_recording-eye0"
        )
        _rename_run_files(
            old_root, new_root, bids_dir, deriv_dir, qc_fail,
        )

        # Rename old figures (rm identifiers)
        old_fig_root = (
            f"{sub}/figures/{sub}_{ses}_{fnum}_{task}_{run}"
        )
        new_fig_root = (
            f"{sub}/figures/{sub}_{ses}_{task}_{run}"
        )
        _rename_run_files(
            old_fig_root, new_fig_root, bids_dir, deriv_dir, qc_fail,
        )


if __name__ == "__main__":
    main()