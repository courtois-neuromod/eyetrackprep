import os, json
import datetime
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

import types
import msgpack
import collections


def unpacking_object_hook(obj):
    if isinstance(obj, dict):
        return types.MappingProxyType(obj)
    return obj


def unpacking_ext_hook(code, data):
    if code == MSGPACK_EXT_CODE:
        return msgpack.unpackb(
            data,
            use_list=False,
            object_hook=unpacking_object_hook,
            ext_hook=unpacking_ext_hook,
            strict_map_key=False,
        )
    return msgpack.ExtType(code, data)


def load_pldata_file(directory, topic):
    """
    Deserialize pldata
    """
    msgpack_file = os.path.join(directory, topic + ".pldata")
    is_deserialized = True

    try:
        data = collections.deque()

        with open(msgpack_file, "rb") as stream:
            unpacker = msgpack.Unpacker(
                stream,
                use_list=False,
                strict_map_key=False
            )

            for _, to_unpack in unpacker:
                unpacked = msgpack.unpackb(
                    to_unpack, 
                    use_list=False, 
                    object_hook=unpacking_object_hook, 
                    ext_hook=unpacking_ext_hook, 
                    strict_map_key=False
                )
                data.append(unpacked)

    except FileNotFoundError as err:
        print(f"Couldn't read or unpack: {err}")
        data = []
        is_deserialized = False

    return data, is_deserialized


def log_qc(
    log_message,
    qc_path
) -> None:
    """."""
    print(log_message)    
    with open(qc_path, 'a') as qc_report:
        qc_report.write(f"{log_message}\n")    


def init_log(
    log_dir: str,
    task_root: str,
    job_tag: str,
) -> None: 
    """."""
    log_path = f"{log_dir}/code/QC_gaze/{job_tag}_report_{task_root}.txt"
    if Path(log_path).exists():
        log_qc(f"\n---------------------------\n{datetime.datetime.now()}\n", log_path)
    else:
        Path(f"{log_dir}/code/QC_gaze").mkdir(parents=True, exist_ok=True)
        Path(log_path).touch()


def init_logs(
    task_root: str,
    correct_drift: bool,
    export_plots: bool,
    out_dir: str,
    deriv_dir: Union[str, None],
) -> None: 
    """."""
    init_log(out_dir, task_root, 'qc')
    if correct_drift:
        init_log(deriv_dir, task_root, 'qc')
        if export_plots:
            init_log(deriv_dir, task_root, 'plot')
    elif export_plots:
        init_log(out_dir, task_root, 'plot')
        

def get_metadata(
    start_time: float,
    col_names: list[str],
) -> dict :
    """."""
    return {
        "StartTime": start_time,
        "Columns": col_names,
        "SamplingFrequency": 250.0,
    }


def parse_task_name(
    pupfile_path,
    task_root,
) -> tuple[str]:
    """."""
    pupil_str, task_str = pupfile_path.split('/')[-3:-1]
    sub, ses, fnum = pupil_str.split('.')[0].split('_')

    if task_root == "friends":
        task_type = f'task-{task_str.split("-")[-1]}'
    elif task_root == "ood":
        task_type = task_str
    elif task_root == "narratives":
        task_type = task_str.split("_run-")[0].replace("_", "")

    return sub, ses, fnum, task_type


def get_event_path(
    pupil_path: str,
) -> str:
    '''
    Parses gaze.pldata parent directory's path, 
    returns path to corresponding run's events.tsv file.

    Works for tasks with events.tsv files: 
        emotionsvideos, floc, friends_fix, langloc, mario, 
        mariostars, mario3, movie10_fix, multfs, mutemusic, 
        narratives (recency sub-task only), retino, things, 
        and triplets
    '''
    parent_path = str(Path(pupil_path).parents[2])
    pupil_str, task_str = pupil_path.split('/')[-3:-1]
    
    return f"{parent_path}/{pupil_str.split('.')[0]}_{task_str}_events.tsv"
