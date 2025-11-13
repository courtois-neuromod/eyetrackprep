import os, sys, json

from pathlib import Path
import click
from src import pupil2bids

# TODO: fix this by importing pupil as library 
sys.path.append(
    os.path.join(
        "/home/mariestl/cneuromod/ds_prep/eyetracking",
        "pupil",
        "pupil_src",
        "shared_modules",
    )
)
from file_methods import load_pldata_file

@click.command()
@click.argument(
    "raw_et_dir",
    type=click.Path(),
)
@click.argument(
    "out_dir",
    type=click.Path(),
)
def main(
    raw_et_dir,
    out_dir,
    export_plots=False,
    drift_corr=False,
):
    """."""

    pupil_file_paths = pupil2bids.compile_rawfile_list(
        raw_et_dir, out_dir)

    task_root = raw_et_dir.split('/')[-2]
    qc_report = open(f"{out_dir}/QC_gaze/qc_report_{task_root}.txt", 'w+')

    for pupil_path in pupil_file_paths:
        seri_gaze = load_pldata_file(pupil_path[0], 'gaze')[0]
        sub, ses, run, task, fnum = pupil_path[1]
        
        # early mario3 runs accidentally labelled task-mariostars...
        pseudo_task = 'task-mario3' if task_root == 'mario3' else task    
        qc_report.write(f"\n{sub} {ses} {run} {pseudo_task} {task} {len(seri_gaze)}\n")    
        #print(sub, ses, run, pseudo_task, len(seri_gaze))
        if len(seri_gaze) < 1:
            qc_report.write(f"No pupils found for {pupil_path[1]}\n")
        elif len(seri_gaze) < 11:
            qc_report.write(f"Fewer than 11 pupils found for {pupil_path[1]}\n")
        else:
            gz_ts = seri_gaze[10]['timestamp']

            if task_root in ['floc', 'retino', 'langloc', 'ood']:
                tname = task
            elif task_root == 'friends':
                tname = task.replace("task-", "task-friends-")
            elif task_root == 'narratives':
                tname = f'{task.replace("part", "_part")}_run-01'
            else:
                tname = f'{task}_{run}'

            infoplayer_path = f'{raw_et_dir}/{sub}/{ses}/{sub}_{ses}_{fnum}.pupil/{tname}/000/info.player.json'                
            log_path = f'{raw_et_dir}/{sub}/{ses}/{sub}_{ses}_{fnum}.log'
            
            onset_time_dict = {}
            TTL_0 = -1
            has_lines = True

            if not Path(log_path).exists():
                has_lines = False
                qc_report.write(
                    f"No .log file found for {pupil_path[1]} \n")
            else:
                with open(log_path) as f:
                    lines = f.readlines()
                    if len(lines) == 0:
                        has_lines = False
                        qc_report.write(f"Empty .log file for {pupil_path[1]}\n")
                    for line in lines:
                        if "fMRI TTL 0" in line:
                            TTL_0 = line.split('\t')[0]
                        elif "saved wide-format data to /scratch/neuromod/data" in line:
                            run_id = line.split('\t')[-1].split(f'{fnum}_')[-1].split('_events.tsv')[0]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.localizers.FLoc'" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.retinotopy.Retinotopy'" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.mutemusic.Playlist'" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.videogame.VideoGameMultiLevel'" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.video.SingleVideo'" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
                        elif "class 'src.tasks.narratives" in line:
                            run_id = line.split(': ')[-2]
                            onset_time_dict[run_id] = float(TTL_0)
            
            if Path(infoplayer_path).exists():
                try:
                    with open(infoplayer_path, 'r') as f:
                        iplayer = json.load(f)
                    sync_ts = iplayer['start_time_synced_s']
                    syst_ts = iplayer['start_time_system_s']   
                except: 
                    qc_report.write(f"Failed info.player.json parsing for {pupil_path[1]}\n")
            else:
                qc_report.write(f"No info.player.json file found for {pupil_path[1]}\n")
            if has_lines:
                try:
                    o_time = onset_time_dict[tname]
                except: 
                    qc_report.write(f"Run name not found during .log text parsing for {pupil_path[1]}\n")

    qc_report.close()

