import os
import json
import joblib
import glob

import pandas as pd
import pythia

MODEL_SIZES = ["70m", "410m", "1b", "1.4b", "2.8b", "6.9b"]
SHOT = {"zero": "", "five": "-5shot"}
SHOT_VALUE = {0: 'zero', 5: 'five'}

BASEPATH = "pythia/evals/pythia-v1/"


def make_paths(model_size, shot):
    shot_tag = SHOT[SHOT_VALUE[shot]]
    return glob.glob(
        os.path.join(
            BASEPATH,
            f"pythia-{model_size}-deduped/{SHOT_VALUE[shot]}-shot/{model_size}-deduped{shot_tag}_step*.json"
        ))


def get_eval(path, task):

    json_log = json.load(open(path, 'r'))
    acc, acc_stderr, lh_diff, lh_diff_stderr = None, None, None, None
    if task in json_log['results'].keys():
        results = json_log['results'][task]
        config_log = json_log['config']
        config_model_size = config_log['model_args'].split('-')[2]
        config_step = int(config_log['model_args'].split('step')[-1])
        config_fewshot = config_log['num_fewshot']

        #assert (config_fewshot == shot) and (config_model_size
        #                                     == model_size) and (config_step
        #
        #                                                        == step)
        if 'acc' in results.keys():
            acc = results['acc']
        if 'acc_stderr' in results.keys():
            acc_stderr = results['acc_stderr']
        if 'likelihood_difference' in results.keys():
            lh_diff = results['likelihood_difference']
        if 'likelihood_difference_stderr' in results.keys():
            lh_diff_stderr = results['likelihood_difference_stderr']

    return acc, acc_stderr, lh_diff, lh_diff_stderr


def get_tasks(path):
    tasks = list(json.load(open(path, 'r'))['results'].keys())
    return tasks


def list_tasks():
    tasks = []
    for model_size in MODEL_SIZES:
        for shot in SHOT_VALUE.keys():
            paths = make_paths(model_size, shot)
            for p in paths:
                task_list = get_tasks(p)
                for t in task_list:
                    if not t in tasks:
                        tasks.append(t)

    return tasks


if __name__ == "__main__":
    df_dict = {
        "task": [],
        "shot": [],
        "model_size": [],
        "step": [],
        "acc": [],
        "acc_stderr": [],
        'likelihood_difference': [],
        'likelihood_difference_stderr': []
    }
    entire_task = list_tasks()
    for task in entire_task:
        for model_size in MODEL_SIZES:
            for shot in SHOT_VALUE.keys():
                paths = make_paths(model_size, shot)
                for p in paths:
                    step = int(p.split('step')[-1].split('.json')[0])
                    acc, acc_stderr, lh_diff, lh_diff_stderr = get_eval(
                        p, task=task)
                    df_dict['task'].append(task)
                    df_dict['shot'].append(shot)
                    df_dict['step'].append(step)
                    df_dict['model_size'].append(model_size)
                    df_dict['acc'].append(acc)
                    df_dict['acc_stderr'].append(acc_stderr)
                    df_dict['likelihood_difference'].append(lh_diff)
                    df_dict['likelihood_difference_stderr'].append(
                        lh_diff_stderr)
    df = pd.DataFrame(df_dict)

    df.to_csv("task_performance_summary.csv")
