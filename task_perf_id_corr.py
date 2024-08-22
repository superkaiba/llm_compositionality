import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

SUMMARY = pd.read_csv('task_performance_summary.csv')

TASKS = SUMMARY['task'].unique()
MODELS = SUMMARY['model_size'].unique()

ID_MODELS = ['70m', '160m', '410m', '1b', '1.4b', '2.8b', '6.9b']

ALPHAS = {'70m': 0, '410m': 1, '1b': 2, '1.4b': 3, '2.8b': 4, '6.9b': 5}
STEPS = SUMMARY['step'].unique()

colormap = plt.cm.viridis  #or any other colormap
normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)


def get_results(task, model_size):
    cond = (SUMMARY['task'] == task) & (SUMMARY['model_size']
                                        == model_size) & (SUMMARY['shot'] == 5)
    task_perf = SUMMARY[cond].get(key=[
        'step', 'acc', 'acc_stderr', 'likelihood_difference',
        'likelihood_difference_stderr'
    ])

    result_dict = {
        'acc': {},
        'acc_stderr': {},
        'likelihood_difference': {},
        'likelihood_difference_stderr': {}
    }
    for i, item in task_perf.iterrows():
        for metric in result_dict.keys():
            result_dict[metric][int(item['step'])] = item[metric]

    return result_dict


def get_results_id_perf():

    result_dict = {
        'acc': [],
        'acc_stderr': [],
        'likelihood_difference': [],
        'likelihood_difference_stderr': [],
        'method': [],
        'layer_num': [],
        'ordered_id': [],
        'shuffled_id': [],
        'checkpoint_step': [],
        'model_size': [],
        'task': [],
        'shot': []
    }
    for model_size in MODELS:
        print(model_size)
        id_result_ordered = pd.read_csv(
            f'results/ordered/results_{model_size}.csv')
        id_result_shuffled = pd.read_csv(
            f'results/shuffled/results_{model_size}.csv')
        for layer_num in id_result_ordered['layer_num'].unique():
            for method in id_result_ordered['method'].unique():
                for n in id_result_ordered['n_words_correlated'].unique():
                    for shot in [0, 5]:
                        for task in TASKS:
                            for step in STEPS:
                                task_cond = (SUMMARY['task'] == task) & (
                                    SUMMARY['model_size']
                                    == model_size) & (SUMMARY['shot'] == shot)
                                id_cond = (
                                    id_result_ordered['model_size']
                                    == model_size
                                ) & (id_result_ordered['checkpoint_step']
                                     == step) & (
                                         id_result_ordered['layer_num']
                                         == layer_num) & (
                                             id_result_ordered['method']
                                             == method) & (id_result_ordered[
                                                 'n_words_correlated'] == n)
                                if len(id_cond) == 1:
                                    result_dict['shot'].append(shot)
                                    result_dict['model_size'].append(
                                        model_size)
                                    result_dict['task'].append(task)
                                    result_dict['step'].append(step)
                                    result_dict['acc'].append(
                                        SUMMARY[task_cond]['acc'])
                                    result_dict['acc_stderr'].append(
                                        SUMMARY[task_cond]['acc_stderr'])
                                    result_dict[
                                        'likelihood_difference'].append(
                                            SUMMARY[task_cond]
                                            ['likelihood_difference'])
                                    result_dict[
                                        'likelihood_difference_stderr'].append(
                                            SUMMARY[task_cond]
                                            ['likelihood_difference_stderr'])
                                    result_dict['layer_num'].append(layer_num)
                                    result_dict['method'].append(method)
                                    result_dict['ordered_id'].append(
                                        id_result_ordered[id_cond]['id'])
                                    result_dict['shuffled_id'].append(
                                        id_result_shuffled[id_cond]['id'])

    return result_dict


def get_corr(prompt_type="ordered"):

    def append_results(df_dict, **kwargs):
        for k, v in kwargs.items():
            df_dict[k].append(v)
        return df_dict

    corr_results = {
        'task': [],
        'model_size': [],
        'n_words_correlated': [],
        'method': [],
        'layer_num': [],
        'corrcoef': []
    }
    RESULTS_PATH = f'results/{prompt_type}/results_'
    for task in TASKS:
        if 'crows' in task:
            metric = 'likelihood_difference'
        else:
            metric = 'acc'
        for model_size in ID_MODELS:
            performance_result = get_results(task, model_size)[metric]
            if len(performance_result) == 0:
                continue
            id_result = pd.read_csv(RESULTS_PATH + f'{model_size}.csv')
            layer_nums = id_result['layer_num'].unique()
            corr_words_n = id_result['n_words_correlated'].unique()
            methods = id_result['method'].unique()
            for l in layer_nums:
                for c in corr_words_n:
                    for m in methods:
                        cond = (id_result['n_words_correlated']
                                == c) & (id_result['method']
                                         == m) & (id_result['layer_num'] == l)
                        ckpt_inds = id_result[cond]['checkpoint_step']
                        id_list = id_result['id'][np.argsort(ckpt_inds).index]
                        perf_list = []
                        for i in ckpt_inds.values:
                            if i in performance_result.keys():
                                perf_list.append(performance_result[i])
                            else:
                                closest_arg = np.argmin(
                                    np.abs(
                                        np.array(
                                            list(performance_result.keys())) -
                                        i))
                                perf_list.append(performance_result[list(
                                    performance_result.keys())[closest_arg]])
                        corr_coef = np.corrcoef(id_list, perf_list)[0, 1]
                        corr_results = append_results(corr_results,
                                                      task=task,
                                                      model_size=model_size,
                                                      n_words_correlated=c,
                                                      method=m,
                                                      layer_num=l,
                                                      corrcoef=corr_coef)

    return corr_results


### Plot all the task evaluations
def plot():
    if not os.path.isdir('results/task_performance'):
        os.mkdir('results/task_performance')
    for TASK in TASKS:
        fig, axs = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
        for MODEL in MODELS:
            result_dict = get_results(TASK, MODEL)
            for i, metric in enumerate(['acc', 'likelihood_difference']):
                sorted = np.argsort(list(result_dict[metric].keys()))
                axs[i].errorbar(
                    np.array(list(result_dict[metric].keys()))[sorted],
                    np.array(list(result_dict[metric].values()))[sorted],
                    yerr=np.array(
                        list(result_dict[metric +
                                         '_stderr'].values()))[sorted],
                    marker='o',
                    ls='-',
                    label=MODEL,
                    color=colormap(normalize(ALPHAS[MODEL])))

                axs[i].set_ylabel(metric)
                axs[i].set_xscale('log')
                axs[i].spines['right'].set_visible(False)
                axs[i].spines['top'].set_visible(False)
                axs[i].legend(bbox_to_anchor=(1, 1), frameon=False)
            axs[i].set_xlabel('Step')
        fig.suptitle(TASK)
        fig.savefig(f'results/task_performance/{TASK}.png',
                    transparent=False,
                    pad_inches=3)


def plot_perf_ID():
    if not os.path.isdir('results/task_ID'):
        os.mkdir('results/task_ID')

    result = pd.read_csv('task_id_merged.csv')
    tasks = result['task'].unique()
    models = result['model_size'].unique()
    methods = result['method'].unique()
    steps = result['checkpoint_step'].unique()
    shot = 0

    acc_diff = {}
    id_idff = {}

    for task in tasks:
        if 'crow' not in task:
            for i, method in enumerate(methods):
                fig, axs = plt.subplots(1, 4, figsize=(8, 5), sharex=True)
                for model in models:
                    last_layer = max(
                        result[result['model'] == model]['layer_num'].unique())
                    cond = (result['model'] == model) & (
                        result['task']
                        == task) & (result['layer_num'] == last_layer) & (
                            result['method'] == method) & (result['shot']
                                                           == shot)
                    argsorted_steps = np.argsort(task[cond]['step'])
                    sorted_steps = task[cond]['step'][argsorted_steps]
                    ordered_id_series = task[cond]['ordered_id'][
                        argsorted_steps]
                    shuffled_id_series = task[cond]['shuffled_id'][
                        argsorted_steps]
                    acc = task[cond]['acc'][argsorted_steps]
                    acc_stderr = task[cond]['acc_stderr'][argsorted_steps]

                    acc_diff = acc[1:] - acc[:-1]
                    id_diff = ordered_id_series[1:] - ordered_id_series[:-1]
                    """
                    axs[i].plot(list(sorted_steps), ordered_id_series)
                    axs[i].errorbar(list(sorted_steps),
                                    list(acc),
                                    yerr=list(acc_stderr),
                                    marker='o',
                                    ls='-',
                                    label=model,
                                    color=colormap(normalize(ALPHAS[model])))
                    """


if __name__ == "__main__":

    #plot()
    #corr_results = get_corr()
    #df = pd.DataFrame(corr_results)
    #df.to_csv("task_id_corr_ordered_five.csv")
    task_perf_id = get_results_id_perf()
    df = pd.DataFrame(task_perf_id)
    df.to_csv("task_id_merged.csv")
