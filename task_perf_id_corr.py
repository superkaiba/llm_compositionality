import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

SUMMARY = pd.read_csv('task_performance_summary.csv')

TASKS = SUMMARY['task'].unique()
MODELS = SUMMARY['model_size'].unique()

ID_MODELS = ['70m', '160m', '410m', '1b', '1.4b', '2.8b']

ALPHAS = {'70m': 0, '410m': 1, '1b': 2, '1.4b': 3, '2.8b': 4, '6.9b': 5}
STEPS = SUMMARY['step'].unique()

RESULTS_PATH = 'results/ordered/results_'

colormap = plt.cm.viridis  #or any other colormap
normalize = matplotlib.colors.Normalize(vmin=0, vmax=5)


def get_results(task, model_size):
    cond = (SUMMARY['task'] == task) & (SUMMARY['model_size'] == model_size)
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


def get_corr():

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
                        """
                        if all([
                                i in performance_result.keys()
                                for i in ckpt_inds.values
                        ]):
                            print('hey')
                            perf_list = [
                                performance_result[int(ckpt_ind)]
                                for ckpt_ind in ckpt_inds
                            ]
                            corr_coef = np.corrcoef(id_list, perf_list)
                            print('corr_coef')
                            corr_results = append_results(
                                corr_results,
                                task=task,
                                model_size=model_size,
                                n_words_correlated=c,
                                method=m,
                                layer_num=l,
                                corrcoef=corr_coef)
                        
                        else:
                            missing = []
                            for i in performance_result.keys():
                                if i not in ckpt_inds.values:
                                    missing.append(i)
                            print(ckpt_inds.values)
                            print(missing)
                        """
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


corr_results = get_corr()
df = pd.DataFrame(corr_results)
df.to_csv("task_id_corr.csv")
"""

fig, axs = plt.subplots(12, 6, figsize=(15, 10), sharex=True)

ACC_TASKS = [t for t in TASKS if 'crows' not in t]
for i, TASK in enumerate(ACC_TASKS):
    for MODEL in MODELS:
        result_dict = get_results(TASK, MODEL)
        metric = 'acc'

        sorted = np.argsort(list(result_dict[metric].keys()))
        axs[i // 6][i % 6].errorbar(
            np.array(list(result_dict[metric].keys()))[sorted],
            np.array(list(result_dict[metric].values()))[sorted],
            yerr=np.array(list(result_dict[metric +
                                           '_stderr'].values()))[sorted],
            marker='o',
            ls='-',
            label=MODEL,
            color=colormap(normalize(ALPHAS[MODEL])),
            ms=1)

        axs[i // 6][i % 6].set_title(f'{i//10}, {i%10}')

fig.savefig('test.png')
"""
