EXPERIMENTS_ARTIFACT = [PromptType.normal, PromptType.artifact_choices]
EXPERIMENTS_MEMORIZE = [PromptType.normal, PromptType.memorization_empty, PromptType.memorization_gold, PromptType.memorization_no_choice]
EXPERIMENTS_INFER_QUESTION = EXPERIMENTS_ARTIFACT + [PromptType.artifact_choices_cot_twostep_generated, PromptType.artifact_choices_cot_twostep_random]

# set experiments to one of the lists above
EXPERIMENTS = ... 
# set absolute prefix for results folder
res_prefix = ...
# output directory of the plot
out_dir = ...

import sys
import datasets
import pickle
import os
import numpy as np
import pickle
import pandas as pd
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

from data_loader import create_data_evaluation
from data_loader import PromptType, DatasetName

DATASETS = [DatasetName.ARC, DatasetName.mmlu, DatasetName.HellaSwag]
t_test_strats = {}

colors = {
    'Blue': '#4565ae',
    'Red': '#cd2428',
    'Light Red': '#faa5a7',
    'Dark Red': '#910306',
    'Light Purple': '#cc9bfa',
    'Dark Purple': '#552482'
}

colors_map = {
    PromptType.normal.value: colors['Blue'],
    PromptType.artifact_choices.value: colors['Red'],
    PromptType.memorization_no_choice.value: colors['Light Red'],
    PromptType.memorization_empty.value: colors['Red'],
    PromptType.memorization_gold.value: colors['Dark Red'],
    PromptType.artifact_choices_cot_twostep_generated.value: colors['Light Purple'],
    PromptType.artifact_choices_cot_twostep_random.value: colors['Dark Purple'],
    PromptType.artifact_choices_cot.value: colors['Dark Purple'],
}

pt_names_map = {
    PromptType.normal.value: 'Full Prompt',
    PromptType.artifact_choices.value: 'Choices-Only Prompt',
    PromptType.memorization_empty.value: 'Empty Choices',
    PromptType.memorization_gold.value: 'Gold Choices',
    PromptType.memorization_no_choice.value: 'No Choices',
    PromptType.artifact_choices_cot_twostep_generated.value: 'Infer the Question 2-step',
    PromptType.artifact_choices_cot_twostep_random.value: 'Random Question Prompt',
    PromptType.artifact_choices_cot.value: 'Infer the Question 1-step',
}

model_names_map = {
    'llama 70b': 'LLaMA 70B',
    'llama 13b': 'LLaMA 13B',
    'llama 7b': 'LLaMA 7B',
    'falcon 40b': 'Falcon 40B',
    'mistral 7b': 'Mixtral 8x7B',
    'phi 2': 'Phi 2'
}

patterns = {
    PromptType.artifact_choices_cot_twostep_generated.value: '///',
}

bar_width = 0.3 if len(EXPERIMENTS) == 2 else 0.2
legend_margin = 0.22 # higher is more space
p_value_cutoff = 0.00005
if res_prefix[-1] != '/':
    res_prefix += '/'

MODELS = ['llama 70b', 'falcon 40b', 'mistral 7b', 'phi 2']

reported_res = {'llama 7b': {DatasetName.ARC: 0.5307, DatasetName.HellaSwag: 0.7859, DatasetName.mmlu: 0.3876, DatasetName.Winogrande: 0.7403},
                'llama 13b': {DatasetName.ARC: 0.5939, DatasetName.HellaSwag: 0.8213, DatasetName.mmlu: 0.5577, DatasetName.Winogrande: 0.7664},
                'llama 70b': {DatasetName.ARC: 0.6732, DatasetName.HellaSwag: 0.8733, DatasetName.mmlu: 0.6983, DatasetName.Winogrande: 0.8374}}

def format_models(models):
    models_ = []
    for m in models:
        if 'llama' in m:
            models_.append(f'LLaMA {m.split()[1].upper()}')
        elif 'falcon' in m:
            models_.append(f'Falcon {m.split()[1].upper()}')
        elif 'gpt' in m:
            models_.append(f'GPT {m.split()[1].upper()}')
        else:
            models_.append(m)
    return models_

def format_dataset(ds):
    if ds == DatasetName.mmlu:
        return 'MMLU (5-shot)'
    elif ds == DatasetName.ARC:
        return 'ARC (25-shot)'
    elif ds == DatasetName.HellaSwag:
        return 'HellaSwag (10-shot)'
    else:
        return ds.value

ds = datasets.load_dataset('nbalepur/mcqa_artifacts')

def convert_raw_text(rt):
    if rt == None:
        return 'Z'
    rt_old = rt
    if 'Answer:' in rt:
        rt = rt[rt.index('Answer:') + len('Answer:'):]
    rt = rt[:4]
    rt = rt.strip()
    if rt in {'(A)', '(1)'}:
        return 'A'
    if rt in {'(B)', '(2)'}:
        return 'B'
    if rt in {'(3)', '(C)'}:
        return 'C'
    if rt in {'(D)', '(4)'}:
        return 'D' 
    return 'Z'

fig, axs_ = plt.subplots(1, 3, figsize=(14, 3))
try:
    axs = list(axs_.ravel())
except:
    axs = [axs_]
    axs_ = [[axs_]]
idx_ = 0

random_guess_accuracy, majority_accuracy = dict(), dict()

def get_llm_answer(prompt, answer, choices_true):
    prompt = prompt.split('\n\n')[-1]    
    choices_txt = prompt[prompt.index('Choices:\n') + len('Choices:\n'):prompt.index('Answer:')]
    choices = [x[3:].strip() for x in choices_txt.split('\n')[:-1]]
    if ord(answer) - ord('A') >= len(choices):
        return ''
    return choices[ord(answer) - ord('A')]

def compute_accuracy(p, t):
    arr = []
    for i in range(len(p)):
        p_, t_ = p[i], t[i]
        if p_ == ord('Z'):
            arr.append(0.25)
        else:
            arr.append(int(p_ == t_))
    return np.mean(arr), arr

for dataset_idx, dataset in enumerate(DATASETS):

    benchmark_graph_data = {'LLM': [], 'Strategy': [], 'Accuracy': []}
    arr_map = dict()

    for model_nickname in MODELS:

        for pt in EXPERIMENTS:
            data = create_data_evaluation(ds, dataset, pt)
            questions, choices, answer_letters, answer_texts = data['questions'], data['choices'], data['answer_letters'], data['answer_texts']

            if dataset not in majority_accuracy:
                freq = dict()
                for a in answer_letters:
                    freq[a] = freq.get(a, 0) + 1
                v = list(freq.values())
                max_item = max(freq.items(), key = lambda item: item[1])[0]
                majority_arr_ = [max_item for _ in range(len(questions))]
                majority_arr = [int(majority_arr_[m_idx] == answer_letters[m_idx]) for m_idx in range(len(majority_arr_))]
                majority_accuracy[dataset] = max(v) / sum(v)

            res_dir = f'{res_prefix}{dataset.value}/{model_nickname}/{pt.value}.pkl'
            with open(res_dir, 'rb') as handle:
                res = pickle.load(handle)

            pred_answer_letters = [convert_raw_text(rt) for rt in res['raw_text']]
            orig_pred, orig_true = len(pred_answer_letters), len(answer_letters)
            if PromptType.artifact_choices_cot_twostep_generated in EXPERIMENTS and dataset == DatasetName.mmlu:
                valid_idxs = set(range(int(0.75 * orig_true)))
            else:
                valid_idxs = set(range(orig_true))

            pred_answer_letters = [pred_answer_letters[idx] for idx in range(len(pred_answer_letters))]

            pred_idx = [ord(l) for l in pred_answer_letters]
            true_idx = [ord(l) for l in answer_letters]

            true_idx = [true_idx[idx] for idx in valid_idxs]
            pred_idx = [pred_idx[idx] for idx in valid_idxs]

            assert(len(pred_idx) == len(true_idx))

            accuracy, arr = compute_accuracy(pred_idx, true_idx)
            arr_map[(model_nickname, pt.value)] = arr

            benchmark_graph_data['Accuracy'].append(accuracy)
            benchmark_graph_data['LLM'].append(model_names_map[model_nickname])
            benchmark_graph_data['Strategy'].append(pt.value)

     
    df = pd.DataFrame(benchmark_graph_data)
    df['Strategy'] = pd.Categorical(df['Strategy'], categories=[e.value for e in EXPERIMENTS], ordered=True) #+ ['reported']
    df['LLM'] = pd.Categorical(df['LLM'], categories=[model_names_map[m] for m in MODELS], ordered=True)

    ax = axs[idx_]
    idx_ += 1

    offset = 0
    
    llm_positions = list(range(len(MODELS)))
    relative_positions = list(0 + np.arange(len(EXPERIMENTS) + 1))


    #ax.axhline(random_guess_accuracy[dataset], color='orange', linewidth=2, label='Guessing', ls='--')
    ax.axhline(majority_accuracy[dataset], color='orange', linewidth=2, label='Majority Class', ls='--')

    for idx, strategy in enumerate(df['Strategy'].unique()):
        accuracies = df[df['Strategy'] == strategy]['Accuracy'].values
        bars = ax.bar([pos + relative_positions[idx]*bar_width for pos in llm_positions], accuracies, bar_width, 
                    label=pt_names_map[strategy], color=colors_map[strategy], hatch=patterns.get(strategy, ''))
        
        if strategy in t_test_strats:
            for bar_idx in range(len(bars)):
                bar_offset = 0.02
                t_test = stats.ttest_ind(arr_map[(MODELS[bar_idx], strategy)], majority_arr, alternative='greater')
                
                if t_test.pvalue < p_value_cutoff:
                    ax.text(bars[bar_idx].xy[0] + 0.5 * bar_width, bars[bar_idx].xy[1] + bars[bar_idx]._height + bar_offset, '*', fontsize=12, fontweight='semibold', verticalalignment='center', horizontalalignment='center')
        
        
    ax.set_title(f'{format_dataset(dataset)}')
    ax.set_xticks(np.array(llm_positions) + (len(EXPERIMENTS) * 0.5) * bar_width - 0.5 * bar_width)
    ax.set_xticklabels([m for m in df['LLM'].unique()])
    ax.set_ylim(top=1.0)
    ax.set_ylim(bottom=0)

    if dataset_idx % 2 == 0:
        ax.set_ylabel('Accuracy')

all_handles = []
all_labels = []

if type(axs_[0]) == type([]):
    for ax in axs_:
        for a in ax:
            h, l = a.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)
else:
    for a in axs_:
        h, l = a.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels.extend(l)


# To ensure that the legend is unique
unique = [(h, l) for i, (h, l) in enumerate(zip(all_handles, all_labels)) if l not in all_labels[:i]]
unique_handles, unique_labels = zip(*unique)

fig.legend(unique_handles, unique_labels, fontsize=12, loc='lower center', ncol=5)

plt.tight_layout()
plt.subplots_adjust(bottom=legend_margin)
plt.savefig(out_dir, dpi=500)