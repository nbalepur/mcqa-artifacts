import pickle
import datasets
import copy
import random
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))
from data_loader import create_data_evaluation, DatasetName

# specify models, dataset directory (huggingface), datasets, and the results directory
res_dir = ...
ds_dir = ...
MODELS = ['llama 70b', 'falcon 40b', 'mixtral 7b']
DATASETS = [DatasetName.ARC, DatasetName.HellaSwag, DatasetName.mmlu]


pt = 'artifact_choices_cot'
ds = datasets.load_dataset(ds_dir)

def check_any_match(l1, l2):
    for i in range(len(l1)):
        if l1[i] == l2[i]:
            return True
    return False

for dataset_name in DATASETS:

    data = create_data_evaluation(ds, dataset_name)
    qs = data['questions']
    qs_copy = copy.deepcopy(qs)

    while check_any_match(qs, qs_copy):
        random.shuffle(qs_copy)

    dataset = dataset_name.value

    for model_nickname in MODELS:

        res_dir = f'{res_dir}{dataset}/{model_nickname}/{pt}.pkl'
        out_dir = f'{res_dir}{dataset}/{model_nickname}/random_question_data.pkl'
        with open(res_dir, 'rb') as handle:
            res = pickle.load(handle)

        qs = []
        cs = []
        invalid_count = 0
        for i, r in enumerate(res['raw_text']):
            p = res['prompt'][i]
            if r != None and 'Answer:' in r:
                r_ = r[:r.index('Answer:')].strip()
                qs.append(r_)
                p_ = p.split('\n\n')[-1]
                cs.append(p_.replace('Question:', '').strip())
            else:
                invalid_count += 1
                qs.append(None)
                cs.append(None)

        out = {'questions': qs_copy, 'choices': cs}
        with open(out_dir, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)