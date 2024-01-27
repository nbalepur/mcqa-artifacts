import pickle
import datasets
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model'))

# specify models, datasets, and the results directory
res_dir = ...
MODELS = ['llama 70b', 'falcon 40b', 'mixtral 7b']
DATASETS = ['ARC', 'MMLU', 'HellaSwag']

pt = 'artifact_choices_cot'

for model_nickname in MODELS:
    for dataset in DATASETS:

        res_dir = f'{res_dir}{dataset}/{model_nickname}/{pt}.pkl'
        out_dir = f'{res_dir}{dataset}/{model_nickname}/gen_question_data.pkl'
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

        out = {'questions': qs, 'choices': cs}
        with open(out_dir, 'wb') as handle:
            pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)