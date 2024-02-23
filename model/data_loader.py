from enum import Enum
import numpy as np
from prompt import Normal, MemoriazationNoChoices, MemoriazationRepeatGoldChoices, MemoriazationEmptyChoices, ArtifactChoices, ArtifactChoicesQuestionCOT, TwoChoices, ThreeChoices, ShuffleChoices, ChoiceA, ChoiceB, ChoiceC, ChoiceD, ChoiceAQuestion, ChoiceBQuestion, ChoiceCQuestion, ChoiceDQuestion
from datasets.utils.logging import disable_progress_bar
disable_progress_bar()

class PromptType(Enum):
    normal = 'normal' # Full MCQA Prompt
    artifact_choices = 'artifact_choices' # Choices-Only Prompt

    memorization_no_choice = 'memorization_no_choice' # Memorization Prompt - no choices shown
    memorization_gold = 'memorization_gold' # Memorization Prompt - all choices are the gold answer
    memorization_empty = 'memorization_empty' # Memorization Prompt - all choices are empty

    choice_a_even = 'choice_a_even' # Independently classify the correctness of each option A, without the question
    choice_b_even = 'choice_b_even' # Independently classify the correctness of each option B, without the question
    choice_c_even = 'choice_c_even' # Independently classify the correctness of each option C, without the question
    choice_d_even = 'choice_d_even' # Independently classify the correctness of each option D, without the question

    choice_a_question_even = 'choice_a_question_even' # Independently classify the correctness of each option A, with the question
    choice_b_question_even = 'choice_b_question_even' # Independently classify the correctness of each option B, with the question
    choice_c_question_even = 'choice_c_question_even' # Independently classify the correctness of each option C, with the question
    choice_d_question_even = 'choice_d_question_even' # Independently classify the correctness of each option D, with the question

    artifact_choices_cot = 'artifact_choices_cot' # Step 1 of Inferring the Question
    artifact_choices_cot_twostep_generated = 'artifact_choices_cot_twostep_generated' # Step 2 of Inferring the Question
    artifact_choices_cot_twostep_random = 'artifact_choices_cot_twostep_random' # Inferring the Question comparison with Random Question

    two_choices = 'two_choices' # 2 Choices out of 4 (not in paper)
    three_choices = 'three_choices' # 3 Choices out of 4 (not in paper)
    shuffle_choices = 'shuffle_choices' # Shuffle the MC choices (not in paper)

    choice_a = 'choice_a' # Independently classify the correctness of each option A, without the question (75/25 prior)
    choice_b = 'choice_b' # Independently classify the correctness of each option B, without the question (75/25 prior)
    choice_c = 'choice_c' # Independently classify the correctness of each option C, without the question (75/25 prior)
    choice_d = 'choice_d' # Independently classify the correctness of each option D, without the question (75/25 prior)

    choice_a_question = 'choice_a_question_' # Independently classify the correctness of each option A, with the question (75/25 prior)
    choice_b_question = 'choice_b_question' # Independently classify the correctness of each option B, with the question (75/25 prior)
    choice_c_question = 'choice_c_question' # Independently classify the correctness of each option C, with the question (75/25 prior)
    choice_d_question = 'choice_d_question' # Independently classify the correctness of each option D, with the question (75/25 prior)


class DatasetName(Enum):
    mmlu = 'mmlu' # MMLU
    HellaSwag = 'HellaSwag' # HellaSwag
    ARC = 'ARC' # ARC
    Winogrande = 'Winogrande' # Winogrande (not in paper)

# class DatasetName(Enum):
#     ARC = 'ARC'
#     CQA = 'CQA'
#     OBQA = 'OBQA'
#     PIQA = 'PIQA'
#     QASC = 'QASC'
#     SIQA = 'SIQA'

prompt_type_map = {
    PromptType.normal: Normal,
    PromptType.memorization_no_choice: MemoriazationNoChoices,
    PromptType.memorization_gold: MemoriazationRepeatGoldChoices,
    PromptType.memorization_empty: MemoriazationEmptyChoices,
    PromptType.artifact_choices: ArtifactChoices,
    PromptType.artifact_choices_cot: ArtifactChoicesQuestionCOT,
    PromptType.two_choices: TwoChoices,
    PromptType.three_choices: ThreeChoices,
    PromptType.shuffle_choices: ShuffleChoices,
    PromptType.choice_a: ChoiceA,
    PromptType.choice_b: ChoiceB,
    PromptType.choice_c: ChoiceC,
    PromptType.choice_d: ChoiceD,
    PromptType.choice_a_question: ChoiceAQuestion,
    PromptType.choice_b_question: ChoiceBQuestion,
    PromptType.choice_c_question: ChoiceCQuestion,
    PromptType.choice_d_question: ChoiceDQuestion,
}

def create_data_choices_even_mmlu(dataset, dataset_name, prompt_type, prompt_dir, use_20_fewshot=False):

    if prompt_dir[-1] != '/':
        prompt_dir += '/'

    suffix = ''
    if 'question' in prompt_type.value:
        suffix += "_question"
    if use_20_fewshot:
        suffix += "_20"

    # load data and prompt objects
    train_ds, test_ds = dataset['train'], dataset['test']

    # get all tagged datasets
    train_ds_ = train_ds.filter(lambda example: dataset_name.value in example['dataset'])
    test_ds_ = test_ds.filter(lambda example: dataset_name.value in example['dataset'])
    unique_datasets = sorted(list(set(train_ds['dataset'])))

    final_input_prompts = []
    final_output_letters = []

    # get prompts for each unique subdataset (only special for MMLU)
    for ds in unique_datasets:
        train_ds = train_ds_.filter(lambda example: example['dataset'] == ds)
        test_ds = test_ds_.filter(lambda example: example['dataset'] == ds)

        if train_ds.num_rows == 0:
            continue

        f = open(f'{prompt_dir}{ds}/{dataset_name.value}{suffix}.txt', 'r')
        base_prompt = ''.join(f.readlines())

        idx_map = {PromptType.choice_a_even: 0, PromptType.choice_a_question_even: 0, PromptType.choice_b_even: 1, PromptType.choice_b_question_even: 1, PromptType.choice_c_question_even: 2, PromptType.choice_c_even: 2, PromptType.choice_d_even: 3, PromptType.choice_d_question_even: 3}
        choice_idx = idx_map[prompt_type]

        input_prompts = []
        output_letters = []
        questions, choices, answers = test_ds['question'], test_ds['choices'], test_ds['answer_letter']
        for i in range(len(questions)):
            q, c, a = questions[i], choices[i], answers[i]
            if 'question' in prompt_type.value:
                curr_prompt = f"{base_prompt}\n\nQuestion: {q}\nChoice: {c[choice_idx]}\nAnswer:"
            else:
                curr_prompt = f"{base_prompt}\n\nChoice: {c[choice_idx]}\nAnswer:"
            input_prompts.append(curr_prompt)
            output_letters.append('True' if ((ord(a) - ord('A')) == choice_idx) else 'False')
            
        # append to list
        final_input_prompts.extend(input_prompts)
        final_output_letters.extend(output_letters)


    return {'input': final_input_prompts, 'output': final_output_letters, 'stop_token': '\nQuestion:' if 'question' in prompt_type.value else '\nChoice:'}

def create_data_choices_even(dataset, dataset_name, prompt_type, prompt_dir, use_20_fewshot=False):

    if prompt_dir[-1] != '/':
        prompt_dir += '/'

    if dataset_name == DatasetName.mmlu:
        return create_data_choices_even_mmlu(dataset, dataset_name, prompt_type, prompt_dir, use_20_fewshot)

    if dataset_name not in [DatasetName.ARC, DatasetName.HellaSwag]:
        print(f"Sorry, {dataset_name} is not supported!")
        exit(0)

    suffix = ''
    if 'question' in prompt_type.value:
        suffix += "_question"
    if use_20_fewshot:
        suffix += "_20"

    f = open(f'{prompt_dir}{dataset_name.value}{suffix}.txt', 'r')
    base_prompt = ''.join(f.readlines())

    idx_map = {PromptType.choice_a_even: 0, PromptType.choice_a_question_even: 0, PromptType.choice_b_even: 1, PromptType.choice_b_question_even: 1, PromptType.choice_c_question_even: 2, PromptType.choice_c_even: 2, PromptType.choice_d_even: 3, PromptType.choice_d_question_even: 3}
    choice_idx = idx_map[prompt_type]

    test_ds = dataset['test']
    test_ds = test_ds.filter(lambda example: dataset_name.value in example['dataset'])

    final_input_prompts = []
    final_output_letters = []

    questions, choices, answers = test_ds['question'], test_ds['choices'], test_ds['answer_letter']
    for i in range(len(questions)):
        q, c, a = questions[i], choices[i], answers[i]
        if 'question' in prompt_type.value:
            curr_prompt = f"{base_prompt}\n\nQuestion: {q}\nChoice: {c[choice_idx]}\nAnswer:"
        else:
            curr_prompt = f"{base_prompt}\n\nChoice: {c[choice_idx]}\nAnswer:"
        final_input_prompts.append(curr_prompt)
        final_output_letters.append('True' if ((ord(a) - ord('A')) == choice_idx) else 'False')

    return {'input': final_input_prompts, 'output': final_output_letters, 'stop_token': '\nQuestion:' if 'question' in prompt_type.value else '\nChoice:'}

def create_data(dataset, dataset_name, prompt_type, prompt_dir, use_20_fewshot=False):

    if 'even' in prompt_type.value:
        return create_data_choices_even(dataset, dataset_name, prompt_type, prompt_dir, use_20_fewshot)

    # load data and prompt objects
    train_ds, test_ds = dataset['train'], dataset['test']
    prompt_object = prompt_type_map[prompt_type]()

    # get all tagged datasets
    train_ds_ = train_ds.filter(lambda example: dataset_name.value in example['dataset'])
    test_ds_ = test_ds.filter(lambda example: dataset_name.value in example['dataset'])
    unique_datasets = sorted(list(set(train_ds['dataset'])))

    final_input_prompts = []
    final_output_letters = []

    for ds in unique_datasets:
        train_ds = train_ds_.filter(lambda example: example['dataset'] == ds)
        test_ds = test_ds_.filter(lambda example: example['dataset'] == ds)

        if use_20_fewshot:
            train_ds = train_ds.select(np.arange(20))

        # create few-shot prompt
        questions, choices, answers = train_ds['question'], train_ds['choices'], train_ds['answer_letter']
        prompt_object.create_fewshot_prompt(questions, choices, answers)

        # create inference prompts
        questions, choices, answers = test_ds['question'], test_ds['choices'], test_ds['answer_letter']
        input_prompts = [prompt_object.create_inference_prompt(questions[i], choices[i], answers[i]) for i in range(len(questions))]

        # append to list
        final_input_prompts.extend(input_prompts)
        final_output_letters.extend(list(answers))

    return {'input': final_input_prompts, 'output': final_output_letters, 'stop_token': prompt_object.define_stop_token()}

def create_data_evaluation(dataset, dataset_name, prompt_type=PromptType.normal):

    # load data and prompt objects
    train_ds, test_ds = dataset['train'], dataset['test']

    # get all tagged datasets
    train_ds_ = train_ds.filter(lambda example: dataset_name.value in example['dataset'])
    test_ds_ = test_ds.filter(lambda example: dataset_name.value in example['dataset'])

    unique_datasets = sorted(list(set(train_ds['dataset'])))

    # get prompts for each unique subdataset (only special for MMLU)
    all_questions, all_choices, all_answer_letters, all_answer_texts = [], [], [], []
    for ds in unique_datasets:
        train_ds = train_ds_.filter(lambda example: example['dataset'] == ds)
        test_ds = test_ds_.filter(lambda example: example['dataset'] == ds)

        # inference data
        questions, choices, answers = test_ds['question'], test_ds['choices'], test_ds['answer_letter']
        answer_texts = [choices[i][ord(answers[i]) - ord('A')] for i in range(len(choices))]
        all_questions.extend(questions)
        all_choices.extend(choices)
        all_answer_letters.extend(answers)
        all_answer_texts.extend(answer_texts)

    return {'questions': all_questions, 'choices': all_choices, 'answer_letters': all_answer_letters, 'answer_texts': all_answer_texts}

def create_data_merge(dataset, dataset_name):

    # load data and prompt objects
    train_ds, test_ds = dataset['train'], dataset['test']

    # get all tagged datasets
    train_ds_ = train_ds.filter(lambda example: dataset_name in example['dataset'])
    test_ds_ = test_ds.filter(lambda example: dataset_name in example['dataset'])

    unique_datasets = sorted(list(set(train_ds['dataset'])))

    # get prompts for each unique subdataset (only special for MMLU)
    all_questions, all_choices, all_answer_letters, all_answer_texts = [], [], [], []
    for ds in unique_datasets:
        train_ds = train_ds_.filter(lambda example: example['dataset'] == ds)
        test_ds = test_ds_.filter(lambda example: example['dataset'] == ds)

        # inference data
        questions, choices, answers = test_ds['question'], test_ds['choices'], test_ds['answer_letter']
        answer_texts = [choices[i][ord(answers[i]) - ord('A')] for i in range(len(choices))]
        all_questions.extend(questions)
        all_choices.extend(choices)
        all_answer_letters.extend(answers)
        all_answer_texts.extend(answer_texts)

    return {'questions': all_questions, 'choices': all_choices, 'answer_letters': all_answer_letters, 'answer_texts': all_answer_texts}