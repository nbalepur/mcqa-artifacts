from abc import ABC, abstractmethod
import random
import copy

# Abstract base class for implementing few-shot prompts
class MultipleChoicePrompt(ABC):

    DELIM = '\n\n'

    def __init__(self):
        self.few_shot_prompt = ''

    # turns options into text format
    def options_to_text(self, options):
        return '\n'.join([f'({chr(ord("A") + i)}) {c}' for i, c in enumerate(options)]) + '\n'

    @abstractmethod
    def create_fewshot_example(self, question, options, is_inference, answer):
        pass

    @abstractmethod
    def define_stop_token(self):
        pass

    # creates an n-shot prompt (training)
    def create_fewshot_prompt(self, questions, options, answers):
        assert(len(questions) == len(options) == len(answers))
        few_shot_examples = []
        for i in range(len(questions)):
            curr_prompt = self.create_fewshot_example(questions[i], options[i], False, answers[i])
            few_shot_examples.append(curr_prompt)
        out = self.DELIM.join(few_shot_examples)
        self.few_shot_prompt = out
        return out

    # creates an inference prompt
    def create_inference_prompt(self, question, options, answer):
        curr_prompt = self.few_shot_prompt
        curr_prompt += self.DELIM
        curr_prompt += self.create_fewshot_example(question, options, True, answer)
        return curr_prompt


# Normal MCQA prompt
class Normal(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += f'Choices:\n{self.options_to_text(options)}'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

# ========================================== Memorization Prompts ==========================================

class MemoriazationNoChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class MemoriazationRepeatGoldChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        true_text = options[ord(answer) - ord('A')]
        choices = [true_text for _ in options]
        prompt += f'Choices:\n{self.options_to_text(choices)}'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class MemoriazationEmptyChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        choices = ['' for _ in options]
        prompt += f'Choices:\n{self.options_to_text(choices)}'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

# ========================================== Artifact Prompts ==========================================

def eliminate_choice(options, answer):
    idxs = list(range(len(options)))
    options_cpy = copy.deepcopy(options)
    answer_text = options_cpy[ord(answer) - ord('A')]
    while True:
        rand_idx = random.choice(idxs)
        if rand_idx == (ord(answer) - ord('A')):
            continue
        options_cpy.pop(rand_idx)
        break
    new_answer = chr(ord('A') + options_cpy.index(answer_text))
    return options_cpy, new_answer

class ArtifactChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n{self.options_to_text(options)}'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

class ArtifactChoicesQuestionCOT(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n{self.options_to_text(options)}'
        if not is_inference:
            prompt += f'Question: {question}\nAnswer: ({answer})'
        else:
            prompt += 'Question:'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

# ========================================== Robustness Prompts ==========================================

class ShuffleChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'

        if is_inference:
            options_copy = copy.deepcopy(options)
            while options_copy == options:
                random.shuffle(options_copy)
            options = options_copy

        prompt += f'Choices:\n{self.options_to_text(options)}'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' ({answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class ThreeChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        answer_text = options[ord(answer) - ord('A')]

        options_, new_answer = eliminate_choice(options, answer)
        random.shuffle(options_)
        new_answer = chr(ord('A') + options_.index(answer_text))

        prompt += f'Choices:\n{self.options_to_text(options_)}'
        prompt += 'Answer:'
        
        if not is_inference:
            prompt += f' ({new_answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class TwoChoices(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'

        answer_text = options[ord(answer) - ord('A')]
        options__, answer = eliminate_choice(options, answer)
        options_, answer = eliminate_choice(options__, answer)

        random.shuffle(options_)
        new_answer = chr(ord('A') + options_.index(answer_text))

        prompt += f'Choices:\n{self.options_to_text(options_)}'
        prompt += 'Answer:'

        if not is_inference:
            prompt += f' ({new_answer})'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

# ========================================== Individual Choice Prompts ==========================================

class ChoiceA(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n(A) {options[0]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "A" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

class ChoiceB(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n(B) {options[1]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "B" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

class ChoiceC(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n(C) {options[2]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "C" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

class ChoiceD(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Choices:\n(D) {options[3]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "D" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nChoices:'

# ========================================== Individual Choice Prompts w/ Question ==========================================

class ChoiceAQuestion(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += f'Choices:\n(A) {options[0]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "A" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class ChoiceBQuestion(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += f'Choices:\n(B) {options[1]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "B" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class ChoiceCQuestion(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += f'Choices:\n(C) {options[2]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "C" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'

class ChoiceDQuestion(MultipleChoicePrompt):

    def create_fewshot_example(self, question, options, is_inference, answer):
        prompt = ''
        prompt += f'Question: {question}\n'
        prompt += f'Choices:\n(D) {options[3]}\n'
        prompt += 'Answer:'
        if not is_inference:
            prompt += f' {"True" if answer == "D" else "False"}'
        return prompt

    def define_stop_token(self):
        return '\nQuestion:'