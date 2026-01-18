from datasets import load_dataset, Dataset
import random
from functools import partial

import dataclasses
from typing import Dict, Optional, Union

from lm_eval.tasks.ifeval import instructions_registry
from lm_eval.tasks.bbh.zeroshot.utils import MultiChoiceRegexFilter, MapRegexFilter, NumberParseRegexFilter
import ast
# MATH evaluation imports
import re
import signal

import sympy
from math_verify import LatexExtractionConfig, parse, verify
from sympy.parsing.latex import parse_latex


class CrossDiffMultiChoiceRegexFilter:
    def __init__(self, **kwargs):
        self.bbh_filter = MultiChoiceRegexFilter(**kwargs)

    def apply(self, resps, docs):
        bbh_docs = [{"input": doc["question"]} for doc in docs]
        return self.bbh_filter.apply(resps, bbh_docs)


class CrossDiffMapRegexFilter:
    def __init__(self, **kwargs):
        self.bbh_filter = MapRegexFilter(**kwargs)

    def apply(self, resps, docs):
        bbh_docs = [{"input": doc["question"]} for doc in docs]
        return self.bbh_filter.apply(resps, bbh_docs)


class CrossDiffNumberParseRegexFilter:
    def __init__(self, **kwargs):
        self.bbh_filter = NumberParseRegexFilter(**kwargs)

    def apply(self, resps, docs):
        bbh_docs = [{"input": doc["question"]} for doc in docs]
        return self.bbh_filter.apply(resps, bbh_docs)


choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
]



@dataclasses.dataclass
class InputExample:
    key: int
    instruction_id_list: list[str]
    prompt: str
    kwargs: list[Dict[str, Optional[Union[str, int]]]]


@dataclasses.dataclass
class OutputExample:
    instruction_id_list: list[str]
    prompt: str
    response: str
    follow_all_instructions: bool
    follow_instruction_list: list[bool]


def test_instruction_following_strict(
    inp,
    response,
):
    """Tests response to see if instructions are followed."""
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def test_instruction_following_loose(
    inp,
    response,
):
    """Tests response for an upper bound for following instructions."""
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    instruction_list = inp.instruction_id_list
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs = {k: v for k, v in inp.kwargs[index].items() if v}
        instruction.build_description(**kwargs)
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp.prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        is_following_list.append(is_following)

    return OutputExample(
        instruction_id_list=inp.instruction_id_list,
        prompt=inp.prompt,
        response=response,
        follow_all_instructions=all(is_following_list),
        follow_instruction_list=is_following_list,
    )


def process_results(doc, results):
    inp = InputExample(
        key=doc["key"],
        instruction_id_list=doc["instruction_id_list"],
        prompt=doc["prompt"],
        kwargs=doc["kwargs"],
    )
    response = results[0]

    out_strict = test_instruction_following_strict(inp, response)
    out_loose = test_instruction_following_loose(inp, response)

    return {
        "prompt_level_strict_acc": out_strict.follow_all_instructions,
        "inst_level_strict_acc": out_strict.follow_instruction_list,
        "prompt_level_loose_acc": out_loose.follow_all_instructions,
        "inst_level_loose_acc": out_loose.follow_instruction_list,
    }


def agg_inst_level_acc(items):
    flat_items = [item for sublist in items for item in sublist]
    inst_level_acc = sum(flat_items) / len(flat_items)
    return inst_level_acc

def format_arc_doc_to_text(example, include_in_context=False, bin=None, num_examples=20):
    question = example['question']
    choices = example['choices']

    choices_prompt = '\n'.join(f'{label}) {choice}' for label, choice in zip(
        choices['label'],
        choices['text'],
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "arc")['train']

            # Correctly filter examples based on bin ranges
            # bin=0: 0.0 <= 1pl_quantile < 0.1
            # bin=1: 0.1 <= 1pl_quantile < 0.2, etc.
            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['1pl_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['1pl_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_question = ex['question']
                    ex_choices = ex['choices']

                    ex_choices_prompt = '\n'.join(f'{label}) {choice}' for label, choice in zip(
                        ex_choices['label'],
                        ex_choices['text'],
                    ))

                    answer = ex['answerKey']
                    examples_text.append(f"Question: {ex_question}\n{ex_choices_prompt}\nAnswer: {answer}\n")

                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt

def format_mmlu_pro_example(example, including_answer=True, include_in_context=False, bin=None, num_examples=20):
    question = example['question']
    choices = example['options']
    letter_options = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

    choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
        choices,
        letter_options[:len(choices)]
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "mmlu_pro")['train']

            # Filter examples based on bin ranges
            # bin=0: 0.0 <= 1pl_quantile < 0.1
            # bin=1: 0.1 <= 1pl_quantile < 0.2, etc.
            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['1pl_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['1pl_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_question = ex['question']
                    ex_choices = ex['options']

                    ex_choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
                        ex_choices,
                        letter_options[:len(ex_choices)]
                    ))

                    answer = ex.get('answer', '')
                    examples_text.append(f"Question: {ex_question}\n{ex_choices_prompt}\nAnswer: {answer}\n")

                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt

def format_gsm8k_example(example, include_in_context=False, bin=None, num_examples=20):
    question = example['question']
    user_prompt = f"Q: {question}\nA:\n"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "gsm8k")['train']

            # Filter examples based on bin ranges
            # bin=0: 0.0 <= 1pl_quantile < 0.1
            # bin=1: 0.1 <= 1pl_quantile < 0.2, etc.
            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['1pl_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['1pl_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_question = ex['question']
                    ex_answer = ex['answer']
                    examples_text.append(f"Q: {ex_question}\nA: {ex_answer}\n")

                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt

def format_gpqa_extended_example(example, include_in_context=False, bin=None, num_examples=20):
    question = example['question']
    choices = example['options']
    letter_options = 'ABCD'

    choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
        choices,
        letter_options[:len(choices)]
    ))

    user_prompt = f"Question: {question}\n{choices_prompt}\nAnswer:\n"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "gpqa_extended")['train']

            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['rating_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['rating_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_question = ex['question']
                    ex_choices = ex['options']

                    ex_choices_prompt = '\n'.join(f'{label}) {choice}' for choice, label in zip(
                        ex_choices,
                        letter_options[:len(ex_choices)]
                    ))

                    answer = ex.get('answer', '')
                    examples_text.append(f"Question: {ex_question}\n{ex_choices_prompt}\nAnswer: {answer}\n")

                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt

def filter_bbh_boolean_docs(docs):
    boolean_subsets = ["boolean_expressions"]

    filtered_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        if any(doc_id.startswith(f'bbh_{subset}') for subset in boolean_subsets):
            filtered_docs.append(doc)

    if filtered_docs:
        return Dataset.from_list(filtered_docs)
    else:
        return Dataset.from_dict({key: [] for key in docs[0].keys() if len(docs) > 0})

def filter_bbh_multichoice_docs(docs):
    """Filter BBH docs for multiple choice tasks (A), (B), etc."""
    multichoice_subsets = [
        "date_understanding",
        "disambiguation_qa",
        "geometric_shapes",
        "hyperbaton",
        "logical_deduction_five_objects",
        "logical_deduction_seven_objects",
        "logical_deduction_three_objects",
        "movie_recommendation",
        "reasoning_about_colored_objects",
        "ruin_names",
        "salient_translation_error_detection",
        "temporal_sequences",
        "tracking_shuffled_objects_five_objects",
        "tracking_shuffled_objects_seven_objects",
        "tracking_shuffled_objects_three_objects"
    ]

    filtered_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        if any(doc_id.startswith(f'bbh_{subset}') for subset in multichoice_subsets):
            filtered_docs.append(doc)

    if filtered_docs:
        return Dataset.from_list(filtered_docs)
    else:
        return Dataset.from_dict({key: [] for key in docs[0].keys() if len(docs) > 0})

def filter_bbh_yesno_docs(docs):
    yesno_subsets = [
        "navigate",
        "sports_understanding",
        "web_of_lies"
    ]

    filtered_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        if any(doc_id.startswith(f'bbh_{subset}') for subset in yesno_subsets):
            filtered_docs.append(doc)

    if filtered_docs:
        return Dataset.from_list(filtered_docs)
    else:
        return Dataset.from_dict({key: [] for key in docs[0].keys() if len(docs) > 0})

def filter_bbh_numeric_docs(docs):
    numeric_subsets = ["object_counting"]

    filtered_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        if any(doc_id.startswith(f'bbh_{subset}') for subset in numeric_subsets):
            filtered_docs.append(doc)

    if filtered_docs:
        return Dataset.from_list(filtered_docs)
    else:
        return Dataset.from_dict({key: [] for key in docs[0].keys() if len(docs) > 0})

def filter_bbh_valid_invalid_docs(docs):
    valid_invalid_subsets = ["formal_fallacies"]

    filtered_docs = []
    for doc in docs:
        doc_id = doc['doc_id']
        if any(doc_id.startswith(f'bbh_{subset}') for subset in valid_invalid_subsets):
            filtered_docs.append(doc)

    if filtered_docs:
        return Dataset.from_list(filtered_docs)
    else:
        return Dataset.from_dict({key: [] for key in docs[0].keys() if len(docs) > 0})

def format_musr_example(example, include_in_context=False, bin=None, num_examples=20):
    narrative = example['narrative']
    question = example['question']
    choices_list = example['options']

    if isinstance(choices_list, str):
      choices_list = ast.literal_eval(choices_list)

    choices = ""
    for i, choice in enumerate(choices_list):
        choices += f"{i + 1} - {choice}\n"
    user_prompt = f"{narrative}\n\n{question}\n\n{choices}Answer:"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "musr")['train']

            # Filter examples based on bin ranges
            # bin=0: 0.0 <= 1pl_quantile < 0.1
            # bin=1: 0.1 <= 1pl_quantile < 0.2, etc.
            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['1pl_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['1pl_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_narrative = ex['narrative']
                    ex_question = ex['question']
                    ex_choices_list = ex['options']
                    ex_choices = ""
                    for i, choice in enumerate(ex_choices_list):
                        ex_choices += f"{i + 1} - {choice}\n"

                    answer = ex.get('answer', '')
                    examples_text.append(f"{ex_narrative}\n\n{ex_question}\n\n{ex_choices}Answer: {answer}\n")

                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt


def process_math_results(doc: dict, results: list) -> dict:
    """
    Process MATH results using the exact same method as leaderboard MATH.
    """
    if not results:
        return {"exact_match": 0}

    candidates = results[0]

    # Use the exact same method as leaderboard MATH
    try:
        parsed_answer = parse(f'${doc["answer"]}$', extraction_config=[LatexExtractionConfig()])
        if verify(parsed_answer, parse(candidates)) or verify(parsed_answer, parse(f'${candidates}$')):
            retval = 1
        else:
            retval = 0
    except Exception:
        retval = 0

    return {"exact_match": retval}


def format_math_example(example, include_in_context=False, bin=None, num_examples=20):
    question = example['question']
    user_prompt = f"Problem:\n{question}\n\nSolution:"

    if include_in_context and bin is not None:
        try:
            dataset = load_dataset("Yeganeh/Cross-Difficulty", "math")['train']

            # Filter examples based on bin ranges
            # bin=0: 0.0 <= 1pl_quantile < 0.1
            # bin=1: 0.1 <= 1pl_quantile < 0.2, etc.
            lower_bound = bin / 10.0
            upper_bound = (bin + 1) / 10.0

            filtered_examples = [ex for ex in dataset if lower_bound <= ex['1pl_quantile'] < upper_bound]

            sorted_examples = sorted(filtered_examples, key=lambda x: x['1pl_quantile'])
            selected_examples = sorted_examples[:num_examples]

            if selected_examples:
                examples_text = []
                for ex in selected_examples:
                    ex_question = ex['question']
                    ex_answer = ex.get('answer', '')
                    examples_text.append(f"Problem:\n{ex_question}\n\nSolution:\n{ex_answer}\n")

                # Add in-context examples before the main question
                context = "\n".join(examples_text)
                return f"{context}\n{user_prompt}"
        except Exception as e:
            print(f"Error loading or processing dataset: {e}")

    return user_prompt


gsm8k_doc_to_text = partial(format_gsm8k_example, include_in_context=False, bin=3, num_examples=20)
arc_doc_to_text = partial(format_arc_doc_to_text, include_in_context=False, bin=3, num_examples=20)
mmlu_pro_doc_to_text = partial(format_mmlu_pro_example, include_in_context=False, bin=0, num_examples=20)
gpqa_extended_doc_to_text = partial(format_gpqa_extended_example, include_in_context=False, bin=0, num_examples=20)
musr_doc_to_text = partial(format_musr_example, include_in_context=False, bin=3, num_examples=20)
math_doc_to_text = partial(format_math_example, include_in_context=False, bin=3, num_examples=20)
