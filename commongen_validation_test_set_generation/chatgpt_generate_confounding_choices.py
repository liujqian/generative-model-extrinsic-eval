import csv
import time
from typing import Any

import openai

from experiments.chatgpt_generation import make_request_get_all
from experiments.utils import log_progress





confounding_type_dict = {
    "complex": "Create a complex sentence with clauses using the following words: ",
    "simple": "Create a very simple and short using the following words: ",
    "nonsense": "Create a sentence that does not make sense at all and is completely illogical with the following "
                "words: ",
    "grammar_error": "Create a sentence with very bad grammar and multiple grammar errors using the following "
                     "words: "
}


def make_request(prompt: str) -> str | None:
    res = make_request_get_all(prompt, 1)
    if res is None:
        return None
    else:
        return res[0]


def create_confounding_sentence(confounding_type: str, concepts: list) -> str:
    if confounding_type not in confounding_type_dict:
        raise ValueError("The confounding type must be one of 'complex' 'simple' 'nonsense' or 'grammar_error'!")
    prompt_prefix = confounding_type_dict[confounding_type]
    full_prompt = prompt_prefix + ", ".join(concepts) + "."

    result = None
    retry_cnt = 0
    while result is None:
        if retry_cnt == 4:
            raise Exception("Cannot make OPENAI request.")
        result = make_request(full_prompt)
        if result is None:
            retry_cnt += 1
            print("Failed to make OPENAI request once. Sleep for five seconds and try again.")
            time.sleep(5 * retry_cnt)
    sentence = result
    return sentence


if __name__ == '__main__':
    openai.api_key = read_openai_api_key()
    confounding_types = [
        "complex",
        "simple",
        "nonsense",
        "grammar_error",
    ]

    new_csv = [
        [
            "set_name",
            "concept_set_id",
            "concepts",
            "bloomz_1b1", "bloomz_1b7", "bloomz_3b", "bloomz_560m", "flan_t5_xl", "flan_t5_large", "t0_3b",
            "tk_instruct_3b_def", "mt0_large", "mt0_base", "mt0_small", "mt0_xl"
        ] + confounding_types
    ]

    with open("generated_sentences/old_combined_generations_based_on_generation_volcab.csv") as original:
        heading = next(original)
        csv_obj = csv.reader(original)
        for i, row in enumerate(csv_obj):
            log_progress(i, 500, 5, f"Making a new conbined generation CSV!")
            concepts = row[2].split(", ")
            for confounding_type in confounding_types:
                confound = create_confounding_sentence(confounding_type, concepts)
                row.append(confound)
            new_csv.append(row)
    with open("generated_sentences/old_combined_generations_with_confound_based_on_generation_volcab.csv", "w", newline='', encoding='utf-8') as new:
        writer = csv.writer(new)
        for csv_row in new_csv:
            writer.writerow(csv_row)
