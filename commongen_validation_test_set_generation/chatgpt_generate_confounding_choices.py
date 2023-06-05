import csv
import time

import openai

from experiments.utils import log_progress


def read_openai_api_key() -> str:
    with open("../OPENAI_API_KEY.txt") as handle:
        api_key = handle.readline()
        if len(api_key) == 0:
            raise ValueError(
                "Found empty OpenAI API key. "
                "Please Ensure that OPENAI_API_KEY.txt containing your API key is in the root folder of the repository."
            )
    return api_key


confounding_type_dict = {
    "complex": "Create a complex sentence with clauses using the following words: ",
    "simple": "Create a very simple and short using the following words: ",
    "nonsense": "Create a sentence that does not make sense at all and is completely illogical with the following "
                "words: ",
    "grammar_error": "Create a sentence with very bad grammar and multiple grammar errors using the following "
                     "words: "
}


def make_request(prompt: str) -> str:
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant who can create sentences using sets of words."},
                {"role": "user", "content": prompt},
            ]
        )
        return completion["choices"][0]["message"]["content"].replace("\n", "")
    except:
        return "FAILED"


def create_confounding_sentence(confounding_type: str, concepts: list) -> str:
    if confounding_type not in confounding_type_dict:
        raise ValueError("The confounding type must be one of 'complex' 'simple' 'nonsense' or 'grammar_error'!")
    prompt_prefix = confounding_type_dict[confounding_type]
    full_prompt = prompt_prefix + ", ".join(concepts) + "."

    result = "FAILED"
    retry_cnt = 0
    while result == "FAILED":
        if retry_cnt == 4:
            raise Exception("Cannot make OPENAI request.")
        result = make_request(full_prompt)
        if result == "FAILED":
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

    with open("generated_sentences/combined-generations.csv") as original:
        heading = next(original)
        csv_obj = csv.reader(original)
        for i, row in enumerate(csv_obj):
            log_progress(i, 500, 5, f"Making a new conbined generation CSV!")
            concepts = row[2].split(", ")
            for confounding_type in confounding_types:
                confound = create_confounding_sentence(confounding_type, concepts)
                row.append(confound)
            new_csv.append(row)
    with open("generated_sentences/combined_generations_with_confound.csv", "w", newline='', encoding='utf-8') as new:
        writer = csv.writer(new)
        for csv_row in new_csv:
            writer.writerow(csv_row)
