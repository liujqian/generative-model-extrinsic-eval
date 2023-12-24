import json
import time

import datasets
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


def chatgpt_generation(concepts: list, n: int) -> list:
    prompt = f"Create a sentence with the following words:{', '.join(concepts)}"
    results = make_request_get_all(prompt, n)
    failure_cnt = 0
    while results is None:
        failure_cnt += 1
        if failure_cnt == 4:
            raise Exception("OpenAI rejected the generation request for 4 times!")
        print(
            f"OpenAI rejected the generation request for {failure_cnt} times. Sleeping for {failure_cnt * 5} seconds."
        )
        time.sleep(failure_cnt * 5)
        results = make_request_get_all(prompt, n)
    return results


def make_request_get_all(prompt: str, n_generation: int) -> list | None:
    if openai.api_key is None:
        openai.api_key = read_openai_api_key()
        openai.organization = "org-Z6lli0WO1awNKqRJjTToCxj4"
    try:
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "You are a helpful assistant who can create sentences using sets of words."},
                {"role": "user", "content": prompt},
            ],
            n=n_generation
        )
        results = []
        for choice in completion.choices:
            results.append(choice.message.content.replace("\n", ""))
        return results
    except Exception as e:
        print(e)
        return None


# Generated for 50 commonsenseqa questions and cost 0.05 USD
def generate_without_choice_content_words_chatgpt(
        set_name: str,
        examples: datasets.Dataset = None
):
    if examples is None:
        examples = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")[set_name]
    results = {}
    total_item_cnt = len(examples['id'])
    for i in range(total_item_cnt):
        if i % 10 == 0:
            log_progress(
                i,
                total_item_cnt,
                10,
                f"Generating inferences for the {set_name} set. Without choice content words."
            )
        question_content_words = examples["question_content_words"][i]
        generations = chatgpt_generation(question_content_words, 20)
        results[i] = {"id": examples["id"][i], "sentences": generations}
    return results


def generate_with_choice_content_words_chatgpt(
        set_name: str,
        examples: datasets.Dataset = None
):
    if examples is None:
        examples = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")[set_name]
    results = {}
    total_item_cnt = len(examples['id'])
    for i in range(total_item_cnt):
        if i % 10 == 0:
            log_progress(
                i,
                total_item_cnt,
                10,
                f"Generating inferences for the {set_name} set. With choice content words."
            )
        generations_cur_question = []
        question_content_words = examples["question_content_words"][i]
        for choice_idx in range(0, 5):
            choice_content_words = examples[f"choice_{choice_idx}_content_words"][i]
            all_content_words = choice_content_words + question_content_words
            cur_choice_generations = chatgpt_generation(all_content_words, 4)
            generations_cur_question.extend(cur_choice_generations)
        results[i] = {"id": examples["id"][i], "sentences": generations_cur_question}
    return results


if __name__ == '__main__':
    cur_generation_type = "without"
    generation_types = {
        "with": (generate_with_choice_content_words_chatgpt, "WITH"),
        "without": (generate_without_choice_content_words_chatgpt, "WITHOUT")
    }

    for subset_name in [
        "train",
        # "validation"
    ]:
        generations = generation_types[cur_generation_type][0](subset_name)
        with open(
                f'generated_sentences/chatgpt-{subset_name}-{generation_types[cur_generation_type][1]}-choicewords'
                f'-noquestionwordlimit.json',
                'w'
        ) as handle:
            json.dump(generations, handle)
        print(f"Finished generating and dumping the chatgpt inferences for the commonsenseqa {subset_name} set.")
