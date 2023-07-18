import csv
import json
import time

from experiments.chatgpt_generation import make_request_get_all

confounding_type_dict = {
    "complex": "Create a complex sentence with clauses using the following words: ",
    "simple": "Create a very simple and short using the following words: ",
    "nonsense": "Create a sentence that does not make sense at all and is completely illogical with the following "
                "words: ",
    "grammar_error": "Create a sentence with very bad grammar and multiple grammar errors using the following "
                     "words: "
}
confounding_types = [
    "complex",
    "simple",
    "nonsense",
    "grammar_error",
]


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


def chatgpt_make_up_confounding_generations():
    with open("generated_sentences/new_selected_commongen_tasks_greedy_diversity.json", "r") as handle:
        newly_selected_questions = json.load(handle)
    confounding_generation_jsons = {}
    for confounding_type in confounding_types:
        confounding_generation_jsons[confounding_type] = {}
        for set_name in ["validation", "test"]:
            with open(
                    f"generated_sentences/chatgpt-confounding-{confounding_type}-{set_name}-set-generation.json",
                    "r"
            ) as handle:
                confounding_generation_jsons[confounding_type][set_name] = json.load(handle)

    for confounding_type in confounding_types:
        print(f"Checking the {confounding_type} confounding type for make up now.")
        for newly_selected_question in newly_selected_questions:
            newly_selected_question_id = newly_selected_question["id"]
            newly_selected_question_set_name = newly_selected_question_id.split("-")[0]
            newly_selected_question_csid = newly_selected_question_id.split("-")[1]
            if newly_selected_question_csid not in \
                    confounding_generation_jsons[confounding_type][newly_selected_question_set_name]:
                print(f"For the {confounding_type} type, {newly_selected_question_id} is missing. Making up now.")
                concepts = newly_selected_question["concepts"].split(", ")
                new_confounding_sentence = create_confounding_sentence(confounding_type, concepts)
                confounding_generation_jsons[confounding_type][newly_selected_question_set_name][
                    newly_selected_question_csid] = {
                    "sentences": new_confounding_sentence,
                    "concepts": concepts,
                    "targets": "",
                }

    for confounding_type in confounding_types:
        for set_name in ["validation", "test"]:
            with open(
                    f"generated_sentences/chatgpt-confounding-{confounding_type}-{set_name}-set-generation.json",
                    "w"
            ) as handle:
                json.dump(confounding_generation_jsons[confounding_type][set_name], handle)


def extract_generated_confounding_sentences_from_combined_old_file():
    new_csv = [
        [
            "set_name",
            "concept_set_id",
            "concepts",
            "bloomz_1b1", "bloomz_1b7", "bloomz_3b", "bloomz_560m", "flan_t5_xl", "flan_t5_large", "t0_3b",
            "tk_instruct_3b_def", "mt0_large", "mt0_base", "mt0_small", "mt0_xl"
        ] + confounding_types + ["chatgpt"]
    ]
    all_dict = {}
    for confounding_type in confounding_types:
        all_dict[confounding_type] = {"validation": {}, "test": {}}
    all_idx = {}
    for confounding_type in confounding_types:
        all_idx[confounding_type] = new_csv[0].index(confounding_type)

    with open("generated_sentences/old_combined_generations_with_confound_based_on_generation_volcab.csv") as original:
        heading = next(original)
        csv_obj = csv.reader(original)
        for i, row in enumerate(csv_obj):
            set_name = row[0]
            concept_set_id = row[1]
            for confounding_type in confounding_types:
                all_dict[confounding_type][set_name][concept_set_id] = {
                    "sentences": row[all_idx[confounding_type]],
                    "concepts": row[2].split(", "),
                    "targets": ""
                }

    for confounding_type in all_dict:
        for set_name in all_dict[confounding_type]:
            with open(
                    f"generated_sentences/chatgpt-confounding-{confounding_type}-{set_name}-set-generation.json",
                    "w"
            ) as handle:
                json.dump(all_dict[confounding_type][set_name], handle)


if __name__ == '__main__':
    chatgpt_make_up_confounding_generations()
