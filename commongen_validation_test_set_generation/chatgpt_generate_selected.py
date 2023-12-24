import json

import datasets

from experiments.chatgpt_generation import chatgpt_generation
from experiments.utils import log_progress


def chatgpt_generate_selected():
    model_name = "chatgpt"
    validation = open(
        f'generated_sentences/{model_name}-commongen-validation-set-generation.json',
        'w'
    )
    validation_json = {}
    test = open(
        f'generated_sentences/{model_name}-commongen-test-set-generation.json',
        'w'
    )
    test_json = {}
    with open("generated_sentences/old_selected_commongen_tasks_based_on_generation_volcab.json", 'r') as selected_commongen_file:
        selected_commongens = json.load(selected_commongen_file)
        for i, item_id in enumerate(selected_commongens):
            log_progress(i, len(selected_commongens), 10,
                         "Use chatgpt to make commongen generations for selected items.")
            splits = item_id.split("-")
            subset_name = splits[0]
            concept_set_id = splits[1]
            concepts = selected_commongens[item_id].split(", ")
            target_json_file = validation_json if subset_name == "validation" else test_json
            chatgpt_results = chatgpt_generation(concepts, 1)
            target_json_file[str(concept_set_id)] = {
                "sentences": chatgpt_results[0],
                "concepts": concepts,
                "targets": ""
            }
    json.dump(test_json, test)
    test.close()
    json.dump(validation_json, validation)
    validation.close()


def chatgpt_make_up():
    model_name = "chatgpt"
    validation = open(
        f'generated_sentences/{model_name}-commongen-validation-set-generation.json',
        'r'
    )
    validation_json = json.load(validation)
    validation.close()
    test = open(
        f'generated_sentences/{model_name}-commongen-test-set-generation.json',
        'r'
    )
    test_json = json.load(test)
    test.close()
    commongen_sets = {"validation": validation_json, "test": test_json}
    cg = datasets.load_dataset('common_gen')

    new_selected = open("generated_sentences/all_commongen_tasks_validation.json", "r")
    new_selected_json = json.load(new_selected)
    new_selected.close()

    for concept_set in new_selected_json:
        question_id = concept_set["id"]
        set_name = question_id.split("-")[0]
        concept_set_id = question_id.split("-")[1]
        target_set_dict = commongen_sets[set_name]
        if concept_set_id not in target_set_dict:
            concepts = concept_set["concepts"]
            concepts_list = concepts.split(", ")
            print("Seeing a new question ID: " + question_id + ". Trying to make ChatGPT generation for the quesiton.")
            chatgpt_results = chatgpt_generation(concepts_list, 1)
            target_set_dict[str(concept_set_id)] = {
                "sentences": chatgpt_results[0],
                "concepts": concepts,
                "targets": ""
            }
        else:
            print("An old question that is already generated is seen: " + question_id)

    with open(
            f'generated_sentences/{model_name}-commongen-validation-set-generation.json',
            'w'
    ) as handle:
        json.dump(validation_json, handle)
    with open(
            f'generated_sentences/{model_name}-commongen-test-set-generation.json',
            'w'
    ) as handle:
        json.dump(test_json, handle)


if __name__ == '__main__':
    chatgpt_make_up()
