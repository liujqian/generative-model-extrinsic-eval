import json

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
    with open("generated_sentences/selected_commongen_tasks.json", 'r') as selected_commongen_file:
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

if __name__ == '__main__':
    chatgpt_generate_selected()