import csv
import json

from commongen_validation_test_set_generation.language_models import get_language_models


def combine_generated_sentences():
    max_concept_set_ids = {
        "validation": 992,
        "test": 1496
    }

    read_generations = {}
    for language_model in get_language_models():
        for subset_name in ["validation", "test"]:
            file_name = f"{language_model}-commongen-{subset_name}-set-generation.json"
            with open(f"generated_sentences/{file_name}", "r") as handle:
                read_generation = json.load(handle)
                read_generations[f"{language_model}-{subset_name}-set"] = read_generation

    csv_header = [
        "set_name",
        "concept_set_id",
        "concepts",
        "bloomz_1b1", "bloomz_1b7", "bloomz_3b", "bloomz_560m", "flan_t5_xl", "flan_t5_large", "t0_3b",
        "tk_instruct_3b_def", "mt0_large", "mt0_base", "mt0_small", "mt0_xl"]
    csv_rows = [csv_header]
    for subset_name in ["validation", "test"]:
        for concept_set_id in range(max_concept_set_ids[subset_name]):
            concepts = read_generations[f"{csv_header[3]}-{subset_name}-set"][str(concept_set_id)]["concepts"]
            cur_csv_row = [subset_name, concept_set_id, concepts]
            for model_name_idx in range(3, len(csv_header)):
                cur_model_name = csv_header[model_name_idx]
                cur_model_generations = read_generations[f"{cur_model_name}-{subset_name}-set"]
                cur_generation = cur_model_generations[str(concept_set_id)]["sentences"]
                cur_csv_row.append(cur_generation)
            csv_rows.append(cur_csv_row)
        with open('generated_sentences/combined-generations.csv', 'w') as file:
            writer = csv.writer(file)
            writer.writerows(csv_rows)


if __name__ == '__main__':
    combine_generated_sentences()
