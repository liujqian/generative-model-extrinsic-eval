import csv
import json

import spacy

from commongen_validation_test_set_generation.language_models import get_language_models
from experiments.utils import log_progress


def combine_generated_sentences(num_items: int = 500):
    nlp = spacy.load("en_core_web_lg")
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
    all_row_count_tuples = []
    for subset_name in ["test", "validation", ]:
        for concept_set_id in range(max_concept_set_ids[subset_name]):
            log_progress(concept_set_id, max_concept_set_ids[subset_name], 20, f"Processing {subset_name} set.")
            concepts = read_generations[f"{csv_header[3]}-{subset_name}-set"][str(concept_set_id)]["concepts"]
            cur_csv_row = [subset_name, concept_set_id, f"{', '.join(concepts)}"]
            row_count_tuple = (subset_name, concept_set_id, set())
            for model_name_idx in range(3, len(csv_header)):
                cur_model_name = csv_header[model_name_idx]
                cur_model_generations = read_generations[f"{cur_model_name}-{subset_name}-set"]
                cur_generation = cur_model_generations[str(concept_set_id)]["sentences"]
                for word in [w.lemma_ for w in nlp(cur_generation)]:
                    row_count_tuple[2].add(word)
                cur_csv_row.append(f'{cur_generation}')
            all_row_count_tuples.append(row_count_tuple)
            csv_rows.append(cur_csv_row)
        all_row_count_tuples.sort(key=lambda t: len(t[2]), reverse=True)
        selected_row_tuples = all_row_count_tuples[:num_items]
        selected_concept_sets = {f"{t[0]}-{t[1]}" for t in selected_row_tuples}
        with open('generated_sentences/old_combined_generations_based_on_generation_volcab.csv', 'w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(csv_header)
            for csv_row in csv_rows:
                if f"{csv_row[0]}-{csv_row[1]}" in selected_concept_sets:
                    writer.writerow(csv_row)


if __name__ == '__main__':
    combine_generated_sentences()
