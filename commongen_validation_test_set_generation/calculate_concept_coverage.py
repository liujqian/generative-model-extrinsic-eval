import json

import pandas
import spacy

from commongen_validation_test_set_generation.language_models import get_language_models

nlp = spacy.load("en_core_web_lg")


def lemmatize_sentence(sentence: str) -> set:
    lemmatized_words = set()
    doc = nlp(sentence)
    for word in doc:
        lemmatized_word = word.lemma_
        lemmatized_words.add(lemmatized_word)
    return lemmatized_words


def calculate_concept_coverage_and_update_json_files(model_name: str, subset_name: str):
    json_file_name = f"{model_name}-commongen-{subset_name}-set-generation.json"
    with open(f"generated_sentences/{json_file_name}", "r") as handle:
        generation_dict = json.load(handle)
    for concept_set_id in generation_dict:
        cur_concept_set = generation_dict[concept_set_id]
        sentence = cur_concept_set["sentences"]
        concepts = cur_concept_set["concepts"]
        lemmatized_sentence_set = lemmatize_sentence(sentence)
        found_concept_cnt = 0
        for concept in concepts:
            if concept in lemmatized_sentence_set:
                found_concept_cnt += 1
        coverage = found_concept_cnt / len(concepts)
        cur_concept_set["coverage"] = coverage
    with open(f"generated_sentences/generated_sentences_with_coverage/{json_file_name}", "w") as handle:
        json.dump(generation_dict, handle)


def calculate_batch_average_coverage(batch_input_csv_path: str, output_path: str):
    batch = pandas.read_csv(batch_input_csv_path).to_dict(orient="records")
    qids = {(question["set_name"], question["concept_set_id"]) for question in batch}
    generation_coverage = {}
    for lm_name in get_language_models():
        generation_coverage[lm_name] = {}
        for set_name in ["validation", "test"]:
            handle = open(
                f"generated_sentences/generated_sentences_with_coverage/{lm_name}-commongen-{set_name}-set-generation.json"
            )
            generation_coverage[lm_name][set_name] = json.load(handle)
            handle.close()

    lm_coverage = {lm_name: [] for lm_name in get_language_models()}
    for set_name, csid in qids:
        for lm_name in get_language_models():
            lm_coverage[lm_name].append(generation_coverage[lm_name][set_name][str(csid)]["coverage"])
    for lm_name in get_language_models():
        lm_coverage[lm_name] = sum(lm_coverage[lm_name]) / len(lm_coverage[lm_name])
    with open(output_path + "/language_model_coverages.json", "w") as handle:
        json.dump(lm_coverage, handle)


if __name__ == '__main__':
    calculate_batch_average_coverage("generated_sentences/combined_generations_first_100.csv",
                                     "mturk_batches/sandbox_env_internal_first_100_questions")
