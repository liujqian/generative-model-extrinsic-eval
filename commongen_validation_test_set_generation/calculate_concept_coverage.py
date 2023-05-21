import json

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


if __name__ == '__main__':
    for language_model in get_language_models():
        for subset_name in ["validation", "test"]:
            calculate_concept_coverage_and_update_json_files(language_model, subset_name)
