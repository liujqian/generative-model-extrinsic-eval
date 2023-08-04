import json
import math

import spacy
from datasets import load_dataset

from commongen_validation_test_set_generation.language_models import get_language_models
from experiments.utils import log_progress

#######################################################  Utility Helper Functions.  #######################################################
nlp = spacy.load("en_core_web_lg")
language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "liujqian/gpt2-large-finetuned-commongen",
    "gpt2-m": "liujqian/gpt2-medium-finetuned-commongen",
    "gpt2": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-finetuned-commongen",
    # "t5": "mrm8488/t5-base-finetuned-common_gen",
    # "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

file_postfix_without_choice = {
    "train": "-train-WITHOUT-choicewords-noquestionwordlimit.json",
    "validation": "-validation-WITHOUT-choicewords-noquestionwordlimit.json"
}

file_postfix_with_choice = {
    "train": "-train-WITH-choicewords-noquestionwordlimit.json",
    "validation": "-validation-WITH-choicewords-noquestionwordlimit.json"
}


# This function is copied directly from this post: https://stackoverflow.com/a/26726185
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def lemmatize(s: str):
    doc = nlp(s)
    return " ".join([x.lemma_ for x in doc])


substring_check_mode = "use_in"
print("The substring check mode used in this run is: " + substring_check_mode)


def count_occurences(sentences: list[str], targets: list[str]) -> int:
    cnt = 0
    for target in targets:
        lemmatized_target = lemmatize(target).lower()
        if lemmatized_target == "":
            continue
        for sentence in sentences:
            truncated = sentence
            lemmatized_sentence = lemmatize(truncated).lower()
            if substring_check_mode == "use_in":
                cnt += lemmatized_target in lemmatized_sentence
            else:
                cnt += lemmatized_sentence.count(lemmatized_target)
    return cnt


#######################################################  For With Choice Analysis.  #######################################################
def get_current_question_choice_status_with_choice_analysis(question: dict, generations: dict) -> dict:
    choices_stats = {
        "inclusion_count": [],
        "avg_sequences_scores": []
    }

    has_scores = "sequences_scores" in generations

    generated_sentences = generations["sentences"]
    scores = generations["sequences_scores"] if has_scores else [0] * 20
    for choice_idx in range(0, 5):
        choice_content_words = question[f"choice_{choice_idx}_content_words"]
        choice_generations = generated_sentences[choice_idx * 4:choice_idx * 4 + 4]
        choice_scores = scores[choice_idx * 4:choice_idx * 4 + 4]
        inclusion_count = count_occurences(choice_generations, choice_content_words)
        choices_stats["inclusion_count"].append(inclusion_count)
        choices_stats["avg_sequences_scores"].append(sum(choice_scores) / len(choice_scores))
    return choices_stats


def get_correct_choice_idx(question: dict) -> int:
    choice_idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    letter_correct_choice = question["answerKey"]
    idx_correct_choice = choice_idx_map[letter_correct_choice]
    return idx_correct_choice


def check_inclusion_scores_predict_correctness(idx_correct_choice: int, choices_stats: dict) -> (bool, bool):
    highest_inclusion_count = max(choices_stats["inclusion_count"])
    inclusion_predict_correct = choices_stats["inclusion_count"][idx_correct_choice] == highest_inclusion_count

    highest_score = max(choices_stats["avg_sequences_scores"])
    score_predict_correct = math.isclose(choices_stats["avg_sequences_scores"][idx_correct_choice], highest_score)
    return inclusion_predict_correct, score_predict_correct


def update_subset_stats_for_with_choice_analysis(
        stats: dict, choices_stats: dict, idx_correct_choice: int
):
    (inclusion_predict_correct, score_predict_correct) = check_inclusion_scores_predict_correctness(idx_correct_choice, choices_stats)
    stats["correct_prediction_by_inclusion_count"] += inclusion_predict_correct
    stats["correct_prediction_by_sequences_score"] += score_predict_correct
    stats["avg_correct_prediction_inclusion"] = ((stats["avg_correct_prediction_inclusion"] * stats["total_examined"]) +
                                                 choices_stats["inclusion_count"][idx_correct_choice]) / (stats["total_examined"] + 1)

    stats["avg_correct_prediction_sequence_score"] = ((stats["avg_correct_prediction_sequence_score"] * stats["total_examined"]) +
                                                      choices_stats["avg_sequences_scores"][idx_correct_choice]) / (stats["total_examined"] + 1)

    wrong_prediction_inclusion_count = [v for ci, v in enumerate(choices_stats["inclusion_count"]) if ci != idx_correct_choice]
    wrong_sequence_scores = [v for ci, v in enumerate(choices_stats["avg_sequences_scores"]) if ci != idx_correct_choice]

    stats["avg_wrong_prediction_inclusion"] = ((stats["avg_wrong_prediction_inclusion"] * stats["total_examined"]) + sum(
        wrong_prediction_inclusion_count) / len(wrong_prediction_inclusion_count)) / (stats["total_examined"] + 1)
    stats["avg_wrong_prediction_sequence_score"] = ((stats["avg_wrong_prediction_sequence_score"] * stats["total_examined"]) + sum(
        wrong_sequence_scores) / len(wrong_sequence_scores)) / (stats["total_examined"] + 1)

    stats["total_examined"] += 1


def analyze_with_choice_generations(model_name: str):
    dataset = load_dataset("liujqian/commonsenseqa_with_content_words")
    total_questions = len(dataset["train"]) + len(dataset["validation"])
    all_results = {}
    for subset_name in file_postfix_without_choice:
        with open(f"generated_sentences/{model_name}{file_postfix_with_choice[subset_name]}", "r", ) as file:
            model_generations = json.load(file)
        subset = dataset[subset_name]
        stats = {
            "total_examined": 0,
            "correct_prediction_by_inclusion_count": 0,
            "correct_prediction_by_sequences_score": 0,
            "avg_correct_prediction_inclusion": 0,
            "avg_wrong_prediction_inclusion": 0,
            "avg_correct_prediction_sequence_score": 0,
            "avg_wrong_prediction_sequence_score": 0,
        }
        for i in range(len(subset["id"])):
            if i % 200 == 0:
                additional_info = f"Analyzing the results of {model_name} on the {subset_name} set. This is WITH choices' content words."
                log_progress(i, len(subset["id"]), 10, additional_info)
            question_generations = model_generations[str(i)]
            question = subset[i]
            choices_stats = get_current_question_choice_status_with_choice_analysis(question, question_generations)
            idx_correct_choice = get_correct_choice_idx(question)

            update_subset_stats_for_with_choice_analysis(stats, choices_stats, idx_correct_choice)
        all_results[subset_name] = stats

    total_inclusion = sum(
        [all_results[subset_name]["correct_prediction_by_inclusion_count"] for subset_name in all_results])
    total_score = sum(
        [all_results[subset_name]["correct_prediction_by_sequences_score"] for subset_name in all_results])
    all_results["inclusion_accuracy"] = total_inclusion / total_questions
    all_results["score_accuracy"] = total_score / total_questions
    all_results["inclusion_correct_count"] = total_inclusion
    all_results["score_correct_count"] = total_score
    with open(f"analysis-results/{substring_check_mode}_check/{model_name}-analysis-results-WITH-choice.json", "w") as file:
        json.dump(all_results, file)


#######################################################  For WithOUT Choice Analysis.  #######################################################
def get_choice_mention_count(question: dict, question_generations: dict) -> list:
    # choice_content_words_stats = {}
    # for choice_idx in range(0, 5):
    #     choice_content_words = question[f"choice_{choice_idx}_content_words"]
    #     for word in choice_content_words:
    #         if word not in choice_content_words_stats:
    #             choice_content_words_stats[word] = {choice_idx}
    #         else:
    #             choice_content_words_stats[word].add(choice_idx)
    # choice_mention_count = [0, 0, 0, 0, 0]
    # for sentence in question_generations["sentences"]:
    #     lemmatized_sentence = lemmatize(sentence)
    #     for choice_content_word in choice_content_words_stats:
    #         if choice_content_word in lemmatized_sentence:
    #             for choice_idx in choice_content_words_stats[choice_content_word]:
    #                 choice_mention_count[choice_idx] += 1 / len(choice_content_words_stats[choice_content_word])
    # return choice_mention_count
    choice_mention_count = [0, 0, 0, 0, 0]
    for choice_idx in range(0, 5):
        choice_content_words = question[f"choice_{choice_idx}_content_words"]
        generated_sentences = question_generations["sentences"]
        mention_count = count_occurences(sentences=generated_sentences, targets=choice_content_words)
        choice_mention_count[choice_idx] = mention_count
    return choice_mention_count


def check_no_choice_predict_correctness(choice_mention_count: list, idx_correct_choice: int):
    predicted = choice_mention_count[idx_correct_choice] == max(choice_mention_count) and max(choice_mention_count) != 0
    return predicted


def update_subset_stats_for_without_choice_analysis(stats: dict, choice_mention_count: list, idx_correct_choice: int):
    predicted_correct = check_no_choice_predict_correctness(choice_mention_count, idx_correct_choice)
    stats["correct_prediction"] += predicted_correct
    stats["total_correct_choice_word_generated"] += choice_mention_count[idx_correct_choice]
    stats["total_examined"] += 1


def analyze_without_choice_generations(model_name: str):
    dataset = load_dataset("liujqian/commonsenseqa_with_content_words")
    total_questions = len(dataset["train"]) + len(dataset["validation"])
    choice_idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    all_results = {}
    for subset_name in file_postfix_without_choice:
        with open(f"generated_sentences/{model_name}{file_postfix_without_choice[subset_name]}", "r") as file:
            model_generations = json.load(file)
        subset = dataset[subset_name]
        stats = {
            "total_examined": 0,
            "correct_prediction": 0,
            "total_correct_choice_word_generated": 0,
        }
        for i in range(len(subset["id"])):
            if i % 200 == 0:
                info = f"Analyzing the results of {model_name} on the {subset_name} set. This is WITHOUT choices' content words."
                log_progress(i, len(subset["id"]), 10, info)
            question_generations = model_generations[str(i)]
            question = subset[i]
            idx_correct_choice = get_correct_choice_idx(question)
            # key are all choice content words, values are sets indicating which choices appeared in
            choice_mention_count = get_choice_mention_count(question, question_generations)

            update_subset_stats_for_without_choice_analysis(stats, choice_mention_count, idx_correct_choice)

        all_results[subset_name] = stats
    total_correct = sum([all_results[subset_name]["correct_prediction"] for subset_name in all_results])
    all_results["accuracy"] = total_correct / total_questions
    all_results["correct_count"] = total_correct
    with open(f"analysis-results/{substring_check_mode}_check/{model_name}-analysis-results-NO-choice.json", "w") as file:
        json.dump(all_results, file)


if __name__ == '__main__':
    model_names = [name for name in get_language_models()]
    for model_name in model_names:
        analyze_with_choice_generations(model_name)
    for model_name in get_language_models():
        analyze_without_choice_generations(model_name)
