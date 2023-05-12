import datetime
import json
import pickle
import spacy

from datasets import load_dataset

from utils import log_progress

import sys

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


def count_occurences(sentences: list[str], targets: list[str]) -> int:
    cnt = 0
    for target in targets:
        lemmatized_target = lemmatize(target)
        for sentence in sentences:
            truncated = sentence.split("=")[-1]
            lemmatized_sentence = lemmatize(truncated)
            if lemmatized_target in lemmatized_sentence:
                cnt += 1
    return cnt


def analyze_with_choice_generations(model_name: str):
    dataset = load_dataset("liujqian/commonsenseqa_with_content_words")
    choice_idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    all_results = {}
    for subset_name in file_postfix_without_choice:
        with open(
                f"generated_sentences/{model_name}{file_postfix_with_choice[subset_name]}",
                "r",
        ) as file:
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
                log_progress(i, len(subset["id"]),
                             f"Analyzing the results of {model_name} on the {subset_name} set. This is WITH choices' content words.")
            question_generations = model_generations[str(i)]
            question = subset[i]
            all_generated_sentences = question_generations["sentences"]
            all_sequences_scores = question_generations["sequences_scores"]
            cur_question_choices_stats = {
                "inclusion_count": [],
                "avg_sequences_scores": []
            }
            for choice_idx in range(0, 5):
                cur_choice_content_words = question[f"choice_{choice_idx}_content_words"]
                all_generations_for_cur_choice = all_generated_sentences[choice_idx * 4:choice_idx * 4 + 4]
                all_scores_for_cur_choice = all_sequences_scores[choice_idx * 4:choice_idx * 4 + 4]
                inclusion_count = count_occurences(all_generations_for_cur_choice, cur_choice_content_words)
                cur_question_choices_stats["inclusion_count"].append(inclusion_count)
                cur_question_choices_stats["avg_sequences_scores"].append(sum(all_scores_for_cur_choice)/len(all_scores_for_cur_choice))

            letter_correct_choice = question["answerKey"]
            idx_correct_choice = choice_idx_map[letter_correct_choice]
            stats["correct_prediction_by_inclusion_count"] += cur_question_choices_stats["inclusion_count"][
                                                                  idx_correct_choice] == max(
                cur_question_choices_stats["inclusion_count"])
            stats["correct_prediction_by_sequences_score"] += idx_correct_choice == argmax(
                cur_question_choices_stats["avg_sequences_scores"])
            stats["avg_correct_prediction_inclusion"] = ((stats["avg_correct_prediction_inclusion"] * stats[
                "total_examined"]) + cur_question_choices_stats["inclusion_count"][idx_correct_choice]) / (stats[
                                                                                                               "total_examined"] + 1)
            stats["avg_correct_prediction_sequence_score"] = ((stats["avg_correct_prediction_sequence_score"] * stats[
                "total_examined"]) + cur_question_choices_stats["avg_sequences_scores"][idx_correct_choice]) / (stats[
                                                                                                                    "total_examined"] + 1)
            wrong_prediction_inclusion_count = [v for ci, v in enumerate(cur_question_choices_stats["inclusion_count"])
                                                if ci != idx_correct_choice]
            wrong_sequence_scores = [v for ci, v in enumerate(cur_question_choices_stats["avg_sequences_scores"])
                                     if ci != idx_correct_choice]
            stats["avg_wrong_prediction_inclusion"] = ((stats["avg_wrong_prediction_inclusion"] * stats[
                "total_examined"]) + sum(wrong_prediction_inclusion_count) / len(wrong_prediction_inclusion_count)) / (
                                                              stats["total_examined"] + 1)
            stats["avg_wrong_prediction_sequence_score"] = ((stats["avg_wrong_prediction_sequence_score"] * stats[
                "total_examined"]) + sum(wrong_sequence_scores) / len(wrong_sequence_scores)) / (stats[
                                                                                                     "total_examined"] + 1)
            stats["total_examined"] += 1
        all_results[subset_name] = stats
    with open(f"analysis-results/{model_name}-analysis-results-WITH-choice.json", "w") as file:
        json.dump(all_results, file)


def analyze_without_choice_generations(model_name: str):
    dataset = load_dataset("liujqian/commonsenseqa_with_content_words")
    choice_idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    all_results = {}
    for subset_name in file_postfix_without_choice:
        with open(
                f"generated_sentences/{model_name}{file_postfix_without_choice[subset_name]}",
                "r"
        ) as file:
            model_generations = json.load(file)
        subset = dataset[subset_name]
        # what we need record:
        # prediction accuracy, percentage of times the correct term is predicted at all,
        # average correct term shown, percentage of any term shown, average any term shown.
        stats = {
            "total_examined": 0,
            "correct_prediction": 0,
            "total_questions_correct_choice_word_generated": 0,
            "total_questions_any_choice_word_generated": 0
        }
        # rule: if a word is only appearing in one choice, then predicting the word is predicting the choice.
        #       if a word appear in multiple choices, then split the count
        for i in range(len(subset["id"])):
            if i % 200 == 0:
                log_progress(i, len(subset["id"]),
                             f"Analyzing the results of {model_name} on the {subset_name} set. This is WITHOUT choices' content words.")
            question_generations = model_generations[str(i)]
            question = subset[i]
            # key are all choice content words, values are sets indicating which choices appeared in
            choice_content_words_stats = {}
            for choice_idx in range(0, 5):
                choice_content_words = question[f"choice_{choice_idx}_content_words"]
                for word in choice_content_words:
                    if word not in choice_content_words_stats:
                        choice_content_words_stats[word] = {choice_idx}
                    else:
                        choice_content_words_stats[word].add(choice_idx)
            choice_mention_count = [0, 0, 0, 0, 0]
            for sentence in question_generations["sentences"]:
                for choice_content_word in choice_content_words_stats:
                    if choice_content_word in lemmatize(sentence):
                        for choice_idx in choice_content_words_stats[choice_content_word]:
                            choice_mention_count[choice_idx] += 1 / len(
                                choice_content_words_stats[choice_content_word])
            predicted_choice_idx = argmax(choice_mention_count)
            stats["total_examined"] += 1
            letter_correct_choice = question["answerKey"]
            stats["correct_prediction"] += predicted_choice_idx == choice_idx_map[letter_correct_choice]
            stats["total_questions_correct_choice_word_generated"] += choice_mention_count[
                                                                          choice_idx_map[letter_correct_choice]] > 0
            stats["total_questions_any_choice_word_generated"] += sum(choice_mention_count) > 0
        all_results[subset_name] = stats
    with open(f"analysis-results/{model_name}-analysis-results-NO-choice.json", "w") as file:
        json.dump(all_results, file)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give a model size")
        sys.exit(1)

    models = sys.argv[1:]
    for model_name in models:
        analyze_with_choice_generations(model_name)
        analyze_without_choice_generations(model_name)
