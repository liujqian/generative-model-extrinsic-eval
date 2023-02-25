import datetime
import json
import pickle
import numpy

from datasets import load_dataset

language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "liujqian/gpt2-large-finetuned-commongen",
    "gpt2-m": "liujqian/gpt2-medium-finetuned-commongen",
    "gpt2": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-finetuned-commongen",
    # "t5": "mrm8488/t5-base-finetuned-common_gen",
    # "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

pickle_file_postfix_without_choice = {
    "train": "-train-withoutchoicewords-noquestionwordlimit.pickle",
    "validation": "-validation-withoutchoicewords-noquestionwordlimit.pickle"
}

pickle_file_postfix_with_choice = {
    "train": "-train-withchoicewords-noquestionwordlimit.pickle",
    "validation": "-validation-withchoicewords-noquestionwordlimit.pickle"
}


# This function is copied directly from this post: https://stackoverflow.com/a/26726185
def argmax(iterable):
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def log(cur_idx, total, set_name, model_name, interval):
    if cur_idx % 200 != 0:
        return
    print(
        str(datetime.datetime.now()) + " making the " + str(
            cur_idx) + "th generation for the " + set_name + " set using the " + model_name + " model. There are " + str(
            total) + " examples in total!")


def count_occurences(sentences: list[str], targets: list[str]) -> int:
    cnt = 0
    for target in targets:
        for sentence in sentences:
            truncated = sentence.split("=")[-1]
            if target in truncated:
                cnt += 1
    return cnt


def analyze_with_choice_generations(model_name: str):
    dataset = load_dataset("liujqian/commonsenseqa_with_content_words")
    choice_idx_map = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
    all_results = {}
    for subset_name in pickle_file_postfix_without_choice:
        with open(f"generated_sentences/{model_name}{pickle_file_postfix_with_choice[subset_name]}", "rb") as file:
            model_generations = pickle.load(file)
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
            log(i, len(subset["id"]), model_name=model_name, set_name=subset_name, interval=200)
            question_generations = model_generations[i]
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
                cur_question_choices_stats["avg_sequences_scores"].append(all_scores_for_cur_choice.mean())

            letter_correct_choice = question["answerKey"]
            idx_correct_choice = choice_idx_map[letter_correct_choice]
            stats["correct_prediction_by_inclusion_count"] += idx_correct_choice == argmax(
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
    for subset_name in pickle_file_postfix_without_choice:
        with open(f"generated_sentences/{model_name}{pickle_file_postfix_without_choice[subset_name]}", "rb") as file:
            model_generations = pickle.load(file)
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
            log(i, len(subset["id"]), model_name=model_name, set_name=subset_name, interval=200)
            question_generations = model_generations[i]
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
                    if choice_content_word in sentence:
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
    for model_name in ["gpt2", "gpt2-m", "gpt2-l"]:
        analyze_with_choice_generations(model_name)
