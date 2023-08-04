import unittest

from experiments.results_analysis import count_occurences, get_correct_choice_idx, \
    get_current_question_choice_status_with_choice_analysis, \
    check_inclusion_scores_predict_correctness, update_subset_stats_for_with_choice_analysis, get_choice_mention_count, \
    check_no_choice_predict_correctness, update_subset_stats_for_without_choice_analysis, substring_check_mode


def get_sample_question() -> dict:
    return {'id': 'e8a8b3a2061aa0e6d7c6b522e9612824',
            'question': 'What home entertainment equipment requires cable?',
            'question_concept': 'cable',
            'choices': {'label': ['A', 'B', 'C', 'D', 'E'],
                        'text': ['radio shack', 'substation', 'cabinet', 'television', 'desk']},
            'answerKey': 'D',
            'question_content_words': ['home',
                                       'entertainment',
                                       'equipment',
                                       'require',
                                       'cable'],
            'choice_0_content_words': ['radio', 'shack'],
            'choice_1_content_words': [''],
            'choice_2_content_words': ['cabinet'],
            'choice_3_content_words': ['television'],
            'choice_4_content_words': ['desk']}


class TestResultAnalysis(unittest.TestCase):
    def test_count_occurence(self):
        sentences = [
            "I love to go to New York in Germany.",
            "I hope I can go to top of the world one day. I would love that."
        ]
        target = ["love", "top"]
        self.assertEqual(count_occurences(sentences, target), 3)
        self.assertEqual(count_occurences(sentences, []), 0)
        self.assertEqual(count_occurences(sentences, ["New York", "hope"]), 2)
        self.assertEqual(count_occurences(sentences, ["world"]), 1)
        self.assertEqual(count_occurences(sentences, ["apple"]), 0)
        self.assertEqual(count_occurences(sentences, ["germany"]), 1)
        self.assertEqual(count_occurences(sentences, ["in", ""]), 1)
        self.assertEqual(count_occurences(sentences, ["go", "hope", "Germany"]), 4)

    def test_get_correct_idx(self):
        idx = get_correct_choice_idx(get_sample_question())
        self.assertEqual(idx, 3)

    def test_get_current_question_choice_status_with_choice_analysis(self):
        generations = [
            "I love radio shack.", "Radio shack loves me.", "I hate you.", "I play radio.",
            "I like substation.", "Eat.", "Sleep.", "Drink.",
            "Hit.", "Drive.", "Poop.", "Beat.",
            "I watch television.", "I like televisions as a television makes me happy.", "I love televisions.",
            "Televisions are what I have.",
            "I have a desk.", "I don't eat apples.", "I bought many desks.", "I slept."
        ]
        scores = [1, 1, 1, 1, -2, -3, 4, 5, 1, 2, 3, 4, 0, 0, 0, 0, -1, -2, -3, -4]
        g = {"sentences": generations, "sequences_scores": scores, "id": 1}
        results = get_current_question_choice_status_with_choice_analysis(question=get_sample_question(), generations=g)
        self.assertEqual(results["inclusion_count"],
                         [5, 0, 0, 4, 2] if substring_check_mode == "use_in" else [5, 0, 0, 5, 2])
        self.assertEqual(results["avg_sequences_scores"], [1, 1, 2.5, 0, -2.5])

    def test_check_inclusion_scores_predict_correctness(self):
        stats = {
            "inclusion_count": [5, 0, 0, 5, 2],
            "avg_sequences_scores": [1, 2, 2.5, 2.5, -2.5]
        }
        self.assertEqual(check_inclusion_scores_predict_correctness(idx_correct_choice=0, choices_stats=stats),
                         (True, False))
        self.assertEqual(check_inclusion_scores_predict_correctness(idx_correct_choice=3, choices_stats=stats),
                         (True, True))
        self.assertEqual(check_inclusion_scores_predict_correctness(idx_correct_choice=4, choices_stats=stats),
                         (False, False))

    def test_update_subset_stats_for_with_choice_analysis(self):
        stats1 = {
            "total_examined": 2,
            "correct_prediction_by_inclusion_count": 78,
            "correct_prediction_by_sequences_score": 4,
            "avg_correct_prediction_inclusion": 20,
            "avg_wrong_prediction_inclusion": 7,
            "avg_correct_prediction_sequence_score": 2,
            "avg_wrong_prediction_sequence_score": 2,
        }
        choices_stats = {
            "inclusion_count": [5, 0, 0, 5, 2],
            "avg_sequences_scores": [1, 2, 2.5, 2.5, -2.5]
        }
        update_subset_stats_for_with_choice_analysis(
            stats=stats1,
            choices_stats=choices_stats,
            idx_correct_choice=3
        )
        self.assertEqual(stats1["total_examined"], 3)
        self.assertEqual(stats1["correct_prediction_by_inclusion_count"], 79)
        self.assertEqual(stats1["correct_prediction_by_sequences_score"], 5)
        self.assertEqual(stats1["avg_correct_prediction_inclusion"], 15)
        update_subset_stats_for_with_choice_analysis(
            stats=stats1,
            choices_stats=choices_stats,
            idx_correct_choice=0
        )
        self.assertEqual(stats1["total_examined"], 4)
        self.assertEqual(stats1["correct_prediction_by_inclusion_count"], 80)
        self.assertEqual(stats1["correct_prediction_by_sequences_score"], 5)
        update_subset_stats_for_with_choice_analysis(
            stats=stats1,
            choices_stats=choices_stats,
            idx_correct_choice=2
        )
        self.assertEqual(stats1["total_examined"], 5)
        self.assertEqual(stats1["correct_prediction_by_inclusion_count"], 80)
        self.assertEqual(stats1["correct_prediction_by_sequences_score"], 6)
        update_subset_stats_for_with_choice_analysis(
            stats=stats1,
            choices_stats=choices_stats,
            idx_correct_choice=4
        )
        self.assertEqual(stats1["total_examined"], 6)
        self.assertEqual(stats1["correct_prediction_by_inclusion_count"], 80)
        self.assertEqual(stats1["correct_prediction_by_sequences_score"], 6)

    def test_get_choice_mention_count(self):
        question = get_sample_question()
        generations = [
            "I love radio stations.", "I eat shake shack.", 'I have a net', "I have televisions.",
            "My box is on my desk.",
            "I have a radio shack.", "I have a radio on my bed.", "My television is in my room.",
            "I sleep on my television on my desk.",
            "I have a shack.",
            "Sleep.", "Eat", "Hit.", "I love!", "I eat apples.",
            "Play.", "Drive.", "Sing.", "Smile.", "Take it!"
        ]
        counts = [6, 0, 0, 3, 2]
        results = get_choice_mention_count(question, {"sentences": generations})
        self.assertEqual(counts, results)

    def test_check_no_choice_predict_correctness(self):
        self.assertTrue(check_no_choice_predict_correctness([6, 6, 0, 3, 2], 0))
        self.assertTrue(check_no_choice_predict_correctness([6, 6, 0, 3, 2], 1))
        self.assertTrue(not check_no_choice_predict_correctness([6, 6, 0, 3, 2], 2))
        self.assertTrue(check_no_choice_predict_correctness([6, 6, 6, 6, 2], 3))
        self.assertTrue(check_no_choice_predict_correctness([4, 5, 12, 9, 8], 2))
        self.assertTrue(check_no_choice_predict_correctness([6, 6, 6, 6, 2], 2))

    def test_update_subset_stats_for_without_choice_analysis(self):
        stats = {
            "total_examined": 4,
            "correct_prediction": 5,
            "total_correct_choice_word_generated": 10,
        }
        update_subset_stats_for_without_choice_analysis(stats, [6, 6, 0, 3, 2], 1)
        self.assertEqual(stats["total_examined"], 5)
        self.assertEqual(stats["correct_prediction"], 6)
        self.assertEqual(stats["total_correct_choice_word_generated"], 16)
        update_subset_stats_for_without_choice_analysis(stats, [6, 6, 0, 3, 2], 4)
        self.assertEqual(stats["total_examined"], 6)
        self.assertEqual(stats["correct_prediction"], 6)
        self.assertEqual(stats["total_correct_choice_word_generated"], 18)
        update_subset_stats_for_without_choice_analysis(stats, [0, 0, 0, 0, 0], 4)
        self.assertEqual(stats["total_examined"], 7)
        self.assertEqual(stats["correct_prediction"], 6)
        self.assertEqual(stats["total_correct_choice_word_generated"], 18)
