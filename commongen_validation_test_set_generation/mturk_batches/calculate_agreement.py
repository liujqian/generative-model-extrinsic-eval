import json

from commongen_validation_test_set_generation.language_models import get_language_models
from commongen_validation_test_set_generation.process_results_csv import get_measures


def calculate_kendall_w(submissions: dict):
    qids = {(s["SubsetName"], s["ConceptSetID"]) for s in submissions}
    measures = get_measures()
    objects_rank_map = {}
    for qid in qids:
        objects_rank_map[qid] = {}
        q_subs = [s for s in submissions if (s["SubsetName"], s["ConceptSetID"]) == qid]
        assert len(q_subs) == 3
        for measure in measures:
            objects_rank_map[qid][measure] = {}
            objects_rank_map[qid][measure]["total_correction_factor"] = sum([q_sub[f"{measure}_correction_factor"] for q_sub in q_subs])
            for lm in get_language_models():
                rank_sum = sum([q_sub[lm][f"{measure}_rank"] for q_sub in q_subs])
                objects_rank_map[qid][measure][lm] = rank_sum ** 2

    for qid in qids:
        for measure in measures:
            n = len(get_language_models())
            numerator = sum([objects_rank_map[qid][measure][lm] for lm in get_language_models()]) * 12 - 3 * 9 * n * ((n + 1) ** 2)
            denominator = 9 * n * (n * n - 1) - 3 * objects_rank_map[qid][measure]["total_correction_factor"]
            if denominator == 0:
                assert numerator == 0
                q_subs = [s for s in submissions if (s["SubsetName"], s["ConceptSetID"]) == qid]
                for lm in get_language_models():
                    assert q_subs[0][lm][f"{measure}_rank"] == q_subs[1][lm][f"{measure}_rank"] and q_subs[0][lm][f"{measure}_rank"] == q_subs[2][lm][
                        f"{measure}_rank"]
                W = 1
            else:
                W = numerator / denominator
            objects_rank_map[qid][measure]["W"] = W
    total_W = 0
    for qid in qids:
        for measure in measures:
            total_W += objects_rank_map[qid][measure]["W"]
    avg_w = total_W / (len(qids) * len(measures))
    print(avg_w)


def calculate_ranks_for_submission(submission: dict):
    measures = get_measures()
    models = get_language_models()
    num_models = len(models)
    for measure in measures:
        ratings = [submission[lm][measure] for lm in models]
        ranks = list(range(1, num_models + 1))
        rating_ranks = {r: 0 for r in range(5, 0, -1)}
        correction_factor = 0
        for rate in range(5, 0, -1):
            count = ratings.count(rate)
            group_correction_factor = (count ** 3) - count
            correction_factor += group_correction_factor
            if count == 0:
                continue
            rank_sum = 0
            for _ in range(count):
                rank_sum += ranks.pop()
            real_rank = rank_sum / count
            rating_ranks[rate] = real_rank
        assert len(ranks) == 0, f"For submission {submission}, not all ranks are used."

        for lm in models:
            submission[lm][f"{measure}_rank"] = rating_ranks[submission[lm][measure]]
        submission[f"{measure}_correction_factor"] = correction_factor


if __name__ == '__main__':
    batch_1 = "sandbox_env_internal_first_100_questions/Batch_387776_batch_results_with_ranking"
    with open(batch_1 + ".json", "r") as handle:
        submissions1 = json.load(handle)
    # for submission in submissions1:
    #     calculate_ranks_for_submission(submission)
    # with open(batch_1 + ".json", "w") as handle:
    #     json.dump(submissions1, handle)

    batch_2 = "sandbox_env_internal_second_100_questions/Batch_388388_batch_results_with_ranking"
    with open(batch_2 + ".json", "r") as handle:
        submissions2 = json.load(handle)
    # for submission in submissions2:
    #     calculate_ranks_for_submission(submission)
    # with open(batch_2 + ".json", "w") as handle:
    #     json.dump(submissions2, handle)

    calculate_kendall_w(submissions1)
    calculate_kendall_w(submissions2)
