import json
import math

import pandas

from commongen_validation_test_set_generation.chatgpt_generate_confounding_choices import get_confounding_types
from commongen_validation_test_set_generation.language_models import get_language_models

measures = ["complexity", "fluency", "sensibility"]


def get_measures():
    return measures


def process_csv(path: str) -> list:
    df = pandas.read_csv(path)
    results = []
    batch_dict = df.to_dict(orient="records")
    confounding_types = get_confounding_types()
    lm_names = [lm_name for lm_name in get_language_models()]
    for hit in batch_dict:
        HIT_id = hit['HITId']
        worker_id = hit['WorkerId']
        assignment_id = hit['AssignmentId']
        hit_result = {
            "HITId": HIT_id,
            "WorkerId": worker_id,
            "AssignmentId": assignment_id,
            "AcceptTime": hit["AcceptTime"],
            "SubmitTime": hit["SubmitTime"],
            "SubsetName": hit["Input.set_name"],
            "ConceptSetID": hit["Input.concept_set_id"]
        }
        for lm_name in lm_names + confounding_types:
            lm_result = {"sentence": hit[f"Input.{lm_name}"]}
            for measure in measures:
                score = find_score(hit, lm_name, measure)
                lm_result[measure] = score
            hit_result[lm_name] = lm_result
        results.append(hit_result)
    return results


def find_score(hit_submission: dict, lm_name: str, measure: str) -> int:
    for i in range(1, 6):
        key = f"Answer.{lm_name}-{measure}.{i}"
        if hit_submission[key]:
            if math.isnan(hit_submission[key]): return -2
            return i
    raise ValueError(
        f"The dictionary {str(hit_submission)} does not contain a valid score for language model {lm_name}.")


def dump_json_processed_csv(processed_csv: list, file_name: str):
    with open(file_name, "w") as handle:
        json.dump(processed_csv, handle)


def fill_out_missing_values(json_results_path):
    with open(json_results_path, "r") as handle:
        results_dict = json.load(handle)
    records_missing_values = [result for result in results_dict if check_has_missing_values(result)]

    for record in records_missing_values:
        fix_missing_value_records(record)

    for record in records_missing_values:
        assert not check_has_missing_values(record)
    return results_dict


def fix_missing_value_records(record):
    cant_fix = 0
    lm_names = [lm_name for lm_name in list(get_language_models().keys()) + get_confounding_types()]
    for lm_name in lm_names:
        if -2 in record[lm_name].values():
            fixed = False
            sentence = record[lm_name]["sentence"]
            for search_name in lm_names:
                if record[search_name]["sentence"] == sentence and -2 not in record[search_name].values():
                    for measure in get_measures():
                        record[lm_name][measure] = record[search_name][measure]
                    fixed = True
                    break
            if not fixed:
                raise ValueError


def check_has_missing_values(d):
    lm_names = [lm_name for lm_name in get_language_models()]
    for lm_name in lm_names:
        if -2 in d[lm_name].values():
            return True
    return False


predicates = {
    "complex": lambda d: d["complex"]["complexity"] >= 3,
    "nonsense": lambda d: d["nonsense"]["sensibility"] <= 3,
    "grammar_error": lambda d: d["grammar_error"]["fluency"] <= 3,
    "simple": lambda d: d["simple"]["fluency"] >= 3
}


def find_bad_responses(responses: list[dict]) -> list[dict]:
    bad_responses = []
    for response in responses:
        passed_cnt = 0
        for predicate in predicates:
            predicate_func = predicates[predicate]
            passed_cnt += predicate_func(response)
        if not (passed_cnt >= 3):
            bad_responses.append(response)
    return bad_responses


if __name__ == '__main__':
    name = "mturk_batches/Batch_5109423(prod_env_first_100_questions)/Batch_5109423_batch_results"
    processed = process_csv(name + ".csv")
    with open(name + ".json", "w") as handle:
        json.dump(processed, handle)

# bad_ones = find_bad_responses(processed)
# with open(name + "_bad_ones.json", "w") as handle:
#     json.dump(bad_ones, handle)
# print(f"Found {len(bad_ones)} bad responses.")
# bad_assignment_ids = {bad_response["AssignmentId"] for bad_response in bad_ones}
# good_ones = [response for response in processed if response["AssignmentId"] not in bad_assignment_ids]
# with open(name + "_good_ones.json", "w") as handle:
#     json.dump(good_ones, handle)
# print(f"Found {len(good_ones)} good responses.")
#
# lm_names = get_language_models().keys()
# lm_stats = {lm_name: {measure: [] for measure in measures} for lm_name in lm_names}
# for response in processed:
#     for lm_name in lm_names:
#         for measure in measures:
#             lm_stats[lm_name][measure].append(response[lm_name][measure])
# for lm_name in lm_names:
#     for measure in measures:
#         lm_stats[lm_name][measure] = sum(lm_stats[lm_name][measure]) / len(lm_stats[lm_name][measure])
# with open(name + "_averaged_stats.json", 'w') as handle:
#     json.dump(lm_stats, handle)
