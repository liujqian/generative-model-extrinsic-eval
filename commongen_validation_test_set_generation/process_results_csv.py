import pandas

from commongen_validation_test_set_generation.chatgpt_generate_confounding_choices import get_confounding_types
from commongen_validation_test_set_generation.language_models import get_language_models

measures = ["complexity", "fluency", "sensibility"]


def get_measures():
    return measures


def process_csv(path: str) -> dict:
    df = pandas.read_csv(path)
    results = {}
    batch_dict = df.to_dict(orient="records")
    confounding_types = get_confounding_types()
    lm_names = [lm_name for lm_name in get_language_models()]
    for hit in batch_dict:
        HIT_id = hit['HITId']
        worker_id = hit['WorkerId']
        assignment_id = hit['AssignmentId']
        hit_result = {}
        for lm_name in lm_names + confounding_types:
            lm_result = {}
            for measure in measures:
                score = find_score(hit, lm_name, measure)
                lm_result[measure] = score
            hit_result[lm_name] = lm_result
        results[(HIT_id, worker_id, assignment_id)] = hit_result
    return results


def find_score(hit_submission: dict, lm_name: str, measure: str) -> int:
    for i in range(1, 6):
        key = f"Answer.{lm_name}-{measure}.{i}"
        if hit_submission[key]:
            return i
    raise ValueError(
        f"The dictionary {str(hit_submission)} does not contain a valid score for language model {lm_name}.")

if __name__ == '__main__':
    print(process_csv("C:\\Users\\Jingqian\\Downloads\\Batch_387721_batch_results.csv"))