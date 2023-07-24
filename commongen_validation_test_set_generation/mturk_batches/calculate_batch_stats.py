import json

from commongen_validation_test_set_generation.language_models import get_language_models
from commongen_validation_test_set_generation.process_results_csv import get_measures


def calculate_batch_stats(json_batch_results_path: str):
    with open(json_batch_results_path, "r") as handle:
        batch_results = json.load(handle)
    stats = {name: {measure: [] for measure in get_measures()} for name in get_language_models()}
    for result in batch_results:
        for name in get_language_models():
            for measure in get_measures():
                if measure == "sentence": continue
                stats[name][measure].append(result[name][measure])

    for name in stats:
        for measure in get_measures():
            stats[name][measure] = sum(stats[name][measure]) / len(stats[name][measure])

    stats_file_name = json_batch_results_path.replace("results.json", "stats.json")
    with open(stats_file_name, "w") as handle:
        json.dump(stats, handle)

if __name__ == '__main__':
    calculate_batch_stats("Batch_387776(sandbox_env_internal_first_100_questions)/Batch_387776_batch_results.json")
