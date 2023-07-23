import json

import pandas

from commongen_validation_test_set_generation.process_results_csv import get_measures


def find_missed_questions(worker_id: str, results_path: str, batch_input_path: str):
    with open(batch_input_path, "r") as handle:
        lm_names = handle.readline().removesuffix("\n").split(",")[3:]
    batch_inputs = pandas.read_csv(batch_input_path).to_dict(orient="records")
    question_ids = {f"{question['set_name']}-{question['concept_set_id']}" for question in batch_inputs}
    results = pandas.read_csv(results_path).to_dict(orient="records")
    worker_completed_qids = {f"{question['Input.set_name']}-{question['Input.concept_set_id']}" for question in results
                             if question["WorkerId"] == worker_id}
    missed_question_ids = question_ids.difference(worker_completed_qids)
    missed_questions = []
    for batch_input in batch_inputs:
        if f"{batch_input['set_name']}-{batch_input['concept_set_id']}" in missed_question_ids:
            missed_question = {
                "HITId": "n/a",
                "WorkerId": worker_id,
                "AssignmentId": 'n/a',
                "AcceptTime": 'n/a',
                "SubmitTime": 'n/a',
                "SubsetName": batch_input['set_name'],
                "ConceptSetID": batch_input['concept_set_id'],
                "Generations": []
            }
            for lm_name in lm_names:
                lm_dict = {"sentence": batch_input[lm_name], }
                for measure in get_measures():
                    lm_dict[measure] = -1
                missed_question["Generations"].append(lm_dict)
            missed_questions.append(missed_question)
    with open(f"{worker_id}_make_up_questions_form.json", "w") as handle:
        json.dump(missed_questions, handle, indent=4)


def insert_back_made_up_answers(batch_input_path: str, jsoned_results_path: str, made_up_answers_path: str):
    with open(batch_input_path, "r") as handle:
        lm_names = handle.readline().removesuffix("\n").split(",")[3:]
    with open(made_up_answers_path, "r") as handle:
        made_up_answers = json.load(handle)
    made_up_question_ids = set()
    for made_up_answer in made_up_answers:
        made_up_question_ids.add(
            (made_up_answer["SubsetName"], made_up_answer["ConceptSetID"], made_up_answer["WorkerId"]))
        generations = made_up_answer["Generations"]
        for i, generation in enumerate(generations):
            # Check all filled out:
            lm_name = lm_names[i]
            for measure, value in generation.items():
                if measure == "sentence":
                    continue
                assert value > 0, f"The {lm_name}'s {measure} does not have valid value." \
                                  f" Looking at the {made_up_answer['SubsetName']}-{made_up_answer['ConceptSetID']} question."
            made_up_answer[lm_name] = generation
        del made_up_answer["Generations"]
    with open(jsoned_results_path, "r") as handle:
        results_dict = json.load(handle)

    for result in results_dict:
        assert (result["SubsetName"], result["ConceptSetID"], result["WorkerId"]) \
               not in made_up_question_ids, f'{(result["SubsetName"], result["ConceptSetID"])} is already found in the jsoned results file for worker {result["WorkerId"]}'

    results_dict = results_dict + made_up_answers
    with open(jsoned_results_path, "w") as handle:
        json.dump(results_dict, handle)


if __name__ == '__main__':
    worker_id = "A2LRYD92B7Z0GH"
    batch_input_path = "../generated_sentences/combined_generations_first_100.csv"
    results_path = "Batch_387776(sandbox_env_internal_first_100_questions)/Batch_387776_batch_results.csv"
    insert_back_made_up_answers(batch_input_path,
                                "Batch_387776(sandbox_env_internal_first_100_questions)/Batch_387776_batch_results.json",
                                "Batch_387776(sandbox_env_internal_first_100_questions)/A2LRYD92B7Z0GH_make_up_questions_form.json")
