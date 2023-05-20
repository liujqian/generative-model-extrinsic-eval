import json

import datasets, transformers, pickle
from math import ceil

from commongen_validation_test_set_generation.language_models import get_language_models
from experiments.utils import log_progress


# Deprecated. This method uses batch inference. To maintain efficient memory usage, only batch size of one is used
# for this task and hence only use "generate_for_commongen" below.
def generate_for_commongen_test(model, tokenizer, dataset, model_name, ds_name):
    commongen = dataset
    dataset_length = len(commongen["concepts"])
    batch_size = 4
    num_batch = ceil(dataset_length / batch_size)
    results = {
        "dataset_length": dataset_length,
        "concepts": [],
        "generated_sentences": []
    }
    for batch_idx in range(0, num_batch):
        print(
            f"Generating for the {model_name} model, {ds_name} set! This is the {batch_idx}th batch. There are {num_batch} batches in total.")
        indices = [batch_idx * batch_size + idx for idx in [0, 1, 2, 3] if
                   batch_idx * batch_size + idx < dataset_length]
        concepts_batch = commongen[indices]["concepts"]
        prompts_batch = ["<|endoftext|>" + " ".join(concepts) + "=" for concepts in concepts_batch]
        tokenized_input = tokenizer(prompts_batch, return_tensors="pt", padding=True)
        outputs = model.generate(
            **tokenized_input,
            num_beams=3,
            num_return_sequences=1,
            remove_invalid_values=True,
            temperature=1,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sentences = []
        for output in outputs.sequences:
            sentence = tokenizer.decode(output, skip_special_tokens=True)
            sentences.append(sentence)
        results["concepts"].extend(concepts_batch)
        results["generated_sentences"].extend(sentences)
    return results


def generate_for_commongen(model, tokenizer, prompt_generator, result_separator, dataset, model_name, ds_name):
    commongen = dataset
    dataset_length = len(commongen["concepts"])
    results = {}
    for idx in range(0, dataset_length):
        log_progress(idx, dataset_length, f"Generating for CommonGen {ds_name} set. The model name is {model_name}.")
        concept_set_idx = commongen[idx]["concept_set_idx"]
        if concept_set_idx in results:
            continue
        results[concept_set_idx] = {}
        concepts = commongen[idx]["concepts"]
        target = commongen[idx]["target"]
        prompt = prompt_generator(concepts)
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(0)
        outputs = model.generate(
            **tokenized_input,
            num_beams=3,
            num_return_sequences=1,
            remove_invalid_values=True,
            temperature=1,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
        )
        sentences = []
        for output in outputs.sequences:
            sentence = tokenizer.decode(output, skip_special_tokens=True)
            if result_separator is not None:
                sentence = result_separator(sentence)
            sentences.append(sentence)
        results[concept_set_idx]["sentences"] = sentences[0]
        results[concept_set_idx]["concepts"] = concepts
        results[concept_set_idx]["targets"] = target
    return results


if __name__ == '__main__':
    commongen_full = datasets.load_dataset('common_gen')
    for model_name in get_language_models():
        model_func = get_language_models()[model_name]
        model_suite = model_func()
        if len(model_suite) == 3:
            model, tokenizer, prompt_generator = model_suite
            result_separator = None

        elif len(model_suite) == 4:
            model, tokenizer, prompt_generator, result_separator = model_suite
        else:
            raise ValueError(f"The model function of model {model_name} returns {len(model_suite)} values!")
        for subset_name in ["test", "validation"]:
            subset = commongen_full[subset_name]
            results = generate_for_commongen(
                model,
                tokenizer,
                prompt_generator,
                result_separator,
                subset,
                model_name=model_name,
                ds_name=subset_name,
            )
            print(f"Trying to dump the CommonGen {subset_name} set generations from model {model_name} to a file!")
            file = open(
                f'generated_sentences/{model_name}-commongen-{subset_name}-set-generation.json',
                'w'
            )
            # dump information to that file
            json.dump(results, file)
            # close the file
            file.close()
            print(f"Finished dumping the CommonGen {subset_name} set generations from model {model_name} to a file!")
