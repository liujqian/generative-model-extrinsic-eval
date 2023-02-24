import datasets, transformers, pickle
from math import ceil

from transformers import AutoTokenizer, AutoModelForCausalLM


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
        print(f"Generating for the {model_name} model, {ds_name} set! This is the {batch_idx}th batch. There are {num_batch} items in total.")
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


def generate_for_commongen_validation(model, tokenizer, dataset, model_name, ds_name):
    commongen = dataset
    dataset_length = len(commongen["concepts"])
    results = {}
    for idx in range(0, dataset_length):
        print(
            f"Generating for the {model_name} model, {ds_name} set! This is the {idx}th item. There are {dataset_length} items in total.")
        concept_set_idx = commongen[idx]["concept_set_idx"]
        if concept_set_idx in results:
            continue
        results[concept_set_idx] = {}
        concepts = commongen[idx]["concepts"]
        targets = commongen[idx]["target"]
        prompts = "<|endoftext|>" + " ".join(concepts) + "="
        tokenized_input = tokenizer(prompts, return_tensors="pt")
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
        results[concept_set_idx]["sentences"] = sentences[0]
        results[concept_set_idx]["concepts"] = concepts
        results[concept_set_idx]["targets"] = targets
    return results


language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "liujqian/gpt2-large-finetuned-commongen",
    "gpt2-m": "liujqian/gpt2-medium-finetuned-commongen",
    "gpt2": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-finetuned-commongen",
    # "t5": "mrm8488/t5-base-finetuned-common_gen",
    # "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

tokenizers = {
    "gpt2-xl": "gpt2-xl",
    "gpt2-l": "gpt2-large",
    "gpt2-m": "gpt2-medium",
    "gpt2": "gpt2",
    # "t5": "mrm8488/t5-base-finetuned-common_gen",
    # "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

if __name__ == '__main__':
    commongen_full = datasets.load_dataset('common_gen')
    for model_name in language_models:
        tokenizer = AutoTokenizer.from_pretrained(tokenizers[model_name], padding_side='left')
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(language_models[model_name])
        for subset_name in [ "validation"]:
            subset = commongen_full[subset_name]
            if subset_name == "validation":
                results = generate_for_commongen_validation(model, tokenizer, subset, model_name=model_name,
                                                            ds_name=subset_name)
                print(f"Trying to dump the {model_name} {subset_name} generations to file!")
                file = open(
                    f'{model_name}-{subset_name}-commongen-generation.pickle', 'wb')
                # dump information to that file
                pickle.dump(results, file)
                # close the file
                file.close()
                print("Finished dumping the generations to a file!")
            elif subset_name == "test":
                results = generate_for_commongen_test(model, tokenizer, subset, model_name=model_name,
                                                      ds_name=subset_name)
                print(f"Trying to dump the {model_name} {subset_name} generations to file!")
                file = open(
                    f'{model_name}-{subset_name}-commongen-generation.pickle', 'wb')
                # dump information to that file
                pickle.dump(results, file)
                # close the file
                file.close()
                print("Finished dumping the generations to a file!")
            else:
                raise NameError(f"{subset_name} is an invalid subset name!")
