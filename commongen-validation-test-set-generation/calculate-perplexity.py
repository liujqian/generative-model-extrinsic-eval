import json

import datasets
from evaluate import load
from transformers import AutoTokenizer


def calculate_commongen_validation_set_perplexity(full_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    subset_name = "validation"
    commongen_full = datasets.load_dataset('common_gen')
    validation_set = commongen_full[subset_name]
    all_texts = []
    for i in range(len(validation_set["concept_set_idx"])):
        target = validation_set["target"][i]
        concepts = validation_set["concepts"][i]
        text = "<|endoftext|>" + " ".join(concepts) + "=" + target
        all_texts.append(text)
    print("Done preparing texts.")
    perplexity = load("perplexity", module_type="metric")
    results = perplexity.compute(predictions=all_texts, model_id=full_model_name)
    with open(f"perplexity/{full_model_name.split('/')[1]}-commongen-validation-set.json", "w") as handle:
        json.dump(results, handle)


if __name__ == '__main__':
    calculate_commongen_validation_set_perplexity(full_model_name="liujqian/gpt2-xl-finetuned-commongen")
