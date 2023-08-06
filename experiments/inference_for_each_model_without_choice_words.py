import json

import datasets
from transformers import StoppingCriteriaList

from models import mt0, stablelm_tuned_alpha_3b, redpajama_incite_instruct_3b_v1
from utils import log_progress

language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "liujqian/gpt2-large-finetuned-commongen",
    "gpt2-m": "liujqian/gpt2-medium-finetuned-commongen",
    "gpt2": "liujqian/gpt2-finetuned-commongen",
    "t5": "mrm8488/t5-base-finetuned-common_gen",
    "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

tokenizers = {
    "gpt2-xl": "gpt2-xl",
    "gpt2-l": "gpt2-large",
    "gpt2-m": "gpt2-medium",
    "gpt2": "gpt2",
    "t5": "mrm8488/t5-base-finetuned-common_gen",
    "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}


def generate_without_choice_content_words(
        model,
        tokenizer,
        prompt_generator,
        result_separator,
        stop_criteria,
        set_name: str,
        examples: datasets.Dataset
):
    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            log_progress(
                i,
                len(examples["id"]),
                20,
                f"Generating inferences for the {set_name} set. Without choice content words."
            )
        question_content_words = examples["question_content_words"][i]
        prompt = prompt_generator(question_content_words)
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(0)
        outputs = model.generate(
            **tokenized_input,
            num_beams=20,
            num_beam_groups=20,
            num_return_sequences=20,
            diversity_penalty=100.0,
            remove_invalid_values=True,
            temperature=1.0,
            max_new_tokens=256,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=StoppingCriteriaList([stop_criteria] if stop_criteria is not None else [])
        )
        sentences = []
        for output in outputs.sequences:
            sentence = tokenizer.decode(output, skip_special_tokens=True)
            if result_separator is not None:
                sentence = result_separator(sentence)
            sentences.append(sentence)
        results[i] = {"id": examples["id"][i], "sentences": sentences}
    return results


if __name__ == '__main__':
    model_funcs = {
        "stablelm_tuned_alpha_3b": stablelm_tuned_alpha_3b,
        "redpajama_incite_instruct_3b_v1": redpajama_incite_instruct_3b_v1
    }
    for name, func in model_funcs.items():
        model_utils = func()
        model_name = name
        if len(model_utils) == 3:
            (model, tokenizer, prompt_generator) = model_utils
            result_seperator = None
            stop_criteria = None
        elif len(model_utils) == 4:
            (model, tokenizer, prompt_generator, result_seperator) = model_utils
            stop_criteria = None
        elif len(model_utils) == 5:
            (model, tokenizer, prompt_generator, result_seperator, stop_criteria) = model_utils
        else:
            raise ValueError(f"The model func of {name} gives {len(model_utils)} returns.")

        for subset_name in ["train", "validation"]:
            new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
            generations = generate_without_choice_content_words(
                model,
                tokenizer,
                prompt_generator,
                result_seperator,
                stop_criteria,
                subset_name,
                new_ds[subset_name]
            )
            print(f"Trying to dump the generations to a file!")
            file = open(
                f'generated_sentences/{model_name}-{subset_name}-WITHOUT-choicewords-noquestionwordlimit.json',
                'w'
            )
            json.dump(generations, file)
            # close the file
            file.close()
            print("Finished dumping the generations to a file!")
