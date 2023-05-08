import json
from typing import Callable
from models import tk_instruct_3b_def, t0_3b, dolly_v1_6b, flan_t5_large, bloomz, flan_t5_xxl
import datasets

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
        set_name: str,
        examples: datasets.Dataset
):
    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            log_progress(i, len(examples["id"]),
                         f"Generating inferences for the {set_name} set. Without choice content words.")
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
    model, tokenizer, prompt_generator = flan_t5_xxl()
    model_name = "flan_t5_xxl"
    for subset_name in ["train", "validation"]:
        new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
        generations = generate_without_choice_content_words(
            model,
            tokenizer,
            prompt_generator,
            None,
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
