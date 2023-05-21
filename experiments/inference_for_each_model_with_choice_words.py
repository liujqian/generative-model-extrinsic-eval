import json
from typing import Callable

import datasets

import models
from utils import log_progress
import sys

from accelerate import Accelerator
import torch


models_map = {
    "mt0": models.mt0,
    "flan_t5": models.flan_t5,
}


def generate_with_choice_content_words(
        model,
        tokenizer,
        prompt_generator: Callable[[list[str]], str],
        set_name: str,
        examples: datasets.Dataset
):
    # Initialize the accelerator
    accelerator = Accelerator()

    # Prepare the model and tokenizer using the accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.to(accelerator.device)

    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            log_progress(i, len(examples["id"]),
                f"Generating inferences for the {set_name} set. Without choice content words.")

        question_content_words = examples["question_content_words"][i]
        prompt = prompt_generator(question_content_words)
        tokenized_input = tokenizer(prompt, return_tensors="pt").to(accelerator.device)

        outputs = accelerator.unwrap_model(model).generate(
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
                sentence = result_separator(sentence, question_content_words)
            sentences.append(sentence)
        results[i] = {"id": examples["id"][i], "sentences": sentences}
    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give a model size")
        sys.exit(1)

    model_name_and_size = sys.argv[1]
    name, size = model_name_and_size.split("-")

    model, tokenizer, prompt_generator = models_map[name](size)
    for subset_name in ["train", "validation"]:
        new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words", num_proc=4)
        generations = generate_with_choice_content_words(
            model,
            tokenizer,
            prompt_generator,
            subset_name,
            new_ds[subset_name]
        )

        print(f"Trying to dump the generations to a file!")
        file = open(
            f'generated_sentences/{name}-{size}-{subset_name}-WITH-choicewords-noquestionwordlimit.json',
            'w'
        )
        # dump information to that file
        json.dump(generations, file)
        # close the file
        file.close()
        print("Finished dumping the generations to a file!")
