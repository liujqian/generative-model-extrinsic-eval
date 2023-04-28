import datetime
import json
import pickle
import sys
from typing import Callable

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration

from utils import log_progress
import datasets

language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "liujqian/gpt2-large-finetuned-commongen",
    "gpt2-m": "liujqian/gpt2-medium-finetuned-commongen",
    "gpt2": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-finetuned-commongen",
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
        prompt_generator: Callable[[list[str]], str],
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
            sentences.append(sentence)
        results[i] = {"id": examples["id"][i], "sentences": sentences}
    return results


if __name__ == '__main__':
    qualified_model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
    model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto")
    model_name = qualified_model_name.split("/")[-1]
    for subset_name in ["train", "validation"]:
        new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
        generations = generate_without_choice_content_words(
            model,
            tokenizer,
            lambda l: f'Write a sentence with the given words: {", ".join(l)}.',
            subset_name,
            new_ds[subset_name]
        )
        print(f"Trying to dump the generations to a file!")
        file = open(
            f'tk_instruct_3b-{subset_name}-WITHOUT-choicewords-noquestionwordlimit.json',
            'w'
        )
        json.dump(generations, file)
        # close the file
        file.close()
        print("Finished dumping the generations to a file!")
