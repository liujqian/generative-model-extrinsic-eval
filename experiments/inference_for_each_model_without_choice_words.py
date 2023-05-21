import json
from typing import Callable
import datasets
import sys

from utils import log_progress
import models


from accelerate import Accelerator
import torch


models_map = {
    "mt0": models.mt0,
    "flan_t5": models.flan_t5,
}


def generate_without_choice_content_words(
        model,
        tokenizer,
        prompt_generator,
        result_separator,
        set_name: str,
        examples: datasets.Dataset
):
    # Initialize the accelerator
    accelerator = Accelerator(devices=range(torch.cuda.device_count()))

    # Prepare the model and tokenizer using the accelerator
    model, tokenizer = accelerator.prepare(model, tokenizer)
    model.to(accelerator.device)
            
    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            log_progress(i, len(examples['id']),
                f"Generating inferences for the {set_name} set. WITH choice content words.")

        question_content_words = examples["question_content_words"][i]
        prompts_for_this_question = []
        for choice_idx in range(0, 5):
            choice_content_words = examples[f"choice_{choice_idx}_content_words"][i]
            all_content_words = choice_content_words + question_content_words
            prompt = prompt_generator(all_content_words)
            prompts_for_this_question.append(prompt)
        tokenized_input = tokenizer(prompts_for_this_question, return_tensors="pt", padding=True).to(accelerator.device)

        outputs = accelerator.unwrap(model).generate(
            **tokenized_input,
            num_beams=4,
            num_beam_groups=4,
            num_return_sequences=4,
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

        results[i] = {
            "id": examples["id"][i],
            "sentences": sentences,
            "sequences_scores": outputs.sequences_scores.detach().cpu().numpy().tolist(),
        }

    return results


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Give a model size")
        sys.exit(1)

    model_name_and_size = sys.argv[1]
    name, size = model_name_and_size.split("-")

    model, tokenizer, prompt_generator = models_map[name](size)
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
            f'generated_sentences/{name}-{size}-{subset_name}-WITHOUT-choicewords-noquestionwordlimit.json',
            'w'
        )
        json.dump(generations, file)
        # close the file
        file.close()
        print("Finished dumping the generations to a file!")
