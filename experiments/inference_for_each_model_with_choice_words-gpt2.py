import datetime
import pickle
import sys

import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

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


def generate_with_choice_content_words(model_name: str, set_name: str, examples):
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[model_name], padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(language_models[model_name])
    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            print(
                str(datetime.datetime.now()) + " making the " + str(
                    i) + "th generation for the " + set_name + " set using the " + model_name + " model. There are " + str(
                    len(examples['id'])) + " examples in total!")
        question_content_words = examples["question_content_words"][i]
        prompts_for_this_question = []
        for choice_idx in range(0, 5):
            choice_content_words = examples[f"choice_{choice_idx}_content_words"][i]
            all_content_words = choice_content_words + question_content_words
            prompt = "<|endoftext|>" + " ".join(all_content_words) + "="
            prompts_for_this_question.append(prompt)
        tokenized_input = tokenizer(prompts_for_this_question, return_tensors="pt", padding=True)
        outputs = model.generate(
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
            "sequences_scores": outputs.sequences_scores.detach().cpu().numpy(),
        }
    return results


if __name__ == '__main__':
    args = sys.argv
    assert len(args) == 2, "Need two arguments."
    assert args[-1] in ["gpt2", "gpt2-m", "gpt2-l", "gpt2-xl"]
    target_model_name = args[-1]
    for subset_name in ["train", "validation"]:
        new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
        generations = generate_with_choice_content_words(target_model_name, subset_name, new_ds[subset_name])
        print(f"Trying to dump the generations to a file!")
        file = open(
            f'{target_model_name}-{subset_name}-withchoicewords-noquestionwordlimit-promptfixed.pickle',
            'wb')
        # dump information to that file
        pickle.dump(generations, file)
        # close the file
        file.close()
        print("Finished dumping the generations to a file!")
