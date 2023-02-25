import datetime
import pickle

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


def generate_without_choice_content_words(model: str, set_name: str, examples):
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[model])
    model = AutoModelForCausalLM.from_pretrained(language_models[model])
    results = {}
    for i in range(len(examples["id"])):
        if i % 20 == 0:
            print(
                str(datetime.datetime.now()) + " making the " + str(i) + "th generation for the " + set_name + " set.")
        question_content_words = examples["question_content_words"][i]
        prompt = "<|endoftext|>" + " ".join(question_content_words) + "="
        tokenized_input = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(
            **tokenized_input,
            num_beams=20,
            num_beam_groups=20,
            num_return_sequences=20,
            diversity_penalty=100.0,
            length_penalty=0.1,
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


target_model_name = "gpt2"
for subset_name in ["train", "validation"]:
    new_ds = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
    generations = generate_without_choice_content_words(target_model_name, subset_name, new_ds[subset_name])
    print(f"Trying to dump the generations to a file!")
    file = open(
        f'{target_model_name}-{subset_name}-WITHOUTchoicewords-noquestionwordlimit-promptfixed.pickle',
        'wb')
    # dump information to that file
    pickle.dump(generations, file)
    # close the file
    file.close()
    print("Finished dumping the generations to a file!")
