import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, \
    AutoModelForSeq2SeqLM

language_models = {
    "gpt2-xl": "liujqian/gpt2-xl-finetuned-commongen",
    "gpt2-l": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-large-finetuned-commongen",
    "gpt2-m": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-medium-finetuned-commongen",
    "gpt2": "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\fine-tune\\gpt2-finetuned-commongen",
    "t5": "mrm8488/t5-base-finetuned-common_gen",
    "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}

tokenizers = {
    "gpt2-xl": "gpt2-xl",
    "gpt2-l": "gpt2-l",
    "gpt2-m": "gpt2-m",
    "gpt2": "gpt2",
    "t5": "mrm8488/t5-base-finetuned-common_gen",
    "bloom": "mrm8488/bloom-560m-finetuned-common_gen"
}


def result_separator(output: str, content_words: list[str]) -> str:
    return output.split(sep=f'Create a sentence with the following words: {", ".join(content_words)}.')[-1].strip(
        ' \t\n\r')


if __name__ == '__main__':

    checkpoint = "bigscience/bloomz-3b"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    inputs = tokenizer(
        "Create a sentence with the following words: ski, mountain, fall.",
        return_tensors="pt"
    ).to("cuda")
    # def tokenizer_wrapper(*args, **kwargs)
    outputs = model.generate(
        **inputs,
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
    for output in outputs.sequences:
        sentence = tokenizer.decode(output, skip_special_tokens=True)
        print(result_separator(sentence, ["ski", "mountain", "fall"]))
