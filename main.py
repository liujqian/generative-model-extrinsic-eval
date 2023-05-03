imeport torch
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
    return output.split(sep=f'Create a sentence with the following words: {", ".join(content_words)}.')[-1].strip(' \t\n\r')


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("databricks/dolly-v1-6b", device_map="auto", trust_remote_code=True,torch_dtype=torch.float16, )
    PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    prompt_generator = PROMPT_FORMAT.format(
        instruction=f'Create a sentence with the following words: {", ".join(["kid", "dance", "room"])}.'
    )
    input_ids = tokenizer(prompt_generator, return_tensors="pt").input_ids.to("cuda")
    outputs = model.generate(
        input_ids,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.encode("### End")[0],
        max_new_tokens=256,
        return_dict_in_generate=True,
        num_beams=20,
        num_beam_groups=20,
        num_return_sequences=20,
        diversity_penalty=100.0,
        output_scores=True,
    )
    for output in outputs.sequences:
        sentence = tokenizer.decode(output, skip_special_tokens=True)
        print(result_separator(sentence,["kid", "dance", "room"]))
