from transformers import AutoTokenizer, AutoModelForCausalLM

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

tokenizer = AutoTokenizer.from_pretrained(tokenizers["gpt2-xl"],padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(language_models["gpt2-xl"])
encoder_input_str0 = "<|endoftext|>mountain, snow, trap="
encoder_input_str1 = "<|endoftext|>see, pick, hat="
tokenized_input = tokenizer( [encoder_input_str1,encoder_input_str0], return_tensors="pt", padding=True)
outputs = model.generate(
    **tokenized_input,
    num_beams=4,
    num_beam_groups=4,
    num_return_sequences=4,
    diversity_penalty=100.0,
    length_penalty=0.1,
    remove_invalid_values=True,
    temperature=10.0,
    max_new_tokens=256,
    return_dict_in_generate=True,
    output_scores=True,
)
print("Output:\n" + 100 * '-')
for output in outputs.sequences:
    print(tokenizer.decode(output, skip_special_tokens=True))
