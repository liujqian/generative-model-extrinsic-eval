from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("./fine-tune/gpt2-large-finetuned-commongen")
encoder_input_str = "<|endoftext|> feel, after, doing, housework, hours="
input = tokenizer(encoder_input_str, return_tensors="pt")
outputs = model.generate(
    **input,
    # num_beams=10,
    # num_beam_groups=10,
    # num_return_sequences=10,
    # diversity_penalty=100.0,
    length_penalty=0.1,
    remove_invalid_values=True,
    temperature=10.0,
    max_new_tokens=256,
    return_dict_in_generate=True,
    output_scores=True,
)
print(type(outputs))
print("Output:\n" + 100 * '-')
for output in outputs.sequences:
    print(tokenizer.decode(output, skip_special_tokens=True))
