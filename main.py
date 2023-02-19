from transformers import AutoTokenizer, AutoModelWithLMHead
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelWithLMHead.from_pretrained("./fine-tune/GPT2-finetuned-commongen")
encoder_input_str = "<|endoftext|>airport, airplane, wait="
input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
outputs = model.generate(
    input_ids,
    num_beams=10,
    num_return_sequences=10,
    remove_invalid_values=True,
    num_beam_groups=10,
    diversity_penalty =100.0,
    length_penalty=0.1,
    temperature=10.0
)
print("Output:\n" + 100 * '-')
for output in outputs:
    print(tokenizer.decode(output, skip_special_tokens=True))