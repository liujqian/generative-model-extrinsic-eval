from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration


def tk_instruct_3b_def():
    tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def").to(0)
    prompt_generator = lambda \
            l: f'Definition: Write a sentence with the given words. Now complete the following example - Input: {", ".join(l)}. Output:'
    return model, tokenizer, prompt_generator


def flan_t5_xl():
    qualified_model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
    model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto")
    prompt_generator = lambda l: f'Write a sentence with the given words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator
