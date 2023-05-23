import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModelForCausalLM


# Inferenced on RTX 3090
def tk_instruct_3b_def():
    tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "allenai/tk-instruct-3b-def").to(0)
    prompt_generator = lambda \
        l: f'Definition: Write a sentence with the given words. Now complete the following example - Input: {", ".join(l)}. Output:'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def flan_t5_xl():
    qualified_model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
    model = T5ForConditionalGeneration.from_pretrained(
        qualified_model_name, device_map="auto")
    def prompt_generator(
        l): return f'Write a sentence with the given words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def t0_3b():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "bigscience/T0_3B").to("cuda")
    def prompt_generator(
        l): return f'Create a sentence with the following words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


def dolly_v1_6b(conten_words=None):
    tokenizer = AutoTokenizer.from_pretrained(
        "databricks/dolly-v1-6b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v1-6b",
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
    )
    PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """

    def prompt_generator(l): return PROMPT_FORMAT.format(
        instruction=f'Create a sentence with the following words: {", ".join(l)}.'
    )

    def result_separator(output: str, content_words) -> str:
        return output.split(sep=f'Create a sentence with the following words: {", ".join(content_words)}.')[-1].strip(
            ' \t\n\r')

    return model, tokenizer, prompt_generator, result_separator


def mt0_large():
    checkpoint = "bigscience/mt0-large"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="auto")
    def prompt_generator(
        l): return f'Create a sentence with the following words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


def mt0_base():
    checkpoint = "bigscience/mt0-base"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        checkpoint, torch_dtype="auto", device_map="auto")

    def prompt_generator(
        l): return f'Create a sentence with the following words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator
