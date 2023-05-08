import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModelForCausalLM
import re


# If not specified specifically, inferences are done on a single specified GPU.

# Inferenced on RTX 3090
def tk_instruct_3b_def():
    tokenizer = AutoTokenizer.from_pretrained("allenai/tk-instruct-3b-def")
    model = AutoModelForSeq2SeqLM.from_pretrained("allenai/tk-instruct-3b-def").to(0)
    prompt_generator = lambda \
            l: f'Definition: Write a sentence with the given words. Now complete the following example - Input: {", ".join(l)}. Output:'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def flan_t5_xl():
    qualified_model_name = "google/flan-t5-xl"
    tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
    model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto")
    prompt_generator = lambda l: f'Write a sentence with the given words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def flan_t5_large():
    qualified_model_name = "google/flan-t5-large"
    tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
    model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto")
    prompt_generator = lambda l: f'Write a sentence with the given words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


def flan_t5_xxl(cache_dir: str = ""):
    qualified_model_name = "google/flan-t5-xxl"
    if cache_dir == "":
        tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
        model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto",
                                                           torch_dtype=torch.float16)
    else:
        tokenizer = T5Tokenizer.from_pretrained(qualified_model_name, cache_dir=cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto", cache_dir=cache_dir,
                                                           torch_dtype=torch.float16)
    prompt_generator = lambda l: f'Write a sentence with the given words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def t0_3b():
    tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
    model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B").to("cuda")
    prompt_generator = lambda l: f'Create a sentence with the following words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


# Inferenced on RTX 3090
def bloomz(size: str):
    assert size in ["3b", "1b7", "1b1", "560m"], "The given size is not expected."
    checkpoint = f"bigscience/bloomz-{size}"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    prompt_generator = lambda l: f'Create a sentence with the following words: {", ".join(l)}.'

    def result_separator(output: str) -> str:
        r = r"Create a sentence with the following words: [^\.]*\."
        return re.split(r, output)[-1].strip(' \t\n\r')

    return model, tokenizer, prompt_generator, result_separator


def dolly_v1_6b():
    tokenizer = AutoTokenizer.from_pretrained("databricks/dolly-v1-6b", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        "databricks/dolly-v1-6b",
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    PROMPT_FORMAT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

    ### Instruction:
    {instruction}

    ### Response:
    """
    prompt_generator = lambda l: PROMPT_FORMAT.format(
        instruction=f'Create a sentence with the following words: {", ".join(l)}.'
    )

    def result_separator(output: str, content_words) -> str:
        return output.split(sep=f'Create a sentence with the following words: {", ".join(content_words)}.')[-1].strip(
            ' \t\n\r')

    return model, tokenizer, prompt_generator, result_separator
