import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer, T5ForConditionalGeneration, \
    AutoModelForCausalLM, StoppingCriteria
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


# Inferenced on four RTX 3090 for the half precision model.
def flan_t5_xxl(cache_dir: str = ""):
    qualified_model_name = "google/flan-t5-xxl"
    if cache_dir == "":
        tokenizer = T5Tokenizer.from_pretrained(qualified_model_name)
        model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto")
    else:
        tokenizer = T5Tokenizer.from_pretrained(qualified_model_name, cache_dir=cache_dir)
        model = T5ForConditionalGeneration.from_pretrained(qualified_model_name, device_map="auto",
                                                           cache_dir=cache_dir)
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


def mt0(size: str):
    assert size in ["mt0-small", "mt0-base", "mt0-large", "mt0-xl"], "The given size is not expected."
    checkpoint = f"bigscience/{size}"
    if size != "mt0-xl":
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.float16)
    prompt_generator = lambda l: f'Create a sentence with the following words: {", ".join(l)}.'
    return model, tokenizer, prompt_generator


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


def redpajama_incite_instruct_3b_v1():
    def result_separator(output: str, ) -> str:
        return output.split("Output:")[1].split("Input")[0].strip()

    def prompt_generator(l):
        return f'''Create a sentence with the given words.

Input: {", ".join(l)}.
Output:'''

    checkpoint = "togethercomputer/RedPajama-INCITE-Instruct-3B-v1"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    return model, tokenizer, prompt_generator, result_separator


def stablelm_tuned_alpha_3b():
    checkpoint = "stabilityai/stablelm-tuned-alpha-3b"
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""

    def prompt_generator(l):
        prompt = f"{system_prompt}<|USER|>Create a sentence using the following words: {', '.join(l)}.<|ASSISTANT|>"
        return prompt

    def result_separator(s):
        all_found = re.findall(r"Create a sentence using the following words: [^.]*.([^.?!]*)", s)
        return all_found[0] if len(all_found) != 0 else ""

    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")
    return model, tokenizer, prompt_generator, result_separator, StopOnTokens()
