import re

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList

if __name__ == '__main__':
    checkpoint = "stabilityai/stablelm-tuned-alpha-3b"
    system_prompt = """<|SYSTEM|># StableLM Tuned (Alpha version)
- StableLM is a helpful and harmless open-source AI language model developed by StabilityAI.
- StableLM is excited to be able to help the user, but will refuse to do anything that could be considered harmful to the user.
- StableLM is more than just an information source, StableLM is also able to write poetry, short stories, and make jokes.
- StableLM will refuse to participate in anything that could harm a human.
"""


    class StopOnTokens(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
            stop_ids = [50278, 50279, 50277, 1, 0]
            for stop_id in stop_ids:
                if input_ids[0][-1] == stop_id:
                    return True
            return False


    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


    def prompt_generator(l):
        prompt = f"{system_prompt}<|USER|>Create a sentence using the following words: {', '.join(l)}.<|ASSISTANT|>"
        return prompt


    def result_separator(s):
        all_found = re.findall(r"Create a sentence using the following words: [^.]*.([^.?!]*)", s)
        return all_found[0] if len(all_found) != 0 else ""


    inputs = tokenizer(prompt_generator(["mountain", "ski", "snow"]), return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        num_beams=4,
        num_beam_groups=4,
        num_return_sequences=4,
        diversity_penalty=100.0,
        remove_invalid_values=True,
        temperature=1.0,
        max_new_tokens=128,
        return_dict_in_generate=True,
        output_scores=True,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()])
    )
    for output in outputs.sequences:
        sentence = tokenizer.decode(output, skip_special_tokens=True)
        print(result_separator(sentence))
        print("****")
