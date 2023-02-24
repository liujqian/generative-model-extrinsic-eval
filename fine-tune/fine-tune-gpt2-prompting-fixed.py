from transformers import AutoTokenizer, AutoModelForCausalLM
from dataclasses import dataclass
from transformers.tokenization_utils_base import PreTrainedTokenizerBase, PaddingStrategy
from typing import Optional, Union
import torch
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, load_metric

model_names = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
for model_name in model_names:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    # model.config.pad_token_id = model.config.eos_token_id
    datasets = load_dataset('common_gen')


    def process_data(examples):
        concepts = examples["concepts"]
        targets = examples["target"]
        full_string = [", ".join(concepts[i]) + "=" + targets[i] + tokenizer.eos_token
                       for i in
                       range(len(concepts))]
        tokenized = tokenizer(full_string)
        return tokenized


    encoded_datasets = datasets.map(process_data, batched=True,
                                    remove_columns=['concept_set_idx', 'concepts', 'target'])


    def group_texts(
            examples):  # This function is copied directly from the tutorial given by Hugging Face: https://github.com/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
        block_size = 256
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


    lm_datasets = encoded_datasets.map(
        group_texts,
        batched=True,
    )
    # =A wall mounted urinal in a checker tiled rest room.<|endoftext|>lake shore canoe=canoe on a shore of
    # lake.<|endoftext|>lake shore canoe=canoe on shore with rainbow across the lake<|endoftext|>lake shore canoe=Several
    # canoes parked in the grass on the shore of a lake <|endoftext|>mountain skier way=A skier on his way to the
    # mountain.<|endoftext|>mountain skier way=skiers make their way down the mountain<|endoftext|>mountain skier way=A
    # skier making her way down a snowy mountain.<|endoftext|>boat lake drive=driving boat on a lake<|endoftext|>boat
    # lake drive=a boat is being driven through a lake<|endoftext|>boat lake drive=A fisherman drives his boat on the
    # lake<|endoftext|>grass horse eat=A horse is eating grass.<|endoftext|>grass horse eat=The horses are eating
    # grass.<|endoftext|>grass horse eat=The old horse ate grass all day.<|endoftext|>train come track=train coming down
    # the track<|endoftext|>train come track=A train is coming along on a track.<|endoftext|>train come track=a long
    # train in coming down some tracks<|endoftext|>train track move=train moving on the tracks<|endoftext|>train track
    # move=A red train is moving down a track<|endoftext|>train track move=A train moves slowly on some empty
    # tracks<|endoftext|>
    training_args = TrainingArguments(
        f"{model_name}-finetuned-commongen-prompting-fixed",
        evaluation_strategy="epoch",
        learning_rate=2.5e-5,
        weight_decay=0.01,
        push_to_hub=True,
        load_best_model_at_end=True,
        save_strategy="epoch",
        hub_token="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",
        num_train_epochs=2.0,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
    )
    trainer.train()
    trainer.save_model()
