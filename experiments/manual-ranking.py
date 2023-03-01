import json
import pickle
import random
import re
import sys

n = 100


def pack_first_100_generations_in_test_set():
    test_set_generations = []
    for model_name in [
        "gpt2",
        "gpt2-m",
        "gpt2-l",
        "gpt2-xl"
    ]:
        with open(
                f"C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\commongen-validation-test-generation\\{model_name}-test-commongen-generation.pickle",
                "rb",
        ) as handle:
            test_set_generations.append(pickle.load(handle))
    generations_dict = {
        "concepts": test_set_generations[0]["concepts"][:n],
        "gpt2": test_set_generations[0]["generated_sentences"][:n],
        "gpt2-m": test_set_generations[1]["generated_sentences"][:n],
        "gpt2-l": test_set_generations[2]["generated_sentences"][:n],
        "gpt2-xl": test_set_generations[3]["generated_sentences"][:n]
    }
    return generations_dict


def manual_rank(generation_dict):
    sample_cnt = len(generation_dict["concepts"])
    rankings_each_sample = {
        "gpt2": [],
        "gpt2-m": [],
        "gpt2-l": [],
        "gpt2-xl": []
    }
    for i in range(sample_cnt):
        print(f"Ranking the {i + 1}th sample now. There are {sample_cnt} total samples to be ranked.")
        ith_samples = [
            {
                "model_name": "gpt2",
                "generation": generation_dict["gpt2"][i]
            },
            {
                "model_name": "gpt2-m",
                "generation": generation_dict["gpt2-m"][i]
            },
            {
                "model_name": "gpt2-l",
                "generation": generation_dict["gpt2-l"][i]
            },
            {
                "model_name": "gpt2-xl",
                "generation": generation_dict["gpt2-xl"][i]
            },
        ]
        random.shuffle(ith_samples)
        ranking = pair_wise_ranking(shuffled_samples=ith_samples)
        ranking.reverse()
        rank = 4
        for level in ranking:
            for generation in level:
                rankings_each_sample[generation["model_name"]].append(rank)
            rank -= len(level)
    return rankings_each_sample


def pair_wise_ranking(shuffled_samples):
    assert len(shuffled_samples) >= 2, f"pair_wise_ranking receives a list with {len(shuffled_samples)} data points."
    less = []
    equal = [shuffled_samples[0]]
    greater = []
    pivot = shuffled_samples[0]
    for i in range(1, len(shuffled_samples)):
        res = compare(pivot, shuffled_samples[i])
        if res == ">":
            less.append(shuffled_samples[i])
        elif res == "<":
            greater.append(shuffled_samples[i])
        else:
            equal.append(shuffled_samples[i])
    result = []
    if len(less) != 0:
        if len(less) >= 2:
            result.extend(pair_wise_ranking(less))
        else:
            result.append(less)
    result.append(equal)
    if len(greater) != 0:
        if len(greater) >= 2:
            result.extend(pair_wise_ranking(greater))
        else:
            result.append(greater)
    return result


def compare(sample_a, sample_b):
    concepts = sample_a["generation"].split("=")[0]
    assert sample_a["generation"].split("=")[0] == sample_b["generation"].split("=")[
        0], "The concepts of the two samples being compared are not the same."
    print("The concepts used for these generations are: " + str(concepts))
    print("Which do you think is a better generation?")
    print("1) " + sample_a["generation"].split("=")[1])
    print("or")
    print("2) " + sample_b["generation"].split("=")[1])
    res = input(
        "Enter > if you think 1) is better, < if you think 2) is better, = if the two generations are the same!")
    while res.strip() not in [">", "<", "="]:
        print("Invalid input. Please only enter one of <, >, =.")
        res = input(
            "Enter > if you think 1) is better, < if you think 2) is better, = if the two generations are the same!")
    print("\n" * 10)
    return res


def valid_nums(vec):
    for num in vec:
        if num not in {1, 2, 3, 4}:
            print("You cannot enter an index smaller than 1 or larger than 4. Please try again!")
            return False
    return True


if __name__ == '__main__':
    n = 100
    packed_generations = pack_first_100_generations_in_test_set()
    stats = manual_rank(packed_generations)
    print(stats)
    with open(f"analysis-results/manual-analysis-on-commongen-test-set-generations.json", "w") as file:
        json.dump(stats, file)
