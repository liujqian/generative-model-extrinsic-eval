import pickle
import random
import re

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
        # prompt user and read a line
        print(f"This is the {i+1}th comparision. There are {sample_cnt} comparisions to be done in total!")
        print(f"The following sentences are generated using the concepts: {generation_dict['concepts'][i]}")
        for sample_idx, sample in enumerate(ith_samples):
            print(f"{sample_idx + 1}) {sample['generation']}")
            print()
        print(
            "Please rank the generations from the worst quality to "
            "the best quality by typing in the indices of the generations, "
            "separated by a white space. "
            "Do not hit enter until you typed out all the four indices.",
        )
        valid_input = False
        while not valid_input:
            nums_in = input()
            match = re.search("(\d)\s*(\d)\s*(\d)\s*(\d)", nums_in)
            if len(match.groups()) != 4:
                print(
                    "You need to enter exactly four values, which corresponds to the sentences' indices."
                    " Please try again!"
                )
                continue
            ranking = [int(y) for y in match.groups()]
            if valid_nums(ranking):
                valid_input = True
        # Here, quality is better with index higher. Ranked models in the asending order.
        for ranking_idx, sample_idx in enumerate(ranking):
            cur_model = ith_samples[sample_idx - 1]["model_name"]
            rankings_each_sample[cur_model].append(ranking_idx)
    return rankings_each_sample


def valid_nums(vec):
    for num in vec:
        if num not in {1, 2, 3, 4}:
            print("You cannot enter an index smaller than 1 or larger than 4. Please try again!")
            return False
    return True


if __name__ == '__main__':
    n = 100
    print("Hi, thank you for taking your time to complete this survey for me."
          "Here, I need to rank the generations from four language models over 100 questions."
          "For each question, the model is asked to generate a sentence using a few words."
          "You need help me rank the models' generations based on if they are fluent, coherent, sensible and covering many concepts prompted."
          "There is no gold standard for weighing all the apsects. You can subjective rank."
          "When you are shown four generations, just type in the indices of the generations separated by a white space in one line. "
          "The order you type indices should be of ascending quality. That is the first index you type in should be one of a sentence you think is the worst."
          "If you see exactly same generations, please give them the same ranking.")
    packed_generations = pack_first_100_generations_in_test_set()
    stats = manual_rank(packed_generations)
    print(stats)
