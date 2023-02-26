import json
import pickle

import datasets, pandas


def generate_validation_reference_coco(subset_name: str):
    m = {
        "info": {
            "description": "",
            "url": "",
            "version": "",
            "year": 2023,
            "contributor": "Jingqian Liu",
            "date_created": ""
        },
        "images": [

        ],
        "type": "captions",
        "annotations": [

        ]
    }
    included_concept_set_idx = {391895}
    commongen_full = datasets.load_dataset('common_gen')
    validation_set = commongen_full[subset_name]
    for i in range(len(validation_set["concept_set_idx"])):
        if validation_set["concept_set_idx"][i] not in included_concept_set_idx:
            included_concept_set_idx.add(validation_set["concept_set_idx"][i])
            m["images"].append(
                {
                    "license": 3,
                    "url": "",
                    "file_name": "NA.jpg",
                    "id": validation_set["concept_set_idx"][i],
                    "width": 640,
                    "date_captured": "2013-11-14 11:18:45",
                    "height": 360
                }
            )
        m["annotations"].append(
            {
                "image_id": validation_set["concept_set_idx"][i],
                "id": i,
                "caption": validation_set["target"][i]
            }
        )
    with open(f"coco-annotations/commongen-{subset_name}-gold-generations.json", "w") as file:
        json.dump(m, file)


def generate_validation_generation_coco(model_name: str):
    file = open(
        f"C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\commongen-validation-test-generation\\{model_name}-validation-commongen-generation.pickle",
        "rb",
    )
    m = []
    generations = pickle.load(file)
    file.close()
    for concept_set_idx, generation in generations.items():
        m.append(
            {
                "image_id": concept_set_idx,
                "caption": generation["sentences"].split("=")[-1]
            }
        )
    with open(f"coco-annotations/commongen-{model_name}-validation-model-generations.json", "w") as file:
        json.dump(m, file)


def generate_test_generations_coco(model_name: str):
    file = open(
        f"C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\commongen-validation-test-generation\\{model_name}-test-commongen-generation.pickle",
        "rb",
    )
    m = []
    generations = pickle.load(file)
    file.close()
    for i, generated_sentence in enumerate(generations["generated_sentences"]):
        m.append(
            {
                "image_id": i,
                "caption": generated_sentence.split("=")[-1]
            }
        )
    with open(f"coco-annotations/commongen-{model_name}-test-model-generations.json", "w") as file:
        json.dump(m, file)


def generate_test_reference_coco(subset_name: str):
    m = {
        "info": {
            "description": "",
            "url": "",
            "version": "",
            "year": 2023,
            "contributor": "Jingqian Liu",
            "date_created": ""
        },
        "images": [

        ],
        "type": "captions",
        "annotations": [
        ]
    }
    included_concept_set_idx = {391895}
    df = pandas.read_csv(
        "C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\gpt-3\\commongen-test-augmented-gpt3-0-1496.csv")
    for i in range(len(df)):
        if df["id"][i] not in included_concept_set_idx:
            included_concept_set_idx.add(df["id"][i])
            m["images"].append(
                {
                    "license": 3,
                    "url": "",
                    "file_name": "NA.jpg",
                    "id": int(df["id"][i]),
                    "width": 640,
                    "date_captured": "2013-11-14 11:18:45",
                    "height": 360
                }
            )
        m["annotations"].append(
            {
                "image_id": int(df["id"][i]),
                "id": i,
                "caption": df["sentence"][i]
            }
        )
    with open(f"coco-annotations/commongen-{subset_name}-bronze-generations.json", "w") as file:
        json.dump(m, file)


if __name__ == '__main__':
    for model_name in ["gpt2", "gpt2-m", "gpt2-l", "gpt2-xl"]:
        print("Generating COCO for " + model_name + " generations!")
        generate_test_generations_coco(model_name)