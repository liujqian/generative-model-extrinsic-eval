import datasets
import spacy
from spacy.matcher import Matcher
import huggingface_hub

nlp = spacy.load("en_core_web_lg")


# This sub-routine is based on the answer provided by the stackoverflow user mdmjsh on May 27, 2020. The answer is accessible by the link blow.
# https://stackoverflow.com/a/62038950
def extract_content_words(text: str) -> [str]:
    doc = nlp(text)
    pattern = [
        [
            {'POS': 'NOUN', 'OP': '{1}'},
        ],
        [
            {'POS': 'VERB', 'OP': '{1}'},
        ],
        [
            {'POS': 'ADJ', 'OP': '{1}'},
        ],
        [
            {'POS': 'ADV', 'OP': '{1}'},
        ]
    ]
    matcher = Matcher(nlp.vocab)
    matcher.add("phrases", pattern)
    found_matches = matcher(doc)
    spans = [doc[start:end] for _, start, end in found_matches]
    words = [" ".join([token.lemma_ for token in span]) for span in spans]
    dedup = {}  # This method for deduplication is found on stackoverflow https://stackoverflow.com/a/6765391.
    new_list = [dedup.setdefault(x, x) for x in words if x not in dedup]
    if len(new_list) == 0:
        return [text]
    return new_list


def extract_commonsenseqa_terms():
    ds = datasets.load_dataset("commonsense_qa")
    new_ds = ds.map(
        function=lambda x: {
            "question_content_words": extract_content_words(x["question"]),
            "choice_0_content_words": extract_content_words(x["choices"]["text"][0]),
            "choice_1_content_words": extract_content_words(x["choices"]["text"][1]),
            "choice_2_content_words": extract_content_words(x["choices"]["text"][2]),
            "choice_3_content_words": extract_content_words(x["choices"]["text"][3]),
            "choice_4_content_words": extract_content_words(x["choices"]["text"][4]),
        }
    )
    return new_ds


def extract_and_upload():
    huggingface_hub.login("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")
    new_ds = extract_commonsenseqa_terms()
    new_ds.save_to_disk("./commonsenseqa_with_content_words")
    new_ds.push_to_hub("liujqian/commonsenseqa_with_content_words")


def count_average_number_concepts_per_prompt() -> float:
    dataset = datasets.load_dataset("liujqian/commonsenseqa_with_content_words")
    sets = ["train", "validation"]
    num_concepts = []
    for set in sets:
        for i in range(len(dataset[set])):
            num_question_concepts = len(dataset[set][i]['question_content_words'])
            for choice in range(5):
                num_choice_concepts = len(dataset[set][i][f"choice_{choice}_content_words"])
                num_concepts.append(num_choice_concepts + num_question_concepts)
    return sum(num_concepts) / len(num_concepts)


if __name__ == '__main__':
    avg_concepts = count_average_number_concepts_per_prompt()
    print(avg_concepts)
