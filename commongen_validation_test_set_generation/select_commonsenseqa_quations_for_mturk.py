import json
from typing import List, Dict, Any

import datasets


class CombinedCommongenValTest:
    cg = datasets.load_dataset('common_gen')

    def __iter__(self):
        self.cur_dataset = "test"
        self.cur_index = 0
        return self

    def __next__(self):
        if self.cur_index >= len(self.cg[self.cur_dataset]):
            # if self.cur_dataset == "validation":
            #     self.cur_dataset = "test"
            #     self.cur_index = 0
            # else:
            raise StopIteration
        res = (self.cur_dataset, self.cur_index,)
        self.cur_index += 1
        return res


def select_n_questions_maxing_diversity(n: int) -> list[dict[str, str]]:
    cg = datasets.load_dataset('common_gen')
    selected = []
    concepts_cnt = {}
    cur_minimum_repeat_target = 0

    while len(selected) <= n:
        found = False
        for item in iter(CombinedCommongenValTest()):
            concepts = cg[item[0]][item[1]]["concepts"]
            concept_set_id = cg[item[0]][item[1]]["concept_set_idx"]
            cur_repeats = 0
            for concept in concepts:
                cur_repeats += concepts_cnt[concept] if concept in concepts_cnt else 0
            if cur_repeats > cur_minimum_repeat_target:
                continue
            else:
                selected.append(
                    {
                        "id": f"{item[0]}-{concept_set_id}",
                        "concepts": ", ".join(concepts)
                    }
                )
                for concept in concepts:
                    concepts_cnt[concept] = concepts_cnt[concept] + 1 if concept in concepts_cnt else 1
                found = True
                break
        if not found:
            cur_minimum_repeat_target += 1
            print(
                f"Currently found {len(selected)} questions. Increasing the minimum repeat count to {cur_minimum_repeat_target}!")
    return selected


if __name__ == '__main__':
    with open("generated_sentences/new_selected_commongen_tasks_greedy_diversity.json", "w") as handle:
        json.dump(select_n_questions_maxing_diversity(400), handle)
