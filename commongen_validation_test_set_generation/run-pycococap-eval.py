import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from rouge import Rouge
from rouge_score import rouge_scorer

from commongen_validation_test_set_generation.language_models import get_language_models


def calculate_rouge(rouge_type: str) -> dict:
    if rouge_type not in {"rouge2", "rougeL"}:
        raise Exception("Invalid rouge type")
    scorer = rouge_scorer.RougeScorer([rouge_type], use_stemmer=True)
    for model_name in get_language_models():
        for subset_name in ["validation"]:
            print(f"Evalutating {model_name} on the performance of {subset_name}")
            results_file = f'coco-annotations/commongen-{subset_name}-{model_name}-generations.json'
            annotation_file = 'coco-annotations/commongen-validation-gold-references.json' if subset_name == "validation" else "coco-annotations/commongen-test-silver-references.json"

            # create coco object and coco_result object
            coco = COCO(annotation_file)
            coco_result = coco.loadRes(results_file)
            imgIds = coco_result.getImgIds()
            gts = {}
            res = {}
            for imgId in imgIds:
                gts[imgId] = coco.imgToAnns[imgId]
                res[imgId] = coco_result.imgToAnns[imgId]

            all_best_scores = []
            for idx in gts.keys():
                hypo = res[idx]
                candidate = hypo[0]["caption"]
                ref = gts[idx]
                scores = []
                for reference in ref:
                    ref_caption = reference["caption"]
                    all_score = scorer.score(target=ref_caption, prediction=candidate)
                    target_score = all_score[rouge_type].fmeasure if rouge_type == "rougeL" else all_score[rouge_type].recall
                    scores.append(target_score)
                best_score = max(scores)
                all_best_scores.append(best_score)
            avg_rouge_score = sum(all_best_scores) / len(all_best_scores)
            with open(f"pycocoevalcap-results/{model_name}-commongen-{subset_name}-autoeval.json", "r") as file:
                original_json = json.load(file)
            original_json["google" + rouge_type] = avg_rouge_score
            with open(f"pycocoevalcap-results/{model_name}-commongen-{subset_name}-autoeval.json", "w") as file:
                json.dump(original_json, file)


def run_overall_pycoco_eval():
    for model_name in get_language_models():
        for subset_name in [
            "validation",
        ]:
            print(f"Evalutating {model_name} on the performance of {subset_name}")
            results_file = f'coco-annotations/commongen-{subset_name}-{model_name}-generations.json'
            annotation_file = 'coco-annotations/commongen-validation-gold-references.json' if subset_name == "validation" else "coco-annotations/commongen-test-silver-references.json"

            # create coco object and coco_result object
            coco = COCO(annotation_file)
            coco_result = coco.loadRes(results_file)

            # create coco_eval object by taking coco and coco_result
            coco_eval = COCOEvalCap(coco, coco_result)

            # evaluate on a subset of images by setting
            # coco_eval.params['image_id'] = coco_result.getImgIds()
            # please remove this line when evaluating the full validation set
            coco_eval.params['image_id'] = coco_result.getImgIds()

            # evaluate results
            # SPICE will take a few minutes the first time, but speeds up due to caching
            coco_eval.evaluate()

            # print output evaluation scores
            # for metric, score in coco_eval.eval.items():
            #     print(f'{metric}: {score:.3f}')
            with open(f"pycocoevalcap-results/{model_name}-commongen-{subset_name}-autoeval.json", "w") as file:
                json.dump(coco_eval.eval, file)


if __name__ == '__main__':
    for rouge_type in {"rougeL", "rouge2"}:
        calculate_rouge(rouge_type)
    # run_overall_pycoco_eval()
