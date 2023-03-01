import json

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

for model_name in ["gpt2", "gpt2-m", "gpt2-l", "gpt2-xl"]:
    for subset_name in [
        # "validation",
        "test",
    ]:
        print(f"Evalutating {model_name} on the performance of {subset_name}")
        results_file = f'C:\\Users\\Jingqian\\PycharmProjects\\huggingface-playground\\experiments\\coco-annotations\\commongen-{model_name}-{subset_name}-model-generations.json'
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
        with open(f"pycocoevalcap-results/{model_name}-commongen-{subset_name}-chatgpt-autoeval.json", "w") as file:
            json.dump(coco_eval.eval, file)
