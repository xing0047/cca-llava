# Evaluation

We evaluate models on a diverse set of 8 benchmarks, including `POPE`, `CHAIR`, `AMBER` for visual hallucination tasks, and some LVLM multiple-choice benchmarks with lmms-eval.

## POPE

There are 27,000 questions in total from `COCO`, `GQA` and `A-OKVQA`, each of which comprises three subsets, `random`, `popular`, `adversarial`. For our eval scripts, we put them under folder `playground/data`. 
```
POPE_HOME=/path/to/POPE
ln -s ${POPE_HOME}/output playground/data/pope
```
Before running any eval script, make sure that `playground` is organised this way,
```
playground/
└── data
    ├── coco
    │   └── val2014
    ├── gqa
    │   └── images
    └── pope
        ├── coco
        └── seem
            ├── aokvqa
            └── gqa
```
You can run pope evaluations separately, by specifying `data` and `subset`, or run them all with one script.
```bash
# run separately, data: coco, gqa, aokvqa, subset: ran, pop, adv.
bash scripts/v1_5/eval.cca-llava-1.5-7b.pope.${data}.${subset}.sh

# one script for all.
bash scripts/v1_5/eval.cca-llava-1.5-7b.pope.sh
```

## CHAIR

We follow [OPERA](https://github.com/shikiw/OPERA) to set up our CHAIR evaluation. An independent conda env is needed for chair evaluation as version for `transformers` differs. 

```bash
conda create --name cca-llava-chair --clone cca-llava
conda activate cca-llava-chair
pip uninstall transformers
pip install transformers==4.29.2
pip install nltk==3.9.1
python -m nltk.downloader all
```

First, make sure that coco images and [annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip) are prepared under folder `playground` and organised in this manner,
```
playground/
└── data
    ├── coco
    │   ├── annotations
    │   │   └── instances_val2014.json
    │   └── val2014
    └── coco_chair.json

```

To do chair evaluation, simply run
```bash
# cca-llava evaluation
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.chair.short.sh
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.chair.long.sh

# llava-v1.5-7b baseline evaluation
bash scripts/v1_5/eval/eval.llava-1.5-7b.chair.short.sh
bash scripts/v1_5/eval/eval.llava-1.5-7b.chair.long.sh
```
## AMBER

We follow [AMBER](https://github.com/junyangwang0410/AMBER) to set up the evaluation. Additional packages need to be installed:
```bash
pip install nltk==3.9.1 spacy==3.8.2
python -m nltk.downloader all
python -m spacy download en_core_web_lg
```

Prepare AMBER images and make sure that data is organised in the following structure:
```
playground/
└── data
    └── amber
        ├── annotations.json
        ├── relation.json
        ├── metrics.txt
        ├── safe_words.txt
        ├── query
        │   ├── query_generative.json
        │   └── query_discriminative.json
        └── image
```
```bash
# run evaluation for cca-llava model
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.amber.sh

# run evaluation for llava model
bash scripts/v1_5/eval/eval.llava-1.5-7b.amber.sh
```

## Multiple-Choice Benchmarks
For multiple-choice benchmarks, we use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to do evaluations. You may start with installation of `lmms-eval`. We integrate `cca` in `lmms-eval/lmms_eval/models/llava.py` at line `35-38`.
```
pip install -e lmms-eval
```
Then, you can simply run scripts below, by replacing `data` with `seed`, `vizwiz_vqa`, `scienceqa_img`, `mmstar` or `gqa`.
```
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.${data}.sh
```
