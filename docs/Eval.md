# Evaluation

We evaluate models on a diverse set of 8 benchmarks, including `POPE`, `CHAIR`, `AMBER` for visual hallucination tasks, and some LVLM multiple-choice benchmarks with lmms-eval.

## POPE

There are 27,000 questions in total from `COCO`, `GQA` and `A-OKVQA`, each of which comprises three subsets, `random`, `popular`, `adversarial`. For our eval scripts, we put them under folder `playground/data`. 
```
POPE_HOME=/path/to/POPE
mkdir -p playground/data/pope
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
You can run these evaluations separately, by specifying `data` and `subset`, or run them all with one script.
```bash
# run separately, data: coco, gqa, aokvqa, subset: ran, pop, adv
bash scripts/v1_5/eval.cca-llava-1.5-7b.pope.${data}.${subset}.sh

# one script for all
bash scripts/v1_5/eval.cca-llava-1.5-7b.pope.sh
```

## CHAIR

We follow [OPERA](https://github.com/shikiw/OPERA) to set up our CHAIR evaluation. An independent conda env is set up for chair evaluation as version for `transformers` differs. 

```bash
conda create --name cca-llava-chair --clone cca-llava
conda activate cca-llava-chair
pip install transformers==4.29.2
```

First, make sure that coco images and annotations are prepared under folder `playground` and organised in this way,
```
playground/
└── data
    └── coco
        ├── annotations
        │   └── instances_val2014.json
        └── val2014
```

To do chair evaluation, simply run
```
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.chair.sh
```
## AMBER

## Multiple-Choice Benchmarks
For multiple-choice benchmarks, we use [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) to do evaluations. You can simply run scripts below, by replacing `data` with `seed`, `vizwiz_vqa`, `scienceqa_img`, `mmstar` or `gqa`.
```
bash scripts/v1_5/eval/eval.cca-llava-1.5-7b.${data}.sh
```
