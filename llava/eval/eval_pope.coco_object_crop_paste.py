import os, json
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np

def eval_pope(answers, label_file):
    
    question_list = [json.loads(q) for q in open(label_file, 'r')]
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']
        # only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'
            
    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    tp_stats = np.zeros((12, 12), dtype=np.int64)
    gp_stats = np.zeros((12, 12), dtype=np.int64)

    for qs, pred, label in zip(question_list, pred_list, label_list):
        w_grid_pos, h_grid_pos = int(os.path.split(qs['image'])[1][1:4]) // 28, int(os.path.split(qs['image'])[1][6:9]) // 28
        if label == 1:
            gp_stats[h_grid_pos, w_grid_pos] += 1
        if pred == pos and label == pos:
            TP += 1
            tp_stats[h_grid_pos, w_grid_pos] += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio))

    # 空间分布
    fig, axes = plt.subplots(1, 3, figsize=(25, 10))
    ## TN
    sns.heatmap(tp_stats, annot=True, fmt="d", cmap='viridis', ax=axes[0], cbar=False)
    axes[0].set_title('# True Positive')
    ## GP
    sns.heatmap(gp_stats, annot=True, fmt="d", cmap='viridis', ax=axes[1], cbar=False)
    axes[1].set_title('# Positive')
    ## TNR
    sns.heatmap(tp_stats / gp_stats, annot=True, fmt='.2f', cmap='viridis', ax=axes[2], cbar=False)
    axes[2].set_title('True Positive Rate')
    plt.tight_layout()

    plt.savefig(args.output_jpg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--output-jpg", type=str)
    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file, 'r')]
    
    category = args.question_file[10:-5]
    print('# samples: {}'.format(len(answers)))
    eval_pope(answers, args.question_file)
    print("====================================")
