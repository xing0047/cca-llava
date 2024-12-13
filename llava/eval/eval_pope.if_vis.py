import os
import json
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def eval_pope(answers, label_file):
    
    if 'coco' in args.question_file:
        question_list = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    else:
        question_list = json.load(open(args.question_file))

    label_list = [q['label'] for q in question_list]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
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
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
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
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )

def draw_attn(answers, question_file, result_png):
    txt_img_ifs = []
    for line in answers:
        txt_img_if = line["text_image_if"]
        txt_img_if = torch.tensor(txt_img_if)
        txt_img_ifs.append(txt_img_if)

    txt_img_ifs = torch.stack(txt_img_ifs)
    txt_img_ifs = torch.mean(txt_img_ifs, dim=0)
    txt_img_ifs = txt_img_ifs.reshape(24, 24).detach().cpu().numpy()

    txt_img_if_max = txt_img_ifs.max()
    txt_img_if_min = txt_img_ifs.min()
    norm_txt_img_if = (txt_img_ifs - txt_img_if_min) / (txt_img_if_max - txt_img_if_min)

    vmin = norm_txt_img_if.min()
    vmax = norm_txt_img_if.max()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    plt.imshow(norm_txt_img_if, cmap="viridis", interpolation='nearest', norm=norm)
    plt.colorbar()
    plt.axis('off')

    plt.savefig(result_png)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)
    parser.add_argument("--result-png", type=str)
    args = parser.parse_args()
    answers = [json.loads(q) for q in open(args.result_file, 'r')]
    
    category = args.question_file[10:-5]
    print('# samples: {}'.format(len(answers)))
    eval_pope(answers, args.question_file)
    print("====================================")
    draw_attn(answers, args.question_file, args.result_png)
    print('# information flow: {}'.format(args.result_png))
