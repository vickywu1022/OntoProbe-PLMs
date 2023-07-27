import torch
import jsonlines
from get_memorized import split_by_Acc

def evaluate_batch(Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr, logits, batch_gt):
    Recall_1 += Recall_at_k(1, logits, batch_gt)
    Recall_5 += Recall_at_k(5, logits, batch_gt)
    Recall_10 += Recall_at_k(10, logits, batch_gt)
    Recall_25 += Recall_at_k(25, logits, batch_gt)
    Recall_50 += Recall_at_k(50, logits, batch_gt)
    mrr_avg += MRR_avg(logits, batch_gt)
    mrr += MRR(logits, batch_gt)

    return Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr

def MRR_avg(logits, ground_truth):
    score = 0.0
    result = torch.argsort(logits, dim=1, descending=True)
    result = torch.argsort(result, dim=1)
    for i in range(len(ground_truth)):
        indices = ground_truth[i]
        score += 1 / (torch.mean(result[i][indices].float()) + 1)

    return score

def MRR(logits, ground_truth):
    score = 0.0
    result = torch.argsort(logits, dim=1, descending=True)
    result = torch.argsort(result, dim=1)
    for i in range(len(ground_truth)):
        indices = ground_truth[i]
        score += 1 / (torch.min(result[i][indices].float()) + 1)

    return score

def Recall_at_k(k, logits, ground_truth):
    score = 0.0
    pred = torch.topk(logits, k, dim=1)
    indices = pred.indices.tolist()
    for i in range(len(ground_truth)):
        pred_cand = [indices[i][j] for j in range(k)]
        gts = ground_truth[i]
        for gt in gts:
            if gt in pred_cand:
                score += 1.0
                break
    return score

def evaluate_batch_mul_choice(acc, cands_file, logits, batch_gt, cnt, batch_size, cand_ids, save_name):
    '''
    Use a subset of candidates. 
    Take the finest-grained gold as ground truth. 
    Calcaulate Accuracy.
    '''
    with jsonlines.open(cands_file, "r") as reader:
        data = list(reader)

    if '02' in save_name:
        selected = [data[i] for i in split_by_Acc("memorized_gpt/{}/{}.txt".format(cands_file.split('/')[-1].replace(".jsonl",""), save_name.split('/')[1]))[2]]
        data = selected
    elif '12' in save_name:
        selected = [data[i] for i in split_by_Acc("memorized_gpt/{}/{}.txt".format(cands_file.split('/')[-1].replace(".jsonl",""), save_name.split('/')[1]))[1]]
        data = selected
        
    for i in range(len(batch_gt)):
        index = [cand_ids[c] for c in data[cnt * batch_size + i]["cands"]]
        pred_cand = logits[i][index]
        max_index = torch.argmax(pred_cand)
        gt = batch_gt[i][0]
        # print(gt, index[max_index], index)
        if gt == index[max_index]:
            acc += 1.0
    return acc