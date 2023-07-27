from bdb import Breakpoint
from probing import hard_prompt_probing, soft_prompt_probing
from typing import Counter
import argparse
from openprompt.plms import load_plm
from data import load_candidates, DatasetLoader
import torch
import torch.nn as nn
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--knowledge', type=str, default='type')
    parser.add_argument('--model_class', type=str, default='bert')
    parser.add_argument('--model_path', type=str, default='bert-base-cased')
    parser.add_argument('--template', type=str, default=' is a')
    parser.add_argument('--special_token', type=str, default='[MASK]')  #
    parser.add_argument('--mode', type=str, default="zs-f")
    parser.add_argument('--multi_mask', type=str, default="m")
    parser.add_argument('--multi_token_handler', type=str, default="mean")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--mem_at_k', type=int, default=10)
    parser.add_argument('--save_path', type=str, default="result.csv")
    parser.add_argument('--is_demo', type=bool, default=False)
    parser.add_argument('--use_cuda', type=bool, default=False)
    parser.add_argument('--loss', type=str, default="log")
    parser.add_argument('--soft_init', type=str, default="rand")
    parser.add_argument('--trained_soft', type=str, default="none")

    args = parser.parse_args()
    assert args.model_class in ['bert', 'roberta', 'albert', 'gpt', 'gpt2', 't5', 'baseline']
    assert args.trained_soft in ["none", "load"]
    print(args)

    if args.trained_soft == "none":
        save_name = "{}/{}.{}.{}.{}.{}".format(args.knowledge, args.model_path, "zs-f", args.multi_mask, args.multi_token_handler,args.template)
    else:
        save_name = "{}/{}.{}.{}.{}.{}".format(args.knowledge, args.model_path, "fs", args.multi_mask, args.multi_token_handler,args.loss)

    if args.model_class == "baseline":
        plm, tokenizer, model_config, WrapperClass = load_plm("bert", "bert-base-cased")
        candidates = load_candidates(args.knowledge, tokenizer)
        K = [1, 5]   # recall at k
        DatasetLoader = DatasetLoader(args.knowledge, candidates, args.template, args.mode, save_name, args.trained_soft)
        _, gt, special_tokens, template_text = DatasetLoader.load_data()
        import itertools
        import random
        gt_all = list(itertools.chain.from_iterable(gt["train"]))   # [["a"], ["b", "c"]] -> ["a", "b", "c"]
        pred_all = sorted(Counter(gt_all).items(), key=lambda x: x[1], reverse=True)

        for k in K:
            pred = [l[0] for l in pred_all[:k]]
            recall = 0.0
            for golds in gt["test"]:
                for l in golds:
                    if l in pred:
                        recall += 1.0
                        break
            recall /= len(gt["test"])
            print("Recall@{}: {}".format(k, recall))

        mrr = 0
        pred = [l[0] for l in pred_all]
        other = list(set(candidates.values()) - set(pred))
        random.shuffle(other)
        pred = pred + other

        for golds in gt["test"]:
            rank_avg = 0
            for l in golds:
                rank_avg += 1 + pred.index(l)
            rank_avg /= len(golds)
            mrr += 1 / rank_avg
            rank_avg = 0
        print("MRR_avg: {}".format(mrr / len(gt["test"])))

        mrr = 0
        for golds in gt["test"]:
            ranks = []
            for l in golds:
                ranks.append(1 + pred.index(l))
            mrr += 1 / min(ranks)
        print("MRR: {}".format(mrr / len(gt["test"])))
        exit()

    plm, tokenizer, model_config, WrapperClass = load_plm(args.model_class, args.model_path)
    candidates = load_candidates(args.knowledge, tokenizer)

    DatasetLoader = DatasetLoader(args.knowledge, candidates, args.template, args.mode, save_name, args.trained_soft)
    dataset, gt, special_tokens, template_text = DatasetLoader.load_data()

    special_tokens_dict = {'additional_special_tokens': special_tokens}
    tokenizer.add_special_tokens(special_tokens_dict)
    plm.resize_token_embeddings(len(tokenizer))
    
    ######### Create Pseudoword for Reasoning Task #######
    if "rdfs" in args.knowledge:
        emb = torch.Tensor()
        if args.model_class == "bert":
            emb = plm.bert.embeddings.word_embeddings.weight 
        elif args.model_class == "roberta":
            emb = plm.roberta.embeddings.word_embeddings.weight 
        token_id = tokenizer.convert_tokens_to_ids(args.special_token)
        emb = torch.Tensor(emb).detach()
        spec_token_emb = emb[token_id]
        other_emb = torch.cat((emb[:token_id], emb[token_id+1:-len(special_tokens)]),dim=0)     # compare to all other words except args.special_token and added special tokens with randomized embeddings to be replaced
        pairwise_dist = nn.PairwiseDistance(p=2)    # Euclidian Distance
        dist = pairwise_dist(spec_token_emb.repeat(other_emb.size()[0],1), other_emb)
        min_dist = torch.min(dist)
        if int(torch.argmin(dist)) < token_id: 
            print("Distance between [MASK] and other words in {}: Min=".format(args.model_path), min_dist, tokenizer.convert_ids_to_tokens(int(torch.argmin(dist))))   # nearest word from [mask]
        else:
            print("Distance between [MASK] and other words in {}: Min=".format(args.model_path), min_dist, tokenizer.convert_ids_to_tokens(int(torch.argmin(dist)) + 1))   # nearest word from [mask]
        sample_dist = min_dist * args.threshold
        torch.manual_seed(args.seed)
        pwords = []
        while len(pwords) < len(special_tokens):
            new_emb = torch.randn_like(spec_token_emb) 
            new_emb = spec_token_emb + sample_dist * new_emb / new_emb.norm(p=2)
            print("Distance between pseudoword and [MASK]: ", pairwise_dist(new_emb.unsqueeze(0), spec_token_emb.unsqueeze(0))) 
            dist = pairwise_dist(new_emb.repeat(other_emb.size()[0],1), other_emb)
            if int(torch.argmin(dist)) < token_id: 
                print("Distance between pseudoword and other words in {}: Min=".format(args.model_path), torch.min(dist), tokenizer.convert_ids_to_tokens(int(torch.argmin(dist))))   # nearest word from [mask]
            else:
                print("Distance between pseudoword and other words in {}: Min=".format(args.model_path), torch.min(dist), tokenizer.convert_ids_to_tokens(int(torch.argmin(dist)) + 1))   # nearest word from [mask]
            for w in pwords:
                if pairwise_dist(w.unsqueeze(0), new_emb.unsqueeze(0)) < sample_dist:
                    break
            else:
                t_id = tokenizer.convert_tokens_to_ids(special_tokens[len(pwords)])
                emb[t_id] = new_emb
                pwords.append(new_emb)

        if args.model_class == "bert":
            plm.bert.embeddings.word_embeddings = nn.Embedding(emb.size()[0],emb.size()[1], _weight=emb)
        elif args.model_class == "roberta":
            plm.roberta.embeddings.word_embeddings = nn.Embedding(emb.size()[0],emb.size()[1], _weight=emb)
    
    metric = tuple()
    if "zs" in args.mode:
        save_name = "{}/{}.{}.{}.{}.{}".format(args.knowledge, args.model_path, args.mode, args.multi_mask, args.multi_token_handler,args.template)
        metric = hard_prompt_probing(plm, tokenizer, WrapperClass, dataset, candidates, gt, save_name, args.multi_mask, args.multi_token_handler, args.is_demo, args.use_cuda, args.mem_at_k, args.trained_soft, template_text)
    else:
        save_name = "{}/{}.{}.{}.{}.{}".format(args.knowledge, args.model_path, args.mode, args.multi_mask, args.multi_token_handler,args.loss)
        metric = soft_prompt_probing(plm, tokenizer, WrapperClass, dataset, candidates, gt, save_name, args.multi_mask, args.multi_token_handler, args.is_demo, args.use_cuda, args.mem_at_k, args.loss, args.trained_soft, template_text, args.soft_init)
    
    if len(metric) == 1:
        print("Acc with subset candidates: {:.4f}".format(metric[0]))
    else:
        print("R@1: {:.4f} R@5: {:.4f} R@10: {:.4f} R@25: {:.4f} R@50: {:.4f} MRR_avg: {:.4f} MRR: {:.4f}".format(metric[0], metric[1], metric[2], metric[3], metric[4], metric[5], metric[6]))
    print("===========================================")
    
    with open(args.save_path, "a", newline='') as f:
        writer = csv.writer(f)
        cont = [args.knowledge, args.model_path, args.mode, args.template, args.seed, args.multi_mask, args.multi_token_handler, args.loss, args.mem_at_k, args.threshold, args.trained_soft, args.soft_init]
        for m in metric:
            cont.append(float(m))
        writer.writerow(cont)
