from typing import Counter
import torch
import torch.nn as nn
from tqdm import tqdm
from metric import evaluate_batch, evaluate_batch_mul_choice
from utils import load_soft_emb_with_conj, load_soft_emb_without_conj

from openprompt.prompts import ManualTemplate
from openprompt.prompts import MixedTemplate
from openprompt import PromptDataLoader
from openprompt import PromptForClassification

def group_logits(logits, batch_result, candidate_tokens, i, multi_mask, multi_token_handler):
    curr_token_num_result = logits[:,0,candidate_tokens[i][:,0]].unsqueeze(1)
    for j in range(1, i):
        if multi_mask == "m":
            curr_token_num_result = torch.cat((curr_token_num_result, logits[:,j,candidate_tokens[i][:,j]].unsqueeze(1)), 1)
        else:
            curr_token_num_result = torch.cat((curr_token_num_result, logits[:,0,candidate_tokens[i][:,j]].unsqueeze(1)), 1)
    
    if multi_token_handler == "mean":
        curr_token_num_result = torch.mean(curr_token_num_result, dim=1)
    elif multi_token_handler == "max":
        curr_token_num_result = torch.max(curr_token_num_result, dim=1).values
    elif multi_token_handler == "first":
        curr_token_num_result = torch.select(curr_token_num_result, dim=1, index=0)
    else:
        raise ValueError("multi_token_handler is not configured")
    
    if len(batch_result) == 0:
        batch_result = curr_token_num_result
    else:
        batch_result = torch.cat((batch_result, curr_token_num_result),1)
    return batch_result

def get_batch_curr_num_token(batch, tokenizer, input_ids, attention_mask, firstmask_index, device, i):
    batch['inputs_embeds'] = None   # process_batch() in MixedTemplate sets input_ids to None and sets input_embeds
    batch['input_ids'] = input_ids.clone()
    for idx, mask_idx in firstmask_index.tolist():
        batch['input_ids'][idx] = torch.cat((input_ids[idx][:mask_idx], torch.tensor([tokenizer.mask_token_id]*i).to(device), input_ids[idx][mask_idx+1:1-i]), dim=0)
        batch['attention_mask'][idx] = torch.cat((attention_mask[idx][:mask_idx], torch.tensor([1]*i).to(device), attention_mask[idx][mask_idx+1:1-i]), dim=0)

    loss_ids = torch.zeros_like(batch['input_ids'])
    loss_ids[batch['input_ids'] == tokenizer.cls_token_id] = -100
    loss_ids[batch['input_ids'] == tokenizer.mask_token_id] = 1
    loss_ids[batch['input_ids'] == tokenizer.sep_token_id] = -100
    batch['loss_ids'] = loss_ids
    return batch

def hard_prompt_probing(plm, tokenizer, WrapperClass, dataset, candidate_set, gt, save_name, multi_mask, multi_token_handler, is_premise, use_cuda, mem_at_k, trained_soft, template_text):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    candidate_ids = []
    for cand in candidate_set:
         ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand))
         candidate_ids.append(ids)

    count = Counter([len(k) for k in candidate_ids])
    num_token = sorted(count.keys())

    candidate_tokens = {key:[] for key in count.keys()}
    for cand_id in candidate_ids:   # group candidates by # of composing tokens
        candidate_tokens[len(cand_id)].append(cand_id)
    for cand_tok_num in candidate_tokens:   # convert to tensor
        candidate_tokens[cand_tok_num] = torch.tensor(candidate_tokens[cand_tok_num])

    if not template_text:
        template = ManualTemplate(
                text = '{"placeholder":"text_a"} {"mask"} {"placeholder":"text_b"}',
                tokenizer = tokenizer
            )
    else:
        if trained_soft == "none":
            template = ManualTemplate(
                    text = template_text,
                    tokenizer = tokenizer
                )
        elif trained_soft == "load":
            template = MixedTemplate(
                    model = plm,
                    text = template_text,
                    tokenizer = tokenizer
                )
            template = load_soft_emb_without_conj(template, save_name, device)

    prompt_model = PromptForClassification(
        template = template,
        plm = plm,
        verbalizer = None,
        freeze_plm = True
    )
    
    if use_cuda:
        prompt_model = prompt_model.cuda()

    batch_size = 16
    data_loader = PromptDataLoader(
        dataset = dataset,
        tokenizer = tokenizer,
        template = template,
        tokenizer_wrapper_class = WrapperClass,
        batch_size = batch_size
    )

    Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr, mrr_avg = 0, 0, 0, 0, 0, 0, 0
    acc = 0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(data_loader):
            logits = torch.Tensor()
            firstmask_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero()
            batch_result = torch.Tensor()
            input_ids = batch['input_ids'].clone()
            attention_mask = batch['attention_mask'].clone()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if use_cuda:
                batch = batch.cuda()
            for i in num_token:
                if i > 1:
                    if multi_mask == "m":
                        batch = get_batch_curr_num_token(batch, tokenizer, input_ids, attention_mask, firstmask_index, device, i)
                        outputs = prompt_model.prompt_model(batch)
                        logits = prompt_model.extract_at_mask(outputs.logits, batch)
                    else:
                        outputs = outputs
                        logits = logits
                else:
                    outputs = prompt_model.prompt_model(batch)
                    logits = prompt_model.extract_at_mask(outputs.logits, batch) # 16 * V / 16 * mask_num * V
                    logits = torch.unsqueeze(logits, 1)

                LogSoftmax = nn.LogSoftmax(dim=-1)
                logits = LogSoftmax(logits)
                batch_result = group_logits(logits, batch_result, candidate_tokens, i, multi_mask, multi_token_handler)
                
            batch_gt = gt[cnt*batch_size: (cnt+1)*batch_size]
            logits = batch_result.to("cpu")
            if is_premise: # for illustrating & check good case / premise of rdfs rules
                pred = torch.topk(logits, mem_at_k, dim=1)
                indices = pred.indices.tolist()
                inv_cand = dict(zip(candidate_set.values(), candidate_set.keys()))
                cand_result = torch.argsort(logits, dim=1, descending=True)
                cand_result = torch.argsort(cand_result, dim=1)
                f = open("memorized/{}.txt".format(save_name),'a')
                for i in range(len(batch_gt)):
                    pred_cand = [inv_cand[indices[i][j]] for j in range(mem_at_k)]
                    print(cnt * batch_size + i, pred_cand, [inv_cand[k] for k in batch_gt[i]], file=f)
                    case_indices = batch_gt[i]
                    print(cnt * batch_size + i, float(1 / (torch.mean(cand_result[i][case_indices].float()) + 1)), float(1 / (torch.min(cand_result[i][case_indices].float()) + 1)), file=f)
            
            cnt += 1

            # Normal test
            Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = evaluate_batch(Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr, logits, batch_gt)
    return (Recall_1/len(gt), Recall_5/len(gt), Recall_10/len(gt), Recall_25/len(gt), Recall_50/len(gt), mrr_avg/len(gt), mrr/len(gt))

    #         # To compare memorization with gpt with sampled candidates
    #         acc = evaluate_batch_mul_choice(acc, "test-gpt/data-gpt/{}.jsonl".format(save_name.split('/')[0]), logits, batch_gt, cnt - 1, batch_size, candidate_set, save_name)
    # return (acc / len(gt))

    #         # To compare reasoning with gpt with sampled candidates
    #         if "rdfs2" in save_name.split('/')[0]:
    #             acc = evaluate_batch_mul_choice(acc, "test-gpt/data-gpt/domain.jsonl", logits, batch_gt, cnt - 1, batch_size, candidate_set, save_name)
    #         elif "rdfs3" in save_name.split('/')[0]:
    #             acc = evaluate_batch_mul_choice(acc, "test-gpt/data-gpt/range.jsonl", logits, batch_gt, cnt - 1, batch_size, candidate_set, save_name)
    #         if "rdfs5" in save_name.split('/')[0] or "rdfs7" in save_name.split('/')[0]:
    #             acc = evaluate_batch_mul_choice(acc, "test-gpt/data-gpt/subPropertyOf.jsonl", logits, batch_gt, cnt - 1, batch_size, candidate_set, save_name)
    #         if "rdfs9" in save_name.split('/')[0] or "rdfs11" in save_name.split('/')[0]:
    #             acc = evaluate_batch_mul_choice(acc, "test-gpt/data-gpt/subClassOf.jsonl", logits, batch_gt, cnt - 1, batch_size, candidate_set, save_name)
    # return (acc / len(gt))

def soft_prompt_probing(plm, tokenizer, WrapperClass, dataset, candidate_set, ground_truths, save_name, multi_mask, multi_token_handler, is_premise, use_cuda, mem_at_k, loss_type, trained_soft, template_text, soft_init):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    candidate_ids = []
    for cand in candidate_set:
         ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(cand))
         candidate_ids.append(ids)

    count = Counter([len(k) for k in candidate_ids])
    num_token = sorted(count.keys())

    candidate_tokens = {key:[] for key in count.keys()}
    for cand_id in candidate_ids:
        candidate_tokens[len(cand_id)].append(cand_id)
    for cand_tok_num in candidate_tokens:
        candidate_tokens[cand_tok_num] = torch.tensor(candidate_tokens[cand_tok_num])

    batch_size = 8
    tot_epoch = 100

    from torch.optim import AdamW
    if loss_type == "NLL":
        loss_func = nn.NLLLoss()
        pass
    else:
        pos_weight = torch.ones([len(candidate_set)])
        loss_func = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if not template_text:
        if soft_init == "rand":
            template = MixedTemplate(
                model = plm,
                tokenizer = tokenizer,
                text = '{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"mask"} {"placeholder":"text_b"}'
            )
        else:
            if "subClassOf" in save_name or "type" in save_name:
                template = MixedTemplate(
                    model = plm,
                    tokenizer = tokenizer,
                    text = '{"placeholder":"text_a"} {"soft":"is"} {"soft":"a"} {"soft":"particular"} {"mask"} {"placeholder":"text_b"}'
                )
            elif "subPropertyOf" in save_name:
                template = MixedTemplate(
                    model = plm,
                    tokenizer = tokenizer,
                    text = '{"placeholder":"text_a"} {"soft":"implies"} {"soft": "the"} {"soft": "property"} {"mask"} {"placeholder":"text_b"}'
                )
            elif "domain" in save_name:
                template = MixedTemplate(
                    model = plm,
                    tokenizer = tokenizer,
                    text = '{"placeholder":"text_a"} {"soft":"\'s"} {"soft": "domain"} {"soft": "is"} {"mask"} {"placeholder":"text_b"}'
                )
            elif "range" in save_name:
                template = MixedTemplate(
                    model = plm,
                    tokenizer = tokenizer,
                    text = '{"placeholder":"text_a"} {"soft":"\'s"} {"soft": "range"} {"soft": "is"} {"mask"} {"placeholder":"text_b"}'
                )
            # else:
            #     template = MixedTemplate(
            #         model = plm,
            #         tokenizer = tokenizer,
            #         text = '{"placeholder":"text_a"} {"soft"} {"soft"} {"soft"} {"mask"} {"placeholder":"text_b"}'
            #     )
    else:
        template = MixedTemplate(
                    model = plm,
                    text = template_text,
                    tokenizer = tokenizer
        )
        if trained_soft == "load":
            template = load_soft_emb_with_conj(template, save_name, device)
    
    prompt_model = PromptForClassification(
        template = template,
        plm = plm,
        verbalizer = None,
        freeze_plm = True
    )
    if use_cuda:
        prompt_model = prompt_model.cuda()

    train_dataloader = PromptDataLoader(
                dataset = dataset["train"],
                tokenizer = tokenizer,
                template = template,
                tokenizer_wrapper_class = WrapperClass,
                batch_size = batch_size,
                shuffle = False,
                max_seq_length = 64
            )
    train_gt = ground_truths["train"]

    val_dataloader = PromptDataLoader(
                dataset = dataset["val"],
                tokenizer = tokenizer,
                template = template,
                tokenizer_wrapper_class = WrapperClass,
                batch_size = batch_size,
                shuffle = False,
                max_seq_length = 64
            )
    val_gt = ground_truths["val"]

    # Only tune the prompt parameters
    optimizer_grouped_parameters = [
        {'params': [p for n,p in prompt_model.template.named_parameters() if "raw_embedding" not in n]}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=0.5)

    from transformers.optimization import get_linear_schedule_with_warmup

    tot_step  = len(train_dataloader)*tot_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, tot_step)

    best_val_mrr_avg = 0
    best_val_soft_emb = prompt_model.template.soft_embedding.weight.data.clone().detach()
    best_val_epoch = 0

    for epoch in tqdm(range(tot_epoch)):
        ## training ##
        tot_loss = 0
        cnt = 0
        Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = 0, 0, 0, 0, 0, 0, 0
        for batch in train_dataloader:
            logits = torch.Tensor()
            firstmask_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero()
            batch_result = torch.Tensor()
            input_ids = batch['input_ids'].clone()
            attention_mask = batch['attention_mask'].clone()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if use_cuda:
                batch = batch.cuda()
            for i in num_token:
                if i > 1:
                    if multi_mask == "m":
                        batch = get_batch_curr_num_token(batch, tokenizer, input_ids, attention_mask, firstmask_index, device, i)
                        outputs = prompt_model.prompt_model(batch)
                        logits = prompt_model.extract_at_mask(outputs.logits, batch)
                    else:
                        outputs = outputs
                        logits = logits
                else:
                    outputs = prompt_model.prompt_model(batch)
                    logits = prompt_model.extract_at_mask(outputs.logits, batch) # 16 * V / 16 * mask_num * V
                    logits = torch.unsqueeze(logits, 1)

                if loss_type == "log" or loss_type == "NLL":
                    LogSoftmax = nn.LogSoftmax(dim=-1)
                    logits = LogSoftmax(logits)

                batch_result = group_logits(logits, batch_result, candidate_tokens, i, multi_mask, multi_token_handler)
            
            batch_train_gt = train_gt[cnt*batch_size: (cnt+1)*batch_size]
            logits = batch_result.to("cpu")
            if loss_type == "NLL":
                target = []
                for i in range(len(batch_train_gt)):
                    for gt in batch_train_gt[i]:
                        target.append(gt)
                    begin_idx = len(target) - len(batch_train_gt[i])
                    logits = torch.cat((logits[:begin_idx],logits[begin_idx].expand(len(batch_train_gt[i]),-1),logits[begin_idx+1:]),dim=0)
                target = torch.tensor(target)
            else:
                target = torch.zeros_like(logits, dtype=float)
                for i in range(len(batch_train_gt)):
                    target[i][torch.LongTensor(batch_train_gt[i])] = 1
                
            loss = loss_func(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            tot_loss += loss.item()

            Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = evaluate_batch(Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr, logits, batch_train_gt)
            cnt += 1
        
        print("Epoch {} Average loss:".format(epoch), tot_loss/(len(train_dataloader)))
        print("Epoch {} Training Set: R@1:{} R@5:{} R@10:{} mrr_avg:{} mrr:{}".format(epoch, Recall_1 / len(train_gt), Recall_5 / len(train_gt), Recall_10 / len(train_gt), mrr_avg / len(train_gt), mrr / len(train_gt)))

        ## validation ##
        cnt = 0
        Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = 0, 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for batch in val_dataloader:
                logits = torch.Tensor()
                firstmask_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero()
                batch_result = torch.Tensor()
                input_ids = batch['input_ids'].clone()
                attention_mask = batch['attention_mask'].clone()
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                if use_cuda:
                    batch = batch.cuda()
                for i in num_token:
                    if i > 1:
                        if multi_mask == "m":
                            batch = get_batch_curr_num_token(batch, tokenizer, input_ids, attention_mask, firstmask_index, device, i)
                            outputs = prompt_model.prompt_model(batch)
                            logits = prompt_model.extract_at_mask(outputs.logits, batch)
                        else:
                            outputs = outputs
                            logits = logits
                    else:
                        outputs = prompt_model.prompt_model(batch)
                        logits = prompt_model.extract_at_mask(outputs.logits, batch) # 16 * V / 16 * mask_num * V
                        logits = torch.unsqueeze(logits, 1)

                    batch_result = group_logits(logits, batch_result, candidate_tokens, i, multi_mask, multi_token_handler)
                    
                batch_val_gt = val_gt[cnt*batch_size: (cnt+1)*batch_size]
                logits = batch_result.to("cpu")

                Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = evaluate_batch(Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr, logits, batch_val_gt)
                cnt += 1

        if mrr_avg > best_val_mrr_avg:
            # torch.save(prompt_model.template.state_dict(), '')
            best_val_epoch = epoch
            best_val_mrr_avg = mrr_avg
            best_val_soft_emb = prompt_model.template.soft_embedding.weight.data.clone().detach()
        print("Epoch {} Validation Set: R@1:{} R@5:{} R@10:{} mrr_avg:{} mrr:{}".format(epoch, Recall_1 / len(val_gt), Recall_5 / len(val_gt), Recall_10 / len(val_gt), mrr_avg / len(val_gt), mrr / len(val_gt)))

    print("Load soft embedding from epoch {}...".format(best_val_epoch))
    ## Test ##
    prompt_model.template.soft_embedding.weight.data = best_val_soft_emb
    if not "rdfs" in save_name:
        torch.save(best_val_soft_emb, 'soft_embedding/{}.pth'.format(save_name))
    test_dataloader = PromptDataLoader(
        dataset = dataset["test"],
        tokenizer = tokenizer,
        template = template,
        tokenizer_wrapper_class = WrapperClass,
        batch_size = batch_size
    )
    test_gt = ground_truths["test"]
    Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr, mrr_avg = 0, 0, 0, 0, 0, 0, 0
    cnt = 0
    with torch.no_grad():
        for batch in tqdm(test_dataloader):
            logits = torch.Tensor()
            firstmask_index = (batch['input_ids'] == tokenizer.mask_token_id).nonzero()
            batch_result = torch.Tensor()
            input_ids = batch['input_ids'].clone()
            attention_mask = batch['attention_mask'].clone()
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            if use_cuda:
                batch = batch.cuda()
            for i in num_token:
                if i > 1:
                    if multi_mask == "m":
                        batch = get_batch_curr_num_token(batch, tokenizer, input_ids, attention_mask, firstmask_index, device, i)
                        outputs = prompt_model.prompt_model(batch)
                        logits = prompt_model.extract_at_mask(outputs.logits, batch)
                    else:
                        outputs = outputs
                        logits = logits
                else:
                    outputs = prompt_model.prompt_model(batch)
                    logits = prompt_model.extract_at_mask(outputs.logits, batch) # 16 * V / 16 * mask_num * V
                    logits = torch.unsqueeze(logits, 1)

                LogSoftmax = nn.LogSoftmax(dim=-1)
                logits = LogSoftmax(logits)
                batch_result = group_logits(logits, batch_result, candidate_tokens, i, multi_mask, multi_token_handler)
                
            batch_test_gt = test_gt[cnt*batch_size: (cnt+1)*batch_size]
            logits = batch_result.to("cpu")
            if is_premise: # for illustrating & check good case / premise of rdfs rules
                pred = torch.topk(logits, mem_at_k, dim=1)
                indices = pred.indices.tolist()
                inv_cand = dict(zip(candidate_set.values(), candidate_set.keys()))
                cand_result = torch.argsort(logits, dim=1, descending=True)
                cand_result = torch.argsort(cand_result, dim=1)
                f = open("memorized_new/{}.txt".format(save_name),'a')
                for i in range(len(batch_test_gt)):
                    pred_cand = [inv_cand[indices[i][j]] for j in range(mem_at_k)]
                    print(cnt * batch_size + i, pred_cand, [inv_cand[k] for k in batch_test_gt[i]], file=f)
                    case_indices = batch_test_gt[i]
                    print(cnt * batch_size + i, float(1 / (torch.mean(cand_result[i][case_indices].float()) + 1)), float(1 / (torch.min(cand_result[i][case_indices].float()) + 1)), file=f)

            Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr = evaluate_batch(Recall_1, Recall_5, Recall_10, Recall_25, Recall_50, mrr_avg, mrr, logits, batch_test_gt)
            cnt += 1
    
    return (Recall_1/len(test_gt), Recall_5/len(test_gt), Recall_10/len(test_gt), Recall_25/len(test_gt), Recall_50/len(test_gt), mrr_avg/len(test_gt), mrr/len(test_gt))