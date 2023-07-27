from logging import raiseExceptions
from tqdm import tqdm
import json
import jsonlines
from openprompt.data_utils import InputExample
from get_memorized import split_by_MRR, split_by_Acc
import re

PW_NUM = 1
FEWSHOT_TRAIN_NUM = 10
FEWSHOT_VAL_NUM = 10

def load_candidates(knowledge, tokenizer):
    candidate_list = []
    if 'rdfs7' in knowledge or "prop" in knowledge:
        with open("data/property_v2.json") as f:
            data = json.loads(f.read())
        for d in tqdm(data):
            candidate_list.append(re.sub(r'-()/.',' ', d["pattern"]).strip('[XY] .'))
            if d["rdfs:subPropertyOf"]:
                candidate_list += [re.sub(r'-()/.',' ', word).strip('[XY] .') for word in d["rdfs:subPropertyOf"].values()]
    elif knowledge in ['subPropertyOf', 'subPropertyOf-sub'] or 'rdfs5' in knowledge:
        with open("data/property_v2.json") as f:
            data = json.loads(f.read())
        for d in tqdm(data):
            candidate_list.append(re.sub(r'[-()/]',' ', d["rdfs:label"]))
            if d["rdfs:subPropertyOf"]:
                candidate_list += [re.sub(r'[-()/]',' ', word) for word in d["rdfs:subPropertyOf"].keys()]
    else:
        with open("data/class.json") as f:
            data = json.loads(f.read())
        for d in tqdm(data):
            candidate_list.append(re.sub(r'[-()/]',' ', d["rdfs:label"]))
            candidate_list += [re.sub(r'[-()/]',' ', word) for word in d["rdfs:subClassOf"]]
    
    candidate_list = [' '.join(w.split()) for w in candidate_list]    # remove redundant space
    candidate_set = list(dict.fromkeys(candidate_list))  # remove duplicate candidates
    candidate_set.sort(key = lambda x: len(' '.join(tokenizer.tokenize(x)).split()))
    return dict((v, i) for i, v in enumerate(candidate_set))

class DatasetLoader():
    def __init__(self, knowledge, candidate_set, pattern, mode, read_name, trained_soft):
        self.knowledge = knowledge
        self.pattern = pattern
        self.mode = mode
        self.candidate_set = candidate_set
        self.read_name = read_name.split('/')[1]
        self.trained_soft = trained_soft

    def load_data(self):
        dataset, gt, special_tokens = [], [], ["[X]", "[Y]"]
        template_text = None
        if self.knowledge in ["type", "subClassOf", "subClassOf-type", "subPropertyOf", "subPropertyOf-prop", "subPropertyOf-sub", "domain", "domain-prop", "domain-type", "range", "range-prop", "range-type"]:
            path = "data/memorizing/{}.jsonl".format(self.knowledge.split('-')[0])
            dataset, gt = self.load_mem_data(path)
        elif "rdfs2" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs2_data()
        elif "rdfs3" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs3_data()
        elif "rdfs5" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs5_data()
        elif "rdfs7" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs7_data()
        elif "rdfs9" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs9_data()
        elif "rdfs11" in self.knowledge:
            dataset, gt, special_tokens, template_text = self.load_rdfs11_data()

        if self.mode == "zs-f":
            if self.knowledge == "type" or self.knowledge == "subClassOf":
                return dataset[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):520], gt[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):520], special_tokens, template_text
            else:
                return dataset[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):], gt[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):], special_tokens, template_text
        elif self.mode == "fs":
            fewshot_dataset = {}
            fewshot_dataset["train"] = dataset[:FEWSHOT_TRAIN_NUM]
            fewshot_dataset["val"] = dataset[FEWSHOT_TRAIN_NUM:(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM)]
            fewshot_dataset["test"] = dataset[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):]
            fewshot_gt = {}
            fewshot_gt["train"] = gt[:FEWSHOT_TRAIN_NUM]
            fewshot_gt["val"] = gt[FEWSHOT_TRAIN_NUM:(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM)]
            fewshot_gt["test"] = gt[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):]
            return fewshot_dataset, fewshot_gt, special_tokens, template_text
        
        return dataset, gt, special_tokens, template_text
        
    def domain_manual_textb(self, d, r):
        label = list(d["uuu"].keys())[0]
        template = list(d["uuu"].values())[0]
        w = template.strip("[X]").split(" ")[0]
        exp_know = ""
        if w == "'s":
            if "contain" in template:
                exp_know =  "if one has" + template.strip("[X]'s").split("contain ")[0] + "."
            else:
                exp_know =  "if one has" + template.strip("[X]'s").split("is ")[0] + "."
        else:
            exp_know = "if one {}".format(template.strip("[X] "))
        if label in r.keys():
            if r[label][0] in 'aeiouAEIOU':
                exp_know = exp_know.replace(r[label],"").replace("[Y]", "an "+ r[label])
            else:
                exp_know = exp_know.replace("[Y]", "a "+ r[label])
        else:
            exp_know = exp_know.replace("[Y]", "something")
        return exp_know

    def range_manual_textb(self, d, r):
        label = list(d["uuu"].keys())[0]
        template = list(d["uuu"].values())[0]
        w = template.strip(" [Y] .").split(" ")[-1]
        exp_know = ""
        if w == "is":
            exp_know = "if one is ".format(d["xxx"][0]) + template.replace("is [Y] ","")
        elif w == "contain":
            exp_know = "if one is ".format(d["xxx"][0]) + template.replace("contain [Y] ","")
        else:
            exp_know = "if {}".format(template.replace("[Y]", "that"))
        if label in r.keys():
            if r[label][0] in 'aeiouAEIOU':
                exp_know = exp_know.replace("[X]", "an "+ r[label])
            else:
                exp_know = exp_know.replace("[X]", "a "+ r[label])
        else:
            exp_know = exp_know.replace("[X]", "something")
        return exp_know
    
    def load_mem_data(self, path):
        dataset, gt = [], []
        cnt = 0
        with jsonlines.open(path) as data:
            if "type" in path:
                for d in data:
                    if self.mode == "zs-f" or self.mode == "zs-f" or self.mode == "zs-f-m":
                        dataset.append(InputExample(text_a = re.sub(r'[-()/]',' ', d["uuu"]).capitalize() + self.pattern, text_b = ".", guid = cnt))
                    else:
                        dataset.append(InputExample(text_a = re.sub(r'[-()/]',' ', d["uuu"]).capitalize(), text_b = ".", guid = cnt))
                    gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                    cnt += 1
            elif "subClassOf" in path:
                for d in data:
                    if self.mode == "zs-f" or self.mode == "zs-f" or self.mode == "zs-f-m":
                        if "type" in self.knowledge:   # for rdfs premise
                            dataset.append(InputExample(text_a = "[X]" + self.pattern, text_b = ".", guid = cnt))
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', d["uuu"]).split())]])
                        else:
                            dataset.append(InputExample(text_a = d["uuu"].capitalize() + self.pattern, text_b = ".", guid = cnt))
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                    else:
                        dataset.append(InputExample(text_a = d["uuu"].capitalize(), text_b = ".", guid = cnt))
                        gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                    cnt += 1
            elif "domain" in path:
                if self.mode == "zs-f" or self.mode == "zs-f" or self.mode == "zs-f-m":
                    r = {}
                    with jsonlines.open("data/memorizing/range.jsonl") as ranges:
                        for d in ranges:
                            r[list(d["uuu"].keys())[0]] = d["xxx"][0]
                    for d in data:
                        if "prop" in self.knowledge:    # for rdfs premises
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w.strip('[X]').strip('[Y] .')).split())] for w in d["uuu"].values()])
                            dataset.append(InputExample(text_a = "[X]", text_b = "[Y] .", guid = cnt))
                        else:
                            exp_know = self.domain_manual_textb(d, r)
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                            dataset.append(InputExample(text_a = self.pattern, text_b = exp_know, guid = cnt))
                        cnt += 1
                else:
                    for d in data:
                        label = list(d["uuu"].keys())[0]
                        dataset.append(InputExample(text_a = label, text_b = '.', guid = cnt))
                        gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                        cnt += 1
            elif "range" in path:
                if self.mode == "zs-f" or self.mode == "zs-f" or self.mode == "zs-f-m":
                    r = {}
                    with jsonlines.open("data/memorizing/domain.jsonl") as domains:
                        for d in domains:
                            r[list(d["uuu"].keys())[0]] = d["xxx"][0]
                    for d in data:
                        if "prop" in self.knowledge:
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w.strip('[X]').strip('[Y] .')).split())] for w in d["uuu"].values()])
                            dataset.append(InputExample(text_a = "[X]", text_b = "[Y] .", guid = cnt))
                        else:
                            exp_know = self.range_manual_textb(d, r)
                            dataset.append(InputExample(text_a = self.pattern, text_b = exp_know, guid = cnt))
                            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                        cnt += 1
                else:
                    for d in data:
                        label = list(d["uuu"].keys())[0]
                        dataset.append(InputExample(text_a = label, text_b = '.', guid = cnt))
                        gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                        cnt += 1
            elif "subPropertyOf" in path:    
                for d in data:
                    if "prop" in self.knowledge:    # subPropertyOf-prop
                        gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w.strip('[X]').strip('[Y] .')).split())] for w in d["uuu"].values()])
                        dataset.append(InputExample(text_a = "[X]", text_b = "[Y] .", guid = cnt))
                    else:
                        if self.mode == "zs-f" or self.mode == "zs-f" or self.mode == "zs-f-m":
                            dataset.append(InputExample(text_a = list(d["uuu"].keys())[0].capitalize() + self.pattern, text_b = '.', guid = cnt))
                        else:
                            dataset.append(InputExample(text_a = list(d["uuu"].keys())[0].capitalize(), text_b = '.', guid = cnt))
                        gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                    
                    cnt += 1
            else:
                print("Not A Memorizing Task!")
        
        return dataset, gt

    def select_premise(self, premise_knowledge1, premise_knowledge2, premise_mode):
        data_path = "data/memorizing/{}.jsonl".format(premise_knowledge1)
        read_path = "memorized_gpt/{}/{}.txt".format(premise_knowledge1, self.read_name)
        all_premise, memorized_premise, not_memorized_premise = split_by_Acc(read_path)
        if premise_mode[0] == "2":
            selected_premise1 = all_premise
        elif premise_mode[0] == "1":
            selected_premise1 = memorized_premise
        elif premise_mode[0] == "0":
            selected_premise1 = not_memorized_premise
        else:
            raiseExceptions("Wrong Inference Mode!")
        
        if '-prop' in premise_knowledge2:
            args = self.read_name.split(".")
            args[1] = "zs-f"
            args[-1] = self.pattern
            read_path = "memorized_new/{}/{}.txt".format(premise_knowledge2, ".".join(args))
        else:
            read_path = "memorized_new/{}/{}.txt".format(premise_knowledge2, self.read_name)
        all_premise, memorized_premise, not_memorized_premise = split_by_MRR(read_path)
        if premise_mode[1] == "2":
            selected_premise2 = all_premise
        elif premise_mode[1] == "1":
            selected_premise2 = memorized_premise
        elif premise_mode[1] == "0":
            selected_premise2 = not_memorized_premise
        else:
            raiseExceptions("Wrong Inference Mode!")
        
        selected_index = set(selected_premise1).intersection(set(selected_premise2))
        # print(selected_premise1, selected_premise2, selected_index)
        all_premise = []
        with jsonlines.open(data_path) as f:
            for d in f:
                all_premise.append(d)
        all_premise = all_premise[(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM):]  # test set
        selected_premise = [all_premise[k] for k in selected_index]
        return all_premise[:(FEWSHOT_TRAIN_NUM + FEWSHOT_VAL_NUM)] + selected_premise

    def load_rdfs_template(self, p1, p2, soft_p1: bool, soft_p2: bool, soft_h: bool):
        if self.mode == "zs-f" or self.mode == "zs-f":
            if p1 and p2:
                template_text = '{"meta":"1"} {"meta":"2"} Therefore, {"meta":"h"} {"mask"} .'
            elif p1 and not p2:
                template_text = '{"meta":"1"} Therefore, {"meta":"h"} {"mask"} .'
            elif p2 and not p1:
                template_text = '{"meta":"2"} Therefore, {"meta":"h"} {"mask"} .'
            else:
                template_text = 'Therefore, {"meta":"h"} {"mask"} .'
        else:
            if p1 and p2:
                template_text = '{"meta":"1"} {"soft"} {"meta":"2"} {"soft"} {"meta":"h"} {"mask"} .'
            elif p1 and not p2:
                template_text = '{"meta":"1"} {"soft"} {"meta":"h"} {"mask"} .'
            elif p2 and not p1:
                template_text = '{"meta":"2"} {"soft"} {"meta":"h"} {"mask"} .'
            else:
                template_text = '{"soft"} {"meta":"h"} {"mask"} .'
        if self.trained_soft == "load":
            if soft_p1:
                template_text = template_text.replace('{"meta":"1"}', '{"meta":"10"} {"soft"} {"soft"} {"soft"} {"meta":"11"}')
            if soft_p2:
                template_text = template_text.replace('{"meta":"2"}', '{"meta":"20"} {"soft"} {"soft"} {"soft"} {"meta":"21"}')
            if soft_h:
                template_text = template_text.replace('{"meta":"h"}', '{"meta":"h"} {"soft"} {"soft"} {"soft"}')

        return template_text

    def load_rdfs2_data(self):
        dataset, gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("domain", "domain-prop", premise_mode)
        inputs = []
        if self.trained_soft == "none":
            path_2 = "data/memorizing/range.jsonl"
            r = {}
            with jsonlines.open(path_2) as ranges:    # for aaa rdfs:domain xxx pattern 
                for d in ranges:
                    r[list(d["uuu"].keys())[0]] = d["xxx"][0]

            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = self.domain_manual_textb(d, r)
                    if d["xxx"][0] in 'aeiouAEIOU': 
                        p1 = "One has to be an {} ".format(d["xxx"][0]) + p1
                    else:
                        p1 = "One has to be a {} ".format(d["xxx"][0]) + p1
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, "[X] is a particular"])
                    
        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0] + ' {"soft"} {"soft"} {"soft"} ' + d["xxx"][0]
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, '[X]'])

        template_text = self.load_rdfs_template(p1, p2, True, False, True)
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            for i in range(PW_NUM):
                p1 = inputs[l][0].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                p2 = inputs[l][1].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                h = inputs[l][2].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                if self.trained_soft == "none":
                    dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
                else:
                    if p1:
                        t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                    else:
                        t0, t1 = '', ''
                    dataset.append(InputExample(meta={"10": t0, "11": t1, "2":p2, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
                gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
            special_tokens.append("[Y{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text

    def load_rdfs3_data(self):
        dataset, gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("range", "range-prop", premise_mode)
        inputs = []
        if self.trained_soft == "none":
            path_2 = "data/memorizing/domain.jsonl"
            r = {}
            with jsonlines.open(path_2) as ranges:    # for aaa rdfs:domain xxx pattern 
                for d in ranges:
                    r[list(d["uuu"].keys())[0]] = d["xxx"][0]

            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = self.range_manual_textb(d, r)
                    if d["xxx"][0] in 'aeiouAEIOU': 
                        p1 = "One has to be an {} ".format(d["xxx"][0]) + p1
                    else:
                        p1 = "One has to be a {} ".format(d["xxx"][0]) + p1
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, "[Y] is a particular"])

        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0] + ' {"soft"} {"soft"} {"soft"} ' + d["xxx"][0] 
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, '[Y]'])

        template_text = self.load_rdfs_template(p1, p2, True, False, True)
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            for i in range(PW_NUM):
                p1 = inputs[l][0].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                p2 = inputs[l][1].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                h = inputs[l][2].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                if self.trained_soft == "none":
                    dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
                else:
                    if p1:
                        t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                    else:
                        t0, t1 = '', ''
                    dataset.append(InputExample(meta={"10": t0, "11": t1, "2":p2, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
                gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
                cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
            special_tokens.append("[Y{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text

    def load_rdfs5_data(self):
        # hop subproperty
        dataset,gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("subPropertyOf", "subPropertyOf-sub", premise_mode)
        inputs = []
        p1, p2 = "", ""
        if self.trained_soft == "none":
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0].capitalize() + " implies " + list(d["xxx"].keys())[0] + " ."
                if premise_mode[1] == "2":
                    p2 = "[X] implies " + list(d["uuu"].keys())[0] + " ."
                inputs.append([p1, p2, "[X] implies"])
        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0].capitalize() + ' {"soft"} {"soft"} {"soft"} ' + list(d["xxx"].keys())[0] + " ."
                if premise_mode[1] == "2":
                    p2 = "[X]" + ' {"soft"} {"soft"} {"soft"} ' + list(d["uuu"].keys())[0] + " ."
                inputs.append([p1, p2, "[X] implies"])
        
        template_text = self.load_rdfs_template(p1, p2, True, True, True)
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            if self.trained_soft == "none":
                dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
            else:
                if p1:
                    t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t0, t1 = '', ''
                if p2:
                    t2, t3 = p2.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t2, t3 = '', ''
                dataset.append(InputExample(meta={"10": t0, "11": t1, "20":t2, "21": t3, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"].keys()])
            cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text

    def load_rdfs7_data(self):
        # subproperty, aaa bbb
        dataset,gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("subPropertyOf", "subPropertyOf-prop", premise_mode)
        inputs = []
        if self.trained_soft == "none":
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0].capitalize() + " implies " + list(d["xxx"].keys())[0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, "[X]"])
        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = list(d["uuu"].keys())[0].capitalize() + ' {"soft"} {"soft"} {"soft"} ' + list(d["xxx"].keys())[0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = list(d["uuu"].values())[0]
                else:
                    p2 = ""
                inputs.append([p1, p2, "[X]"])

        template_text = self.load_rdfs_template(p1, p2, True, False, False).strip(".") + "[Y] ."
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            for i in range(PW_NUM):
                p1 = inputs[l][0].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                p2 = inputs[l][1].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                h = inputs[l][2].replace("[X]", "[X{}]".format(i)).replace("[Y]", "[Y{}]".format(i))
                if self.trained_soft == "none":
                    dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
                else:
                    if p1:
                        t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                    else:
                        t0, t1 = '', ''
                    dataset.append(InputExample(meta={"10": t0, "11": t1, "2":p2, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
                gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w.strip("[X]").strip(" [Y] .")).split())] for w in d["xxx"].values()])
                cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
            special_tokens.append("[Y{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text

    def load_rdfs9_data(self):
        # subclassof, type
        dataset, gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("subClassOf", "subClassOf-type", premise_mode)
        inputs = []
        if self.trained_soft == "none":
            for d in selected_premises:
                if premise_mode[0] == "2":
                    if d["xxx"][0][0] in 'aeiouAEIOU':
                        p1 = d["uuu"].capitalize() + " is an " + d["xxx"][0] + " ."
                    else:
                        p1 = d["uuu"].capitalize() + " is a " + d["xxx"][0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    if d["uuu"][0] in 'aeiouAEIOU':
                        p2 = "[X] is an {} .".format(d["uuu"])
                    else:
                        p2 = "[X] is a {} .".format(d["uuu"])
                else:
                    p2 = ""
                inputs.append([p1, p2, "[X] is a particular"])
        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                    p1 = d["uuu"].capitalize() + ' {"soft"} {"soft"} {"soft"} ' + d["xxx"][0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = '[X] {"soft"} {"soft"} {"soft"} ' + (d["uuu"]) + ' .'
                else:
                    p2 = ""
                inputs.append([p1, p2, '[X]'])
        
        template_text = self.load_rdfs_template(p1, p2, True, True, True)
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            if self.trained_soft == "none":
                dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
            else:
                if p1:
                    t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t0, t1 = '', ''
                if p2:
                    t2, t3 = p2.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t2, t3 = '', ''
                dataset.append(InputExample(meta={"10": t0, "11": t1, "20":t2, "21": t3, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
            cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text

    def load_rdfs11_data(self):
        # subclassof hop
        dataset, gt, special_tokens = [], [], []
        cnt = 0
        premise_mode = list(self.knowledge[-2:])  # [0/1/2, 0/1/2]
        selected_premises = self.select_premise("subClassOf", "subClassOf-type", premise_mode)
        inputs = []
        if self.trained_soft == "none":
            for d in selected_premises:
                if premise_mode[0] == "2":
                    if d["xxx"][0][0] in 'aeiouAEIOU':
                        p1 = d["uuu"].capitalize() + " is an " + d["xxx"][0] + " ."
                    else:
                        p1 = d["uuu"].capitalize() + " is a " + d["xxx"][0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    if d["uuu"][0] in 'aeiouAEIOU':
                        p2 = "[X] is an {} .".format(d["uuu"])
                    else:
                        p2 = "[X] is a {} .".format(d["uuu"])
                else:
                    p2 = ""
                inputs.append([p1, p2, "[X] is a particular"])
        else:
            for d in selected_premises:
                if premise_mode[0] == "2":
                        p1 = d["uuu"].capitalize() + ' {"soft"} {"soft"} {"soft"} ' + d["xxx"][0] + " ."
                else:
                    p1 = ""
                if premise_mode[1] == "2":
                    p2 = '[X] {"soft"} {"soft"} {"soft"} ' + (d["uuu"]) + ' .'
                else:
                    p2 = ""
                inputs.append([p1, p2, '[X]'])
                
        template_text = self.load_rdfs_template(p1, p2, True, True, True)
        for l in range(len(selected_premises)):
            d = selected_premises[l]
            p1, p2, h = inputs[l]
            if self.trained_soft == "none":
                dataset.append(InputExample(meta={"1": p1, "2":p2, "h": h}, guid=cnt, tgt_text=d["xxx"]))
            else:
                if p1:
                    t0, t1 = p1.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t0, t1 = '', ''
                if p2:
                    t2, t3 = p2.split(' {"soft"} {"soft"} {"soft"} ')
                else:
                    t2, t3 = '', ''
                dataset.append(InputExample(meta={"10": t0, "11": t1, "20":t2, "21": t3, "h": h}, guid = cnt, tgt_text=d["xxx"])) 
            gt.append([self.candidate_set[' '.join(re.sub(r'[-()/]',' ', w).split())] for w in d["xxx"]])
            cnt += 1

        for i in range(PW_NUM):
            special_tokens.append("[X{}]".format(i))
        print(dataset[0], len(dataset) - FEWSHOT_TRAIN_NUM - FEWSHOT_VAL_NUM)
        return dataset, gt, special_tokens, template_text
