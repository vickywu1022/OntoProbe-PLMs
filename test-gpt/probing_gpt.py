import json
import re
import jsonlines
import random

SAMPLE = 20
class_list, prop_list = [] ,[]
r1, r2 = {}, {}

import openai
import jsonlines
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

# OpenAI API credentials
openai.api_key = 'YOUR_API_KEY'

def test_memorization(knowledge):
    acc = 0
    responses = list()
    with jsonlines.open("{}.jsonl".format(knowledge)) as reader:
        for obj in reader:
            query = obj["prompt"]
            gold = obj["gold"]
            response = completion_with_backoff(model="gpt-3.5-turbo", messages=[{"role": "user", "content": query}])
            response_text = response.choices[0].message.content.strip()
            if gold.lower() in response_text.lower():
                acc += 1
                responses.append(response_text + " 1!")
            else:
                responses.append(response_text + " 0!")
            print(responses[-1])

    with open("{}-response.txt".format(knowledge), "w") as f:
        f.write("\n".join(responses))

    return acc / len(list(responses))

def domain_manual_textb(d, r):
    label = list(d["uuu"].keys())[0]
    template = list(d["uuu"].values())[0]
    w = template.strip("[X]").split(" ")[0]
    exp_know = ""
    if w == "'s":
        if "contain" in template:
            exp_know =  "if one has" + template.strip("[X]'s").split("contain ")[0]
        else:
            exp_know =  "if one has" + template.strip("[X]'s").split("is ")[0]
    else:
        exp_know = "if one {}".format(template.strip("[X] "))
    if label in r.keys():
        if r[label][0] in 'aeiouAEIOU':
            exp_know = exp_know.replace(r[label],"").replace("[Y]", "an "+ r[label])
        else:
            exp_know = exp_know.replace("[Y]", "a "+ r[label])
    else:
        exp_know = exp_know.replace("[Y]", "something")
    return exp_know.strip(".").strip()

def range_manual_textb(d, r):
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
    return exp_know.strip(".").strip()

def generate_memorization_data(knowledge):
    with jsonlines.open("../data/memorizing/{}.jsonl".format(knowledge)) as reader: 
        data = list(reader)
        golds = list()
        prompts = list()
        cands = list()
        MAX = min(520, len(data))
        for d in data[20:MAX]:
            if knowledge == "subPropertyOf":
                query, gold = " ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"]))).split()), " ".join(re.sub(r'[-()/]',' ', next(iter(d["xxx"]))).split())
                options = random.sample(set(prop_list) - {gold} - {query}, SAMPLE - 1)
            elif knowledge == "type" or knowledge == "subClassOf":
                query, gold = " ".join(re.sub(r'[-()/]',' ', d["uuu"]).split()), " ".join(re.sub(r'[-()/]',' ', d["xxx"][0]).split())
                options = random.sample(set(class_list) - {gold} - {query}, SAMPLE - 1)
            else:
                query, gold = " ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"]))).split()), " ".join(re.sub(r'[-()/]',' ', d["xxx"][0]).split())
                options = random.sample(set(class_list) - {gold} - {query}, SAMPLE - 1)
            
            options.append(gold)
            random.shuffle(options)

            formatted_options = []
            for i, option in enumerate(options):
                formatted_option = "({}) {}".format(chr(ord('a') + i), option)
                formatted_options.append(formatted_option)
                if gold == option:
                    golds.append(formatted_option)
            
            if knowledge == "type" :
                prompt = "What is the type of {}? {}".format(query, ", ".join(formatted_options))
            elif knowledge == "subClassOf":
                prompt = "What is the superclass of {}? {}".format(query, ", ".join(formatted_options))
            elif knowledge == "subPropertyOf":
                prompt = "What does {} imply? {}".format(query, ", ".join(formatted_options))
            elif knowledge == "domain":
                prompt = "What is the type of it {}? {}".format(domain_manual_textb(d, r1), ", ".join(formatted_options))
            elif knowledge == "range":
                prompt = "What is the type of it {}? {}".format(range_manual_textb(d, r2), ", ".join(formatted_options))
            else:
                pass
            prompts.append(prompt)
            cands.append(options)
            # print(prompt)
                
        with jsonlines.open("data-gpt/memorizing/{}.jsonl".format(knowledge), "w") as f:
            for p, g, i in zip(prompts, golds, cands):
                f.write({"prompt": p, "gold": g, "cands": i})

def generate_reasoning_data(knowledge, rule, index, mode):
    with jsonlines.open("data-gpt/memorizing/{}.jsonl".format(knowledge)) as reader: 
        mem_data = list(reader)
        candidates = [d["cands"] for d in mem_data]

    with jsonlines.open("../data/memorizing/{}.jsonl".format(knowledge)) as reader: 
        data = list(reader)
        golds = list()
        prompts = list()
        MAX = min(520, len(data))
        data = data[20:MAX]
        index = [j-1 for j in index] ##########
        if mode == 1:
            index = list(set(range(len(data))) - set(index))
        if mode == 2:
            index= list(range(len(data)))
        index.sort()

        data = [data[i] for i in index]
        candidates = [candidates[i] for i in index]
        cnt = 0

        for d in data:
            if knowledge == "subPropertyOf":
                if "rdfs7" in rule:
                    prefix, gold = " ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"].values()))).split()), " ".join(re.sub(r'[-()/]',' ', next(iter(d["xxx"].values()))).split())
                else:
                    prefix, gold = "X implies {}. ".format(" ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"]))).split())), " ".join(re.sub(r'[-()/]',' ', next(iter(d["xxx"]))).split())
                if mode == 2:
                    prefix = "{} implies {}. ".format(" ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"]))).split()), gold) + prefix

            elif knowledge == "subClassOf":
                prefix, gold = "X is a " + " ".join(re.sub(r'[-()/]',' ', d["uuu"]).split()), " ".join(re.sub(r'[-()/]',' ', d["xxx"][0]).split())
                if mode == 2:
                    prefix = "{} is a {}. ".format(" ".join(re.sub(r'[-()/]',' ', d["uuu"]).split()), gold) + prefix

            elif knowledge == "domain":
                prefix, gold = " ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"].values()))).split()), " ".join(re.sub(r'[-()/]',' ', d["xxx"][0]).split())
                if mode == 2:
                    prefix = "One has to be a {} {}. ".format(gold, domain_manual_textb(d, r1)) + prefix

            elif knowledge == "range":
                prefix, gold = " ".join(re.sub(r'[-()/]',' ', next(iter(d["uuu"].values()))).split()), " ".join(re.sub(r'[-()/]',' ', d["xxx"][0]).split())
                if mode == 2:
                    prefix = "One has to be a {} {}. ".format(gold, range_manual_textb(d, r2)) + prefix
            
            options = candidates[cnt]
            cnt += 1
            formatted_options = ["({}) {}".format(chr(ord('a') + i), option) for i, option in enumerate(options)]

            if "rdfs2" in rule or "rdfs9" in rule:
                prompt = "{} Therefore, What is the type of X? {}".format(prefix, ", ".join(formatted_options))
            elif "rdfs3" in rule:
                prompt = "{} Therefore, What is the type of Y? {}".format(prefix, ", ".join(formatted_options))
            elif "rdfs5" in rule:
                prompt = "{} Therefore, What does X imply? {}".format(prefix, ", ".join(formatted_options))
            elif "rdfs7" in rule:
                prompt = "{} Therefore, What is the relation between X and Y? {}".format(prefix, ", ".join(formatted_options))
            elif "rdfs11" in rule:
                prompt = "{} Therefore, What is the superclass of X? {}".format(prefix, ", ".join(formatted_options))
            else:
                pass

            prompts.append(prompt.replace("[X]", "X").replace("[Y]", "Y"))
            # print(prompts[-1])
            golds.append(gold)
                
        with jsonlines.open("data-gpt/reasoning/{}-{}.jsonl".format(rule, str(mode)+"2"), "w") as f:
            for p, g in zip(prompts, golds):
                f.write({"prompt": p, "gold": g})

if __name__ == "__main__":
    
    with open("../data/ontology/class.json") as f:
        data = json.loads(f.read())
    for d in data:
        class_list.append(re.sub(r'[-()/]',' ', d["rdfs:label"]))
        class_list += [re.sub(r'[-()/]',' ', word) for word in d["rdfs:subClassOf"]]
    
    with open("../data/ontology/property.json") as f:
        data = json.loads(f.read())
    for d in data:
        prop_list.append(re.sub(r'[()/]',' ', d["rdfs:label"]))
        if d["rdfs:subPropertyOf"]:
            prop_list += [re.sub(r'[()/]',' ', word) for word in d["rdfs:subPropertyOf"].keys()]

    class_list = [' '.join(w.split()) for w in class_list]    # remove redundant space
    class_list = list(dict.fromkeys(class_list))  # remove duplicate candidates

    prop_list = [' '.join(w.split()) for w in prop_list]    # remove redundant space
    prop_list = list(dict.fromkeys(prop_list))  # remove duplicate candidates
    
    with jsonlines.open("../data/memorizing/range.jsonl") as ranges:    # only [Y]->range needs
        for d in ranges:
            r1[list(d["uuu"].keys())[0]] = d["xxx"][0]

    with jsonlines.open("../data/memorizing/domain.jsonl") as domains:
        for d in domains:
            r2[list(d["uuu"].keys())[0]] = d["xxx"][0]

    random.seed(0)

    for k in ["type", "subClassOf", "subPropertyOf", "domain", "range"]:
        generate_memorization_data(k)
        acc = test_memorization("data-gpt/memorizing/{}".format(k))
        print("Acc: ", acc)

        with open("Accuracy.txt", "a") as f:
            f.write("{}: {}\n".format(k, acc))


                


    