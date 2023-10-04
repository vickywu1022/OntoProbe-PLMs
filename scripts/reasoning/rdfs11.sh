#!/bin/bash
BERT_LIST="bert-base-cased bert-base-uncased bert-large-cased bert-large-uncased"
MODE="zs-f fs"
KNOW="rdfs11-"
PREM1="0 1 2"
PREM2="0 1 2"
SOFT="none load"

for B in $BERT_LIST;
do
for m in $MODE;
do
python main.py --knowledge 'subClassOf' --model_class 'bert' --model_path $B --template " is a particular" --multi_token_handler "mean" --mode $m --multi_mask "m" --is_premise True --use_cuda True
python main.py --knowledge 'subClassOf-type' --model_class 'bert' --model_path $B --template " is a particular" --multi_token_handler "mean" --mode $m --multi_mask "m" --is_premise True --use_cuda True
done
done

for B in $BERT_LIST;
do
for m in $MODE;
do
for s in $SOFT;
do
for p1 in $PREM1;
do
for p2 in $PREM2;
do
python main.py --knowledge $KNOW$p1$p2 --model_class 'bert' --model_path $B --template " is a particular"  --multi_token_handler "mean" --mode $m --multi_mask "m" --use_cuda True --trained_soft $s --is_premise True 
done
done
done
done
done



ROBERTA_LIST="roberta-base roberta-large"

for B in $ROBERTA_LIST;
do
for m in $MODE;
do
python main.py --knowledge 'subClassOf' --model_class 'roberta' --model_path $B --template " is a particular" --multi_token_handler "mean" --mode $m --multi_mask "m" --is_premise True --use_cuda True
python main.py --knowledge 'subClassOf-type' --model_class 'roberta' --model_path $B --template " is a particular" --multi_token_handler "mean" --mode $m --multi_mask "m" --is_premise True --use_cuda True
done
done

for B in $ROBERTA_LIST;
do
for m in $MODE;
do
for s in $SOFT;
do
for p1 in $PREM1;
do
for p2 in $PREM2;
do
python main.py --knowledge $KNOW$p1$p2 --model_class 'roberta' --model_path $B --template " is a particular"  --multi_token_handler "mean" --mode $m --multi_mask "m" --use_cuda True --trained_soft $s --is_premise True 
done
done
done
done
done