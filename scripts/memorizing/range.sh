MODE="zs"
HANDLER="mean max first"
MASK="m s"

for m in $MODE;
do
for MM in $MASK;
do
for h in $HANDLER;
do
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-base-cased' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-base-uncased' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-large-cased' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-large-uncased' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
python main.py --knowledge 'range' --model_class 'roberta' --model_path 'roberta-base' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
python main.py --knowledge 'range' --model_class 'roberta' --model_path 'roberta-large' --template "One has to be a particular" --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM
done
done
done

MODE="fs"
LOSS="log NLL"
for m in $MODE;
do
for MM in $MASK;
do
for h in $HANDLER;
do
for l in $LOSS;
do
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-base-cased' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-base-uncased' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-large-cased' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'range' --model_class 'bert' --model_path 'bert-large-uncased' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'range' --model_class 'roberta' --model_path 'roberta-base' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'range' --model_class 'roberta' --model_path 'roberta-large' --template " " --use_cuda True --multi_token_handler $h --mode $m --multi_mask $MM --loss $l --soft_init "init"
done
done
done
done
