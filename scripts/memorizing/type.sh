IFS='|'
TEMPLATE=" is a| has class| is a particular"
HANDLER="mean|max|first"
MASK="m|s"
MODE="zs"
for m in $MODE;
do
for MM in $MASK;
do
for h in $HANDLER;
do
for TEMP in $TEMPLATE;
do
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-base-cased' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-base-uncased' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-large-cased' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-large-uncased' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
python main.py --knowledge 'type' --model_class 'roberta' --model_path 'roberta-base' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
python main.py --knowledge 'type' --model_class 'roberta' --model_path 'roberta-large' --template $TEMP --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM
done
done
done
done

MODE="fs"
LOSS="log|NLL"
for m in $MODE;
do
for MM in $MASK;
do
for h in $HANDLER;
do
for l in $LOSS;
do
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-base-cased' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-base-uncased' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-large-cased' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'type' --model_class 'bert' --model_path 'bert-large-uncased' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'type' --model_class 'roberta' --model_path 'roberta-base' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
python main.py --knowledge 'type' --model_class 'roberta' --model_path 'roberta-large' --template " " --use_cuda True --multi_token_handler $h --save_path "result1205.csv" --mode $m --multi_mask $MM --loss $l --soft_init "init"
done
done
done
done