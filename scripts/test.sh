CUDA_VISIBLE_DEVICES=$1 python main.py --test --config cfgs/full/finetune_modelnet.yaml --exp_name $2 --ckpts $3 --seed $RANDOM 
