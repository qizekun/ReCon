CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/full/finetune_scan_hardest.yaml \
--finetune_model --exp_name $2 --ckpts $3 --seed $RANDOM 
