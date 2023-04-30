CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/zeroshot/modelnet40.yaml \
--zeroshot --exp_name $2 --ckpts $3
