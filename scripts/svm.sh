CUDA_VISIBLE_DEVICES=$1 python main.py --config cfgs/svm/modelnet40.yaml \
--svm --exp_name $2 --ckpts $3
