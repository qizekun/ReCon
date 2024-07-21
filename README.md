# 🪖 ReCon: Contrast with Reconstruct
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/3d-point-cloud-linear-classification-on)](https://paperswithcode.com/sota/3d-point-cloud-linear-classification-on?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/3d-point-cloud-classification-on-scanobjectnn)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-scanobjectnn?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/3d-point-cloud-classification-on-modelnet40)](https://paperswithcode.com/sota/3d-point-cloud-classification-on-modelnet40?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/few-shot-3d-point-cloud-classification-on-1)](https://paperswithcode.com/sota/few-shot-3d-point-cloud-classification-on-1?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/zero-shot-transfer-3d-point-cloud)](https://paperswithcode.com/sota/zero-shot-transfer-3d-point-cloud?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/zero-shot-transfer-3d-point-cloud-1)](https://paperswithcode.com/sota/zero-shot-transfer-3d-point-cloud-1?p=contrast-with-reconstruct-contrastive-3d)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/contrast-with-reconstruct-contrastive-3d/zero-shot-transfer-3d-point-cloud-2)](https://paperswithcode.com/sota/zero-shot-transfer-3d-point-cloud-2?p=contrast-with-reconstruct-contrastive-3d)

> [**Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining**](https://arxiv.org/abs/2302.02318) **ICML 2023** <br>
> [Zekun Qi](https://scholar.google.com/citations?user=ap8yc3oAAAAJ)\*, [Runpei Dong](https://runpeidong.com/)\*, [Guofan Fan](https://github.com/Asterisci), [Zheng Ge](https://scholar.google.com.hk/citations?user=hJ-VrrIAAAAJ&hl=en&oi=ao), [Xiangyu Zhang](https://scholar.google.com.hk/citations?user=yuB-cfoAAAAJ&hl=en&oi=ao), [Kaisheng Ma](http://group.iiis.tsinghua.edu.cn/~maks/leader.html) and [Li Yi](https://ericyi.github.io/) <br>

[OpenReview](https://openreview.net/forum?id=80IfYewOh1) | [arXiv](https://arxiv.org/abs/2302.02318) | [Models](https://drive.google.com/drive/folders/17Eoy5N96dcTQJplCOjyeeVjSYyjW5QEd?usp=share_link)

This repository contains the code release of ReCon: **Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining** (ICML 2023). ReCon is also short for *reconnaissance* 🪖.

## Contrast with Reconstruct (ICML 2023)

[//]: # (Mainstream 3D representation learning approaches are built upon contrastive or generative modeling pretext tasks, where great improvements in performance on various downstream tasks have been achieved. However, by investigating the methods of these two paradigms, we find that &#40;i&#41; contrastive models are data-hungry that suffer from a representation over-fitting issue; &#40;ii&#41; generative models have a data filling issue that shows inferior data scaling capacity compared to contrastive models. This motivates us to learn 3D representations by sharing the merits of both paradigms, which is non-trivial due to the pattern difference between the two paradigms. In this paper, we propose *contrast with reconstruct* &#40;**ReCon**&#41; that unifies these two paradigms. ReCon is trained to learn from both generative modeling teachers and cross-modal contrastive teachers through ensemble distillation, where the generative student is used to guide the contrastive student. An encoder-decoder style ReCon-block is proposed that transfers knowledge through cross attention with stop-gradient, which avoids pretraining over-fitting and pattern difference issues. ReCon achieves a new state-of-the-art in 3D representation learning, e.g., 91.26% accuracy on ScanObjectNN.)

<div  align="center">    
 <img src="./figure/framework.png" width = "1100"  align=center />
</div>


## News

- 🍾 July, 2024: [**ShapeLLM (ReCon++)**](https://qizekun.github.io/shapellm/) accepted by ECCV 2024, check out the [code](https://github.com/qizekun/ShapeLLM)
- 💥 Mar, 2024: Check out our latest work [**ShapeLLM (ReCon++)**](https://qizekun.github.io/shapellm/), which achieves **95.25%** fine-tuned accuracy and **65.4** zero-shot accuracy on ScanObjectNN
- 📌 Aug, 2023: Check out our exploration of efficient conditional 3D generation [**VPP**](https://arxiv.org/abs/2307.16605)
- 📌 Jun, 2023: Check out our exploration of pre-training in 3D scenes [**Point-GCC**](https://arxiv.org/abs/2305.19623)
- 🎉 Apr, 2023: [**ReCon**](https://arxiv.org/abs/2302.02318) accepted by ICML 2023
- 💥 Feb, 2023: Check out our previous work [**ACT**](https://arxiv.org/abs/2212.08320), which has been accepted by ICLR 2023

## 1. Requirements
PyTorch >= 1.7.0;
python >= 3.7;
CUDA >= 9.0;
GCC >= 4.9;
torchvision;

```
# Quick Start
conda create -n recon python=3.10 -y
conda activate recon

conda install pytorch==2.0.1 torchvision==0.15.2 cudatoolkit=11.8 -c pytorch -c nvidia
# pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
```

```
# Install basic required packages
pip install -r requirements.txt
# Chamfer Distance
cd ./extensions/chamfer_dist && python setup.py install --user
# PointNet++
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

## 2. Datasets

We use ShapeNet, ScanObjectNN, ModelNet40 and ShapeNetPart in this work. See [DATASET.md](./DATASET.md) for details.

## 3. ReCon Models
| Task              | Dataset        | Config                                                               | Acc.       | Checkpoints Download                                                                                     |
|-------------------|----------------|----------------------------------------------------------------------|------------|----------------------------------------------------------------------------------------------------------|
| Pre-training      | ShapeNet       | [pretrain_base.yaml](cfgs/pretrain/base.yaml)                        | N.A.       | [ReCon](https://drive.google.com/file/d/1L-TlZUi7umBCDpZW-1F0Gf4X-9Wvf_Zo/view?usp=share_link)           |
| Classification    | ScanObjectNN   | [finetune_scan_hardest.yaml](./cfgs/full/finetune_scan_hardest.yaml) | 91.26%     | [PB_T50_RS](https://drive.google.com/file/d/1kjKqvs8o6jiqZc4-srMFp2DOpIYDHdgf/view?usp=share_link)       |
| Classification    | ScanObjectNN   | [finetune_scan_objbg.yaml](./cfgs/full/finetune_scan_objbg.yaml)     | 95.35%     | [OBJ_BG](https://drive.google.com/file/d/1qjohpaTCl-DzHaIv6Ilq0sLAGG2H3Z9I/view?usp=share_link)          |
| Classification    | ScanObjectNN   | [finetune_scan_objonly.yaml](./cfgs/full/finetune_scan_objonly.yaml) | 93.80%     | [OBJ_ONLY](https://drive.google.com/file/d/1kvowgPbvlFxx3B5WSfL3LiKZ5--5s52b/view?usp=share_link)        |
| Classification    | ModelNet40(1k) | [finetune_modelnet.yaml](./cfgs/full/finetune_modelnet.yaml)         | 94.5%      | [ModelNet_1k](https://drive.google.com/file/d/1UsRuIc7ND2n4PjYyF3n0tT3hf7alyOML/view?usp=share_link)     |
| Classification    | ModelNet40(8k) | [finetune_modelnet_8k.yaml](./cfgs/full/finetune_modelnet_8k.yaml)   | 94.7%      | [ModelNet_8k](https://drive.google.com/file/d/1qUuT6sjhZw3gn0rFj-qDfYG6VMBXF0sT/view?usp=share_link)     |
| Zero-Shot         | ModelNet10     | [zeroshot_modelnet10.yaml](./cfgs/zeroshot/modelnet10.yaml)          | 75.6%      | [ReCon zero-shot](https://drive.google.com/file/d/1Xz6lZn6MI2lJldPiSqLdAnjEm0dwQ4Mg/view?usp=share_link) |
| Zero-Shot         | ModelNet10*    | [zeroshot_modelnet10.yaml](./cfgs/zeroshot/modelnet10.yaml)          | 81.6%      | [ReCon zero-shot](https://drive.google.com/file/d/1Xz6lZn6MI2lJldPiSqLdAnjEm0dwQ4Mg/view?usp=share_link) |
| Zero-Shot         | ModelNet40     | [zeroshot_modelnet40.yaml](./cfgs/zeroshot/modelnet40.yaml)          | 61.7%      | [ReCon zero-shot](https://drive.google.com/file/d/1Xz6lZn6MI2lJldPiSqLdAnjEm0dwQ4Mg/view?usp=share_link) |
| Zero-Shot         | ModelNet40*    | [zeroshot_modelnet40.yaml](./cfgs/zeroshot/modelnet40.yaml)          | 66.8%      | [ReCon zero-shot](https://drive.google.com/file/d/1Xz6lZn6MI2lJldPiSqLdAnjEm0dwQ4Mg/view?usp=share_link) |
| Zero-Shot         | ScanObjectNN   | [zeroshot_scan_objonly.yaml](./cfgs/zeroshot/scan_objonly.yaml)      | 43.7%      | [ReCon zero-shot](https://drive.google.com/file/d/1Xz6lZn6MI2lJldPiSqLdAnjEm0dwQ4Mg/view?usp=share_link) |
| Linear SVM        | ModelNet40     | [svm.yaml](./cfgs/svm/modelnet40.yaml)                               | 93.4%      | [ReCon svm](https://drive.google.com/file/d/1SvCfDzXM2QM7BfOd960z3759HY_c-eQv/view?usp=share_link)       |
| Part Segmentation | ShapeNetPart   | [segmentation](./segmentation)                                       | 86.4% mIoU | [part seg](https://drive.google.com/file/d/13XuEsN7BDu-YX86ZSM1SpUHGMvDys2VH/view?usp=share_link)        |

| Task              | Dataset    | Config                                   | 5w10s (%)  | 5w20s (%)  | 10w10s (%) | 10w20s (%) | Download                                                                                       |
|-------------------|------------|------------------------------------------|------------|------------|------------|------------|------------------------------------------------------------------------------------------------|
| Few-shot learning | ModelNet40 | [fewshot.yaml](./cfgs/full/fewshot.yaml) | 97.3 ± 1.9 | 98.9 ± 1.2 | 93.3 ± 3.9 | 95.8 ± 3.0 | [ReCon](https://drive.google.com/file/d/1L-TlZUi7umBCDpZW-1F0Gf4X-9Wvf_Zo/view?usp=share_link) |

The checkpoints and logs have been released on [Google Drive](https://drive.google.com/drive/folders/17Eoy5N96dcTQJplCOjyeeVjSYyjW5QEd?usp=share_link). You can use the voting strategy in classification testing to reproduce the performance reported in the paper.
For classification downstream tasks, we randomly select 8 seeds to obtain the best checkpoint. 
For zero-shot learning, * means that we use all the train/test data for zero-shot transfer.

## 4. ReCon Pre-training
Pre-training with the default configuration, run the script:
```
sh scripts/pretrain.sh <GPU> <exp_name>
```
If you want to try different models or masking ratios etc., first create a new config file, and pass its path to --config.
```
CUDA_VISIBLE_DEVICES=<GPU> python main.py --config <config_path> --exp_name <exp_name>
```
## 5. ReCon Classification Fine-tuning
Fine-tuning with the default configuration, run the script:
```
bash scripts/cls.sh <GPU> <exp_name> <path/to/pre-trained/model>
```
Or, you can use the command.

Fine-tuning on ScanObjectNN, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/full/finetune_scan_hardest.yaml \
--finetune_model --exp_name <exp_name> --ckpts <path/to/pre-trained/model>
```
Fine-tuning on ModelNet40, run:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/full/finetune_modelnet.yaml \
--finetune_model --exp_name <exp_name> --ckpts <path/to/pre-trained/model>
```
## 6. ReCon Test&Voting
Test&Voting with the default configuration, run the script:
```
bash scripts/test.sh <GPU> <exp_name> <path/to/best/fine-tuned/model>
```
or:
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --test --config cfgs/finetune_modelnet.yaml \
--exp_name <output_file_name> --ckpts <path/to/best/fine-tuned/model>
```
## 7. ReCon Few-Shot
Few-shot with the default configuration, run the script:
```
sh scripts/fewshot.sh <GPU> <exp_name> <path/to/pre-trained/model> <way> <shot> <fold>
```
or
```
CUDA_VISIBLE_DEVICES=<GPUs> python main.py --config cfgs/full/fewshot.yaml --finetune_model \
--ckpts <path/to/pre-trained/model> --exp_name <exp_name> --way <5 or 10> --shot <10 or 20> --fold <0-9>
```
## 8. ReCon Zero-Shot
Zero-shot with the default configuration, run the script:
```
bash scripts/zeroshot.sh <GPU> <exp_name> <path/to/pre-trained/model>
```
## 9. ReCon Part Segmentation
Part segmentation on ShapeNetPart, run:
```
cd segmentation
bash seg.sh <GPU> <exp_name> <path/to/pre-trained/model>
```
or
```
cd segmentation
python main.py --ckpts <path/to/pre-trained/model> --log_dir <path/to/log/dir> --learning_rate 0.0001 --epoch 300
```
Test part segmentation on ShapeNetPart, run:
```
cd segmentation
bash test.sh <GPU> <exp_name> <path/to/best/fine-tuned/model>
```
## 10. ReCon Linear SVM
Linear SVM on ModelNet40, run:
```
sh scripts/svm.sh <GPU> <exp_name> <path/to/pre-trained/model> 
```

## 11. Visualization
We use [PointVisualizaiton](https://github.com/qizekun/PointVisualizaiton) repo to render beautiful point cloud image, including specified color rendering and attention distribution rendering.


## Contact

If you have any questions related to the code or the paper, feel free to email Zekun (`qizekun@gmail.com`) or Runpei (`runpei.dong@gmail.com`). 

## License

ReCon is released under MIT License. See the [LICENSE](./LICENSE) file for more details. Besides, the licensing information for `pointnet2` modules is available [here](https://github.com/erikwijmans/Pointnet2_PyTorch/blob/master/UNLICENSE).

## Acknowledgements

This codebase is built upon [Point-MAE](https://github.com/Pang-Yatian/Point-MAE), [Point-BERT](https://github.com/lulutang0608/Point-BERT), [CLIP](https://github.com/openai/CLIP), [Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch) and [ACT](https://github.com/RunpeiDong/ACT)

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{qi2023recon,
  title={Contrast with Reconstruct: Contrastive 3D Representation Learning Guided by Generative Pretraining},
  author={Qi, Zekun and Dong, Runpei and Fan, Guofan and Ge, Zheng and Zhang, Xiangyu and Ma, Kaisheng and Yi, Li},
  booktitle={International Conference on Machine Learning (ICML) },
  year={2023}
}
```
and closely related work [ACT](https://github.com/RunpeiDong/ACT) and [ShapeLLM](https://github.com/qizekun/ShapeLLM):
```bibtex
@inproceedings{dong2023act,
  title={Autoencoders as Cross-Modal Teachers: Can Pretrained 2D Image Transformers Help 3D Representation Learning?},
  author={Runpei Dong and Zekun Qi and Linfeng Zhang and Junbo Zhang and Jianjian Sun and Zheng Ge and Li Yi and Kaisheng Ma},
  booktitle={The Eleventh International Conference on Learning Representations (ICLR) },
  year={2023},
  url={https://openreview.net/forum?id=8Oun8ZUVe8N}
}
@inproceedings{qi2024shapellm,
  author = {Qi, Zekun and Dong, Runpei and Zhang, Shaochen and Geng, Haoran and Han, Chunrui and Ge, Zheng and Yi, Li and Ma, Kaisheng},
  title = {ShapeLLM: Universal 3D Object Understanding for Embodied Interaction},
  booktitle={European Conference on Computer Vision (ECCV) },
  year = {2024}
}
```
