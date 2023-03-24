# Installation

```shell
git clone 
# Datasets
cp -r /mnt/petrelfs/zhaolin/vln/mp3d/datasets to '__<path_to_project>'
```
- organize the dataset files and code as follows:
```shell
'__<path_to_project>'
├── datasets
│   ├── pretrained
│   ├── R2R
│   └── SOON
├── NavLLM
│   ├── llm
│   ├── Readme.md
│   └── requirements.txt
```

## Env
```shell
Requirements:
    - python 3.8
    - Matterport3D S2集群可安装版本: https://github.com/MuMuJun97/Matterport3DSimulator-Centos7
    - pip library ref: requirements.txt
```


## Training
```shell
cd NavLLM/llm/
# llm/cfg/soon_obj_pretrain_qa.json: batch size, dataset_dir configs
srun -p OpenDialogLab_S2 --gres=gpu:1 python train.py --Net google/flan-t5-small
# or python train.py --Net google/flan-t5-large ...
```

# TODO 
- pretraining & fine-tuning & evaluation, ref: https://github.com/cshizhe/VLN-DUET