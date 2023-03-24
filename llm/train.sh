# script: 
# source ~/env/py38
# nohup srun -p OpenDialogLab_S2 --gres=gpu:1 python train.py > ../../output/logs/train/train_v1.log 2>&1 &