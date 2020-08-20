#!/bin/bash
PATH=/usr/local/anaconda3/bin:/usr/local/anaconda3/condabin:/root/bin/autossh/bin:/root/bin/autossh/bin: \
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:
# not sure this is needed
# conda activate
echo -e "run experiment 1"
# python test.py --model 'FS_ResNet' --k_shot 1 --check_file './checkpoints/FS_ResNet34_hmdb51_SGD_lr_0.01_epoch_100' \
# --n_way 5 --test_episode 600
#python train.py --model 'FS_MENet'  --epi_train --stop_epoch 400
#echo -e "run experiment 2"
#python train.py --model 'FS_ResNet' --epi_train --stop_epoch 400 --backbone_size 50
for((i = 5;i <= 10;i++));
do
  check_num=$((i*10))
  echo -e "current checkpoint model is $check_num"
  if [ ${check_num} -eq 100 ]; then
    python test.py --model 'FS_MENet' --k_shot 1 --check_file './chechpoints/FS_ResNet34_hmdb51_SGD_lr_0.1_epi_True_epoch_100' --checkpoint ${check_num}-1
    python test.py --model 'FS_MENet' --k_shot 5 --check_file './chechpoints/FS_ResNet34_hmdb51_SGD_lr_0.1_epi_True_epoch_100' --checkpoint ${check_num}-1
  else
    python test.py --model 'FS_MENet' --k_shot 1 --check_file './chechpoints/FS_ResNet34_hmdb51_SGD_lr_0.1_epi_True_epoch_100' --checkpoint ${check_num}
    python test.py --model 'FS_MENet' --k_shot 5 --check_file './chechpoints/FS_ResNet34_hmdb51_SGD_lr_0.1_epi_True_epoch_100' --checkpoint ${check_num}
  fi
done
# python train.py --model 'FS_ResNet' --epi_train  --backbone_size 50 --stop_epoch 400 --lr 0.01
#python train.py --model 'FS_MENet' --epi_train  --backbone_size 50 --stop_epoch 200 --lr 0.01