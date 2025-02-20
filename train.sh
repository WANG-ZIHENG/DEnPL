#!/bin/bash
#少样本优先
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset breastmnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset breastmnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset bloodmnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset bloodmnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organamnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organamnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organsmnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organsmnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organcmnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset organcmnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset pneumoniamnist --CE_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset pneumoniamnist --CE_loss_use --CCL_loss_use
#python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset dermamnist --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset OCT2017 --CE_loss_use --CCL_loss_use
# python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset chest_xray --CE_loss_use --CCL_loss_use

#python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset tissuemnist --CE_loss_use --CCL_loss_use
#python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset octmnist --CE_loss_use --CCL_loss_use
#python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset PathMNIST --CE_loss_use --CCL_loss_use
#python Source/Try_DenseNet-121_pretrained-Copy1.py --dataset chestmnist --CE_loss_use --CCL_loss_use

# python Source/Train.py --dataset bloodmnist --CE_loss_use  --model resnet50
# python Source/Train.py --dataset organamnist --CE_loss_use
# python Source/Train.py --dataset organsmnist --CE_loss_use
# python Source/Train.py --dataset organcmnist --CE_loss_use
# python Source/Train.py --dataset pneumoniamnist --CE_loss_use





#21408 master
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --CE_loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --LDAM_loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --CCE_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --LOW_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --GHMC_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --MWN_Loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --CE_loss_use --supcon_loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
python Source/Train.py  --dataset BreastMNIST --other_test_dataset --model resnet50 --CE_loss_use --CCL_loss_use --batch-size 24 --learning-rate 0.0001 --epochs 100
#echo "3分钟后关机..."
#sleep 1
#shutdown
#实验
#python Source/Train.py --dataset cifar10 --model resnet10 --CE_loss_use --CCL_loss_use --uncertain_use  --IF 100 --learning-rate 1e-3 --epochs 60
#python Source/Train.py --dataset cifar10 --model resnet32 --CE_loss_use --CCL_loss_use --uncertain_use  --IF 100 --learning-rate 1e-3 --epochs 200
#python Source/Train.py --dataset cifar10 --model resnet32 --CE_loss_use --CCL_loss_use   --IF 100 --learning-rate 1e-3 --epochs 200
#python Source/Train.py --dataset GastroVision --model DenseNet121 --pretrain_model --CE_loss_use --CCL_loss_use   --batch-size 24 --learning-rate 0.0001 --epochs 60
#shutdown

##机器1  10530
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 50 --CE_loss_use --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 50 --CE_loss_use --CCL_loss_use  --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 50 --CE_loss_use --CCL_loss_use --feature_similarity_use --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 10 --CE_loss_use --CCL_loss_use --feature_similarity_use --learning-rate 1e-3  --epochs 200
#shutdown
#机器2 32780

#python Source/Train.py --dataset cifar100 --model resnet32 --IF 50 --CE_loss_use --CCL_loss_use --uncertain_use --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 50 --CE_loss_use --CCL_loss_use --feature_similarity_use --uncertain_use  --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 10 --CE_loss_use --CCL_loss_use --feature_similarity_use --uncertain_use  --learning-rate 1e-3  --epochs 200
#shutdown
#
#机器3 52367
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 10 --CE_loss_use --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 10 --CE_loss_use --CCL_loss_use  --learning-rate 1e-3  --epochs 200
#python Source/Train.py --dataset cifar100 --model resnet32 --IF 10 --CE_loss_use --CCL_loss_use --uncertain_use --learning-rate 1e-3  --epochs 200
#
#
#shutdown
#



#全部
#python Source/Train.py --dataset tissuemnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset OCT2017 --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset chest_xray --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset PathMNIST --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset chestmnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset dermamnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset octmnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset pneumoniamnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset breastmnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset bloodmnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset organamnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset organcmnist --CE_loss_use --CCL_loss_use
#python Source/Train.py --dataset organsmnist --CE_loss_use --CCL_loss_use

#shutdown


