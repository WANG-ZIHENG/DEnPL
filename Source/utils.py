import pickle

import math
import numpy as np
from torchvision import transforms, datasets, models
import os
import mydatasets
from medmnist import INFO, Evaluator
import medmnist
from collections import defaultdict
import random
from data_pro import split_cifar10, split_cifar100
from torch.utils import data
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split
import pandas as pd
class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]
def adjust_lr(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = lr / args.warmup_epochs * (epoch + 1)
    elif args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs + 1) / (args.max_epochs - args.warmup_epochs + 1)))

    # if lr < 0.001:
    #     lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# def adjust_lr(optimizer, epoch, args):
#     """Decay the learning rate based on schedule"""
#     lr = args.lr
#     if epoch < args.warmup_epochs:
#         lr = 0.15
#     elif epoch < 160:
#         lr = 1e-4
#     elif epoch <= 180:
#         lr = 1e-5
#
#     elif epoch <= 200:
#         lr = 1e-6
#
#     # if lr < 0.001:
#     #     lr = 0.001
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
class CifarDataset(data.Dataset):
    """Dataset class for the Imagenet-LT dataset."""

    def __init__(self, data_dir, cls_num, transform, mode):
        """Initialize and preprocess the Imagenet-LT dataset."""
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.val_dataset = []
        self.test_dataset = []
        self.train_txt = '%s/train.csv' % data_dir
        self.val_txt = '%s/val.csv' % data_dir
        self.test_txt = '%s/test.csv' % data_dir
        self.cls_num_list_train = [0] * cls_num
        self.cls_num_list_val = [0] * cls_num
        self.cls_num_list_test = [0] * cls_num
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.train_dataset)
            ds = self.train_dataset
        elif mode == "val":
            self.num_images = len(self.val_dataset)
            ds = self.val_dataset
        else:
            self.num_images = len(self.test_dataset)
            ds = self.test_dataset



        classes_num = {}
        for path,class_name in ds:
            class_name = str(class_name)

            if class_name not in classes_num:
                classes_num[class_name] = 1
            else:
                classes_num[class_name] += 1
        self.cls_num_list = list(classes_num.values())
        self.class_names = list(classes_num.keys())
        self.id_to_name = {int(i):name for i,name in enumerate(self.class_names)}


    def preprocess(self):
        train_file = open(self.train_txt, "r")
        val_file = open(self.val_txt, "r")
        test_file = open(self.test_txt, "r")

        for elem in train_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.train_dataset.append([filename, label])
            self.cls_num_list_train[label] += 1

        for elem in val_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.val_dataset.append([filename, label])
            self.cls_num_list_val[label] += 1

        for elem in test_file.readlines()[1:]:
            filename = elem.split(',')[0]
            if filename.startswith('data/'):
                filename = filename[5:]
            filename = '%s/%s' % (self.data_dir, filename)
            label = int(elem.split(',')[1])
            self.test_dataset.append([filename, label])
            self.cls_num_list_test[label] += 1

    def __getitem__(self, index):
        """Return one image and its corresponding label."""
        if self.mode == "train":
            dataset = self.train_dataset
        elif self.mode == "val":
            dataset = self.val_dataset
        else:
            dataset = self.test_dataset

        filename, label = dataset[index]
        image = Image.open(filename).convert('RGB')

        return self.transform(image), label

    def __len__(self):
        """Return the number of images."""
        return self.num_images

    def get_cls_num_list(self):
        if self.mode == "train":
            print(self.cls_num_list_train)
            return self.cls_num_list_train
        elif self.mode == "val":
            return self.cls_num_list_val
        else:
            return self.cls_num_list_test


class ISICDataset():

    def __init__(self,path_list,metadata):
        self.data = path_list

        id_path_dict = {os.path.basename(i).split(".")[0]:i  for i in self.data}

        skin_df = pd.read_csv(metadata)
        skin_df = skin_df.iloc[:, :9]
        id_label_dict = {}
        for i in range(len(skin_df)):
            row = skin_df.iloc[i].values
            id = row[0]
            one_hot =  row[1:]
            if sum(one_hot) == 0:
                continue
            label = np.argmax(one_hot)
            id_label_dict[id] = label


        self.data = []
        self.targets = []
        for k,v in id_path_dict.items():
            if k in id_label_dict:
                self.data.append(v)
                self.targets.append(id_label_dict[k])
            else:
                continue

        self.data = np.array(self.data)
        self.targets = np.array(self.targets)



        self.class_name = skin_df.columns[1:].values
        self.class_to_idx = {name:i for i,name in enumerate(self.class_name)}


class BUSBRADataset():

    def __init__(self,path_list,metadata):
        self.data = path_list
        self.targets = []
        self.class_name = {0:"malignant",1:'benign'}
        name_to_id = {v:k for k,v in self.class_name.items()}

        df = pd.read_csv(metadata)



        for i in self.data:
            id = os.path.basename(i).replace(".png","")
            classname = df[df['ID']==id]['Pathology'].values[0]
            self.targets.append(name_to_id[classname])



        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.class_to_idx = {name:i for i,name in self.class_name.items()}





def update_class_sample_counts(dataset):
    class_counts = defaultdict(int)
    for _, target in dataset.samples:
        class_counts[target] += 1
    cls_num_list = [class_counts[class_idx] for class_idx in sorted(class_counts)]
    return cls_num_list
def get_datasets(args):
    trans = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.RandomRotation(10, expand=False, center=None, fill=0),  #测试了180度，效果很差

                transforms.Resize((224, 224)),
                # transforms.RandomResizedCrop((224, 224)),

                # transforms.RandomApply([
                #     transforms.RandomAffine(degrees=30, translate=(0, 0.2), scale=(0.9, 1), shear=45),
                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                # ], p=0.8),
                # transforms.RandomGrayscale(p=0.5),
                # transforms.RandomVerticalFlip(p=0.5),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.4762, 0.3054, 0.2368],
                                     [0.3345, 0.2407, 0.2164])
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4762, 0.3054, 0.2368],
                                     [0.3345, 0.2407, 0.2164])
            ]),

        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4762, 0.3054, 0.2368],
                                     [0.3345, 0.2407, 0.2164])
            ]),

        'train_cifar100' : transforms.Compose([
        transforms.ToTensor(),  # 转化为tensor类型

        # 从[0,1]归一化到[-1,1]
            transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomRotation(15, expand=True, center=None, fill=0),
        transforms.RandomHorizontalFlip(),  # 随机水平镜像

        transforms.RandomErasing(p=0.5,scale=(0.1, 0.2), ratio=(0.5, 3.3)),  # 随机遮挡
        transforms.RandomCrop(224, padding=4),  # 随机裁剪

    ]),

    'test_cifar100' :transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224)),
    ]),

        'train_cifar10' : transforms.Compose([
        transforms.ToTensor(),  # 转化为tensor类型

        # 从[0,1]归一化到[-1,1]
            transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomRotation(15, expand=True, center=None, fill=0),
        transforms.RandomHorizontalFlip(),  # 随机水平镜像
        transforms.RandomErasing(scale=(0.04, 0.2), ratio=(0.5, 2)),  # 随机遮挡
        transforms.RandomCrop(224, padding=4),  # 随机裁剪

    ]),

    'test_cifar10' :transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize((224, 224)),
    ])


    }
    data_root = args.data_root

    if args.dataset == "GastroVision":
        train_root_dir = f"GLMC-2023/data/GastroVision/lack_5_class/train"
        val_root_dir = f"GLMC-2023/data/GastroVision/lack_5_class/validate"
        test_root_dir = f"GLMC-2023/data/GastroVision/lack_5_class/test"


    elif args.dataset == "OCT2017":
        train_root_dir = f"OCT2017/train"
        val_root_dir = f"OCT2017/test"
        test_root_dir = f"OCT2017/test"
    elif args.dataset == "chest_xray":
        train_root_dir = f"chest_xray/train"
        val_root_dir = f"chest_xray/test"
        test_root_dir = f"chest_xray/test"
    elif args.dataset == "cifar10":
        data_dir = os.path.join(args.data_root,'cifar_data')
        args.cls_num = 10
        data_dir = '%s/cifar10' % data_dir
        split_cifar10(data_dir, IF=args.IF)
        training_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['train_cifar10']), 'train')
        validation_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['test_cifar10']), 'val')
        test_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['test_cifar10']), 'test')
        training_dataset.task = 'multi-class'
        validation_dataset.task = 'multi-class'
        test_dataset.task = 'multi-class'
        training_dataset.cls_num_list = training_dataset.get_cls_num_list()
        test_dataset.cls_num_list = test_dataset.get_cls_num_list()
        validation_dataset.cls_num_list = validation_dataset.get_cls_num_list()

        return training_dataset, test_dataset, validation_dataset
    elif args.dataset == "cifar100":
        data_dir = os.path.join(args.data_root,'cifar_data')
        args.cls_num = 100
        data_dir = '%s/cifar100' % data_dir
        split_cifar100(data_dir,IF=args.IF)
        training_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['train_cifar100']), 'train')
        validation_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['test_cifar100']), 'val')
        test_dataset = CifarDataset(data_dir, args.cls_num, TwoCropTransform(trans['test_cifar100']), 'test')
        training_dataset.task = 'multi-class'
        validation_dataset.task = 'multi-class'
        test_dataset.task = 'multi-class'
        training_dataset.cls_num_list = training_dataset.get_cls_num_list()
        test_dataset.cls_num_list = test_dataset.get_cls_num_list()
        validation_dataset.cls_num_list = validation_dataset.get_cls_num_list()



        return training_dataset, test_dataset, validation_dataset

    elif args.dataset == "ISIC":
        data_dir = os.path.join(args.data_root, 'ISIC')
        train_val_image_path = glob(os.path.join(data_dir,"ISIC_2019_Training_Input","*.jpg"))
        test_image_path = glob(os.path.join(data_dir,"ISIC_2019_Test_Input","*.jpg"))
        random.seed(2024)
        random.shuffle(train_val_image_path)
        train_image_len = len(train_val_image_path)
        train_list = train_val_image_path[:int(train_image_len*0.8)]
        validation_list = train_val_image_path[int(train_image_len*0.8):]

        training_dataset = ISICDataset(train_list,os.path.join(data_dir,"ISIC_2019_Training_GroundTruth.csv"))
        validation_dataset = ISICDataset(validation_list,os.path.join(data_dir,"ISIC_2019_Training_GroundTruth.csv"))
        test_dataset = ISICDataset(test_image_path,os.path.join(data_dir,"ISIC_2019_Test_GroundTruth.csv"))
        training_dataset = mydatasets.WrapperDataset(data_class=training_dataset, transform=TwoCropTransform(trans['train']),task='multi-class')
        validation_dataset = mydatasets.WrapperDataset(data_class=validation_dataset, transform=TwoCropTransform(trans['valid']),task='multi-class')
        test_dataset = mydatasets.WrapperDataset(data_class=test_dataset, transform=TwoCropTransform(trans['test']),task='multi-class')
        return training_dataset, test_dataset, validation_dataset
    elif args.dataset.lower() == "breastmnist":
        if not os.path.exists(data_root):
            os.makedirs(data_root)
        data_flag = args.dataset.lower()

        info = INFO[data_flag]
        if 'binary-class' not in info["task"] and 'multi-class' not in info["task"]:
            print(f'task:{info["task"]}')
            raise "Unsupported dataset task"
        DataClass = getattr(medmnist, info['python_class'])

        # validation_dataset = DataClass(split='val', as_rgb=True,download=True, size=224, transform=TwoCropTransform(trans['valid']))
        # test_dataset = DataClass(split='test', as_rgb=True, download=True, size=224, transform=TwoCropTransform(trans['test']))
        # training_dataset = DataClass(split='train', as_rgb=True, download=True,size=224,transform=TwoCropTransform(trans['train']))
        validation_dataset = DataClass(split='val', root=data_root, download=True,
                                       transform=TwoCropTransform(trans['valid']), as_rgb=True, size=224)

        training_dataset = DataClass(split='train', root=data_root, download=True,
                                     transform=TwoCropTransform(trans['train']), as_rgb=True, size=224)

        if args.other_test_dataset:
            test_root_dir = f"BUSBRA"
            print(f"当前使用的测试集：{test_root_dir}")

            data_dir = os.path.join(args.data_root,test_root_dir, "Images")
            test_image_path = glob(os.path.join(data_dir, "*.png"))


            test_dataset = BUSBRADataset(test_image_path,os.path.join(args.data_root,test_root_dir, "bus_data.csv"))
            test_dataset = mydatasets.WrapperDataset(data_class=test_dataset, transform=TwoCropTransform(trans['test']),
                                                     task=info["task"])
            test_dataset.labels = test_dataset.targets
        else:

            test_dataset = DataClass(split='test', root=data_root, download=True, transform=TwoCropTransform(trans['test']),
                                     as_rgb=True, size=224)
            print(f"当前使用的测试集：breastmnist")
            test_dataset.labels = [i[0] for i in test_dataset.labels]


        id_to_class = training_dataset.info["label"]
        classes_num = {}
        training_dataset.labels = [i[0] for i in training_dataset.labels]

        validation_dataset.labels = [i[0] for i in validation_dataset.labels]
        for i in training_dataset.labels:
            i = str(i)
            class_name = id_to_class[i]
            if class_name not in classes_num:
                classes_num[class_name] = 1
            else:
                classes_num[class_name] += 1

        training_dataset.cls_num_list = list(classes_num.values())
        test_dataset.cls_num_list = list(classes_num.values())
        validation_dataset.cls_num_list = list(classes_num.values())
        validation_dataset.task = info["task"]
        test_dataset.task = info["task"]
        training_dataset.task = info["task"]
        test_dataset.class_names = list(training_dataset.info["label"].values())
        test_dataset.id_to_name = {int(k): v for k, v in training_dataset.info["label"].items()}

        return training_dataset, test_dataset, validation_dataset

    elif "mnist"   in args.dataset.lower() :

        if not os.path.exists(data_root):
            os.makedirs(data_root)
        data_flag = args.dataset.lower()

        info = INFO[data_flag]
        if 'binary-class' not in info["task"] and 'multi-class' not in info["task"]:
            print(f'task:{info["task"]}')
            raise "Unsupported dataset task"
        DataClass = getattr(medmnist, info['python_class'])
        
        
        # validation_dataset = DataClass(split='val', as_rgb=True,download=True, size=224, transform=TwoCropTransform(trans['valid']))
        # test_dataset = DataClass(split='test', as_rgb=True, download=True, size=224, transform=TwoCropTransform(trans['test']))
        # training_dataset = DataClass(split='train', as_rgb=True, download=True,size=224,transform=TwoCropTransform(trans['train']))
        validation_dataset = DataClass(split='val', root=data_root, download=True, transform=TwoCropTransform(trans['valid']), as_rgb=True, size=224)
        test_dataset = DataClass(split='test', root=data_root, download=True, transform=TwoCropTransform(trans['test']), as_rgb=True, size=224)
        training_dataset = DataClass(split='train', root=data_root, download=True, transform=TwoCropTransform(trans['train']), as_rgb=True, size=224)

        id_to_class = training_dataset.info["label"]
        classes_num = {}
        training_dataset.labels = [i[0] for i in training_dataset.labels]
        test_dataset.labels = [i[0] for i in test_dataset.labels]
        validation_dataset.labels = [i[0] for i in validation_dataset.labels]
        for i in training_dataset.labels:
            i = str(i)
            class_name = id_to_class[i]
            if class_name not in classes_num:
                classes_num[class_name] = 1
            else:
                classes_num[class_name] += 1

        training_dataset.cls_num_list = list(classes_num.values())
        test_dataset.cls_num_list = list(classes_num.values())
        validation_dataset.cls_num_list = list(classes_num.values())
        validation_dataset.task = info["task"]
        test_dataset.task = info["task"]
        training_dataset.task = info["task"]
        test_dataset.class_names = list(training_dataset.info["label"].values())
        test_dataset.id_to_name = {int(k):v for k,v in training_dataset.info["label"].items()}

        return training_dataset, test_dataset, validation_dataset

    elif args.dataset == "Ulcerative":

        train_root_dir = f"Ulcerative colitis Endoscopy data/train_set"
        val_root_dir = f"Ulcerative colitis Endoscopy data/val_set"

        if args.other_test_dataset:
            test_root_dir = f"Oxford_dataset_UC"

        else:
            test_root_dir = f"Ulcerative colitis Endoscopy data/test_set"

        print(f"当前使用的测试集：{test_root_dir}")
        train_root_dir = os.path.join(data_root, train_root_dir)
        val_root_dir = os.path.join(data_root, val_root_dir)
        test_root_dir = os.path.join(data_root, test_root_dir)


    else:
        raise
    train_root_dir = os.path.join(data_root,train_root_dir)
    val_root_dir = os.path.join(data_root,val_root_dir)
    test_root_dir = os.path.join(data_root,test_root_dir)
    # Generators
    training_dataset = mydatasets.ImageData(train_root_dir, transform=TwoCropTransform(trans['train']))
    validation_dataset = mydatasets.ImageData(val_root_dir, transform=TwoCropTransform(trans['valid']))
    test_dataset = mydatasets.ImageData(test_root_dir, transform=TwoCropTransform(trans['test']))
    training_dataset.task = 'multi-class'
    validation_dataset.task = 'multi-class'
    test_dataset.task = 'multi-class'

    # 降采样后重新统计新的类数量
    classes_num = {}
    id_to_class = {v: k for k, v in training_dataset.class_to_idx.items()}
    for i in training_dataset.targets:
        class_name = id_to_class[i]
        if class_name not in classes_num:
            classes_num[class_name] = 1
        else:
            classes_num[class_name] += 1

    training_dataset.cls_num_list = list(classes_num.values())
    test_dataset.cls_num_list = list(classes_num.values())
    validation_dataset.cls_num_list = list(classes_num.values())
    test_dataset.class_names = classes_num.keys()
    test_dataset.id_to_name = id_to_class
    return training_dataset,test_dataset,validation_dataset

def shot_acc(train_class_count, test_class_count, class_correct, many_shot_thr=100, low_shot_thr=20):
    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] >= many_shot_thr:
            many_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        elif train_class_count[i] <= low_shot_thr:
            low_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))
        else:
            median_shot.append((class_correct[i].detach().cpu().numpy() / test_class_count[i]))

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)
