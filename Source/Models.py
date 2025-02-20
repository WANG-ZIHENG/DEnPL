import torch
import torchvision
import time
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self,args,n_classes, pretrained=False):
        super().__init__()
        device = args.device
        if args.model == "DenseNet121":
            model = torchvision.models.densenet121(weights=pretrained).to(device)
        elif args.model == "resnet18":
            model = torchvision.models.resnet18(weights=pretrained).to(device)
        elif args.model == "resnet34":
            model = torchvision.models.resnet34(weights=pretrained).to(device)
        elif args.model == "resnet50":
            model = torchvision.models.resnet50(weights=pretrained).to(device)
        elif args.model == "resnext50_32x4d":
            model = torchvision.models.resnext50_32x4d(weights=pretrained).to(device)
        elif args.model == 'resnet32':
            from models.resnet import resnet32
            assert pretrained == False
            model = resnet32(pretrained=pretrained,

                             phase_train=False,
                             norm_out=False,
                             add_rsg=False,
                             head_lists=[0,1],
                             add_arc_margin_loss=False,
                             add_add_margin_loss=False,
                             add_sphere_loss=False,
                             epoch_thresh=100,
                             )
            model = model.to(device)
        elif args.model == 'resnet10':
            from models.resnet import resnet10
            assert pretrained == False
            model = resnet10(pretrained=pretrained,

                             phase_train=False,
                             norm_out=False,
                             add_rsg=False,
                             head_lists=[0,1],
                             add_arc_margin_loss=False,
                             add_add_margin_loss=False,
                             add_sphere_loss=False,
                             epoch_thresh=100,
                             )
            model = model.to(device)

        self.features = nn.ModuleList(model.children())[:-1]
        self.features = nn.Sequential(*self.features)
        if args.model == "DenseNet121":
            n_inputs = model.classifier.in_features
        else:
            n_inputs = model.fc.in_features
        self.normal_classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes),
            )
        self.ce_classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes),
        )
        self.args = args
    def forward(self, input_imgs,cal_center=False):
        features = self.features(input_imgs)
        fea = F.relu(features, inplace=True)
        fea = F.adaptive_avg_pool2d(fea, (1, 1))
        fea = torch.flatten(fea, 1)
        normal_fea = F.normalize(fea, dim=1)
        ce_output = self.ce_classifier(fea)
        normal_output= self.normal_classifier(normal_fea)
        if cal_center:
            return normal_fea, normal_output, ce_output
        else:

            bsz = input_imgs.shape[0]//2
            f1, f2 = torch.split(normal_fea, [bsz, bsz], dim=0)
            nor_out1, nor_out2 = torch.split(normal_output, [bsz, bsz], dim=0)
            ce_out1, ce_out2 = torch.split(ce_output, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            normal_output = (nor_out1 + nor_out1) / 2
            if self.args.CE_sampling == "single":
                ce_output = ce_out1
            else:
                ce_output = torch.cat([ce_out1.unsqueeze(1), ce_out2.unsqueeze(1)], dim=1)
                # ce_output = (ce_out1 + ce_out2) / 2
            if self.args.CCL_sampling == "single":
                fea_pair_output = f1
            else:

                fea_pair_output = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            return features,normal_output,ce_output,fea_pair_output
