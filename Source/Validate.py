from collections import defaultdict
import numpy as np
import torch
import csv
import sklearn.metrics as mtc
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score, roc_auc_score
import torch.nn.functional as F

from utils import shot_acc
from evaluator import getAUC,getACC

def validate_net(epoch, model, validation_generator,cls_num_list_train, device, criterion, args, centers):
    if 'cifar' in  args.dataset:
        many_shot_thr = 100
        low_shot_thr = 20
    else:
        cls_num_list = validation_generator.dataset.cls_num_list
        break_point = int(len(cls_num_list) / 4)
        many_shot_thr = cls_num_list[break_point]
        low_shot_thr = cls_num_list[-break_point]

    cls_num_list_test = validation_generator.dataset.cls_num_list
    cls_num = len(validation_generator.dataset.cls_num_list)
    # def validate_net(model,validation_generator,device,criterion,args):
    num_steps = 0
    val_loss = 0
    val_ldam_loss = 0
    val_logit_loss = 0
    val_ce_loss = 0
    val_supcon_loss = 0
    val_ccl_loss = 0
    val_similarity_loss = 0
    val_uncertain_loss = 0
    correct_class = [0] * cls_num
    correct = 0
    val_metrics = defaultdict(float)
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
    all_softmax_output = []
    # loss_weight_alpha = 1 - (epoch/args.max_epochs)**2
    # args.CE_loss_weight, args.CCL_loss_weight = loss_weight_alpha, (1-loss_weight_alpha)

    for image, labels in validation_generator:
        # Transfer to GPU:


        images = torch.cat([image[0], image[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        features, normal_output, ce_output,fea_pair_mean = model(images)
        ce_output = ce_output.mean(1)

        loss, ldam_loss, logit_loss, ce_loss, supcon_loss, ccl_loss, wce_loss,similarity_loss,uncertain_loss = criterion(features, normal_output,
                                                                                          ce_output,
                                                                                          labels, fea_pair_mean,
                                                                                          centers.class_centers,epoch)

        # loss,ldam_loss,logit_loss,ce_loss,supcon_loss = criterion(features,normal_output,ce_output, labels)

        num_steps += bsz

        val_loss += loss.item() * bsz
        val_ldam_loss += ldam_loss.item() * bsz
        val_logit_loss += logit_loss.item() * bsz
        val_ce_loss += ce_loss.item() * bsz
        val_supcon_loss += supcon_loss.item() * bsz
        val_ccl_loss += ccl_loss.item() * bsz
        val_similarity_loss += similarity_loss.item() * bsz
        val_uncertain_loss += uncertain_loss.item() * bsz

        if (
                args.CE_loss_use == True or args.WCE_loss_use == True) and args.LDAM_loss_use == False and args.Logit_loss_use == False:
            predicted_probability, predicted = torch.max(ce_output, dim=1)
            softmax_output = F.softmax(ce_output, dim=1)
        elif (args.CE_loss_use == False or args.WCE_loss_use == False) and (
                args.LDAM_loss_use == True or args.Logit_loss_use == True):

            predicted_probability, predicted = torch.max(normal_output, dim=1)
            softmax_output = F.softmax(normal_output, dim=1)
        else:
            output = (ce_output + normal_output) / 2
            softmax_output = F.softmax(output, dim=1)
            predicted_probability, predicted = torch.max(output, dim=1)
        for l in range(0, cls_num):
            correct_class[l] += (predicted[labels == l] == labels[labels == l]).sum()

        correct += (predicted == labels).sum()
        all_labels_d = torch.cat((all_labels_d, labels), 0)
        all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
        # all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
        all_softmax_output.append(softmax_output.cpu().detach().numpy())
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()  # to('cpu')
    # valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
    all_softmax_output = np.concatenate(all_softmax_output)

    #############################
    # Standard metrics
    #############################
    qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
    micro_precision = mtc.precision_score(y_true, y_predicted, average="micro")
    micro_recall = mtc.recall_score(y_true, y_predicted, average="micro")
    micro_f1 = mtc.f1_score(y_true, y_predicted, average="micro")

    macro_precision = mtc.precision_score(y_true, y_predicted, average="macro")
    macro_recall = mtc.recall_score(y_true, y_predicted, average="macro")
    macro_f1 = mtc.f1_score(y_true, y_predicted, average="macro")

    mcc = mtc.matthews_corrcoef(y_true, y_predicted)

    y_true = y_true.detach().numpy()
    acc = getACC(y_true, all_softmax_output, task=validation_generator.dataset.task)
    if validation_generator.dataset.task == 'binary-class':
        all_softmax_output = np.max(all_softmax_output, axis=1)
    auc = getAUC(y_true, all_softmax_output, task=validation_generator.dataset.task)

    many_shot_overall, median_shot_overall, low_shot_overall = shot_acc(
        cls_num_list_train, cls_num_list_test, correct_class,
        many_shot_thr=many_shot_thr, low_shot_thr=low_shot_thr)

    result = 'val The many shot accuracy: %.2f. The median shot accuracy: %.2f. The low shot accuracy: %.2f.' % (
         many_shot_overall * 100, median_shot_overall * 100, low_shot_overall * 100)
    print(result)

    val_metrics['acc'] = acc
    val_metrics['many_shot_overall'] = many_shot_overall * 100
    val_metrics['median_shot_overall'] = median_shot_overall * 100
    val_metrics['low_shot_overall'] = low_shot_overall * 100
    val_metrics['auc'] = auc
    val_metrics['loss'] = val_loss / num_steps
    val_metrics['ldam_loss'] = val_ldam_loss / num_steps
    val_metrics['logit_loss'] = val_logit_loss / num_steps
    val_metrics['ce_loss'] = val_ce_loss / num_steps
    val_metrics['supcon_loss'] = val_supcon_loss / num_steps
    val_metrics['ccl_loss'] = val_ccl_loss / num_steps
    val_metrics['similarity_loss'] = val_similarity_loss / num_steps
    val_metrics['uncertain_loss'] = val_uncertain_loss / num_steps


    val_metrics['micro_precision'] = micro_precision
    val_metrics['micro_recall'] = micro_recall
    val_metrics['micro_f1'] = micro_f1
    val_metrics['macro_precision'] = macro_precision
    val_metrics['macro_recall'] = macro_recall
    val_metrics['macro_f1'] = macro_f1
    val_metrics['mcc'] = mcc
    val_metrics['qwk'] = qwk_score

    return (val_loss / num_steps), val_metrics, num_steps
