import os
from collections import defaultdict
import numpy as np
import torch

from utils import shot_acc
from misc import plot_confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as mtc
from sklearn.metrics import confusion_matrix
import wandb
from sklearn.metrics import cohen_kappa_score, roc_auc_score
from misc import print_metrics, training_curve
# from openTSNE import TSNE
from sklearn.manifold import TSNE
import tsneutil
from evaluator import getAUC,getACC

def test_net(epoch, model, test_generator,cls_num_list_train, device, criterion, args, centers):
    # def test_net(model,test_generator,device,criterion,args):
    if 'cifar' in  args.dataset:
        many_shot_thr = 100
        low_shot_thr = 20
    else:
        cls_num_list = test_generator.dataset.cls_num_list
        break_point = int(len(cls_num_list) / 4)
        many_shot_thr = cls_num_list[break_point]
        low_shot_thr = cls_num_list[-break_point]
    cls_num_list_test = test_generator.dataset.cls_num_list
    cls_num = len(test_generator.dataset.cls_num_list)
    num_steps = 0
    test_loss = 0
    correct = 0
    correct_class = [0] * cls_num
    test_metrics = defaultdict(float)
    all_labels_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_d = torch.tensor([], dtype=torch.long).to(device)
    all_predictions_probabilities_d = torch.tensor([], dtype=torch.float).to(device)
    all_output_d = []

    # loss_weight_alpha = 1 - (epoch/args.max_epochs)**2
    # args.CE_loss_weight, args.CCL_loss_weight = loss_weight_alpha, (1-loss_weight_alpha)

    for image, labels in test_generator:
        images = torch.cat([image[0], image[1]], dim=0)
        # Transfer to GPU:
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

        test_loss += loss.item() * bsz

        if (
                args.CE_loss_use == True or args.WCE_loss_use == True) and args.LDAM_loss_use == False and args.Logit_loss_use == False:
            predicted_probability, predicted = torch.max(ce_output, dim=1)
            all_output_d.append(ce_output.cpu().detach().numpy())
        elif (args.CE_loss_use == False or args.WCE_loss_use == False) and (
                args.LDAM_loss_use == True or args.Logit_loss_use == True):

            predicted_probability, predicted = torch.max(normal_output, dim=1)
            all_output_d.append(normal_output.cpu().detach().numpy())
        else:
            output = (ce_output + normal_output) / 2
            predicted_probability, predicted = torch.max(output, dim=1)
            all_output_d.append(output.cpu().detach().numpy())
        correct += (predicted == labels).sum()
        all_labels_d = torch.cat((all_labels_d, labels), 0)
        all_predictions_d = torch.cat((all_predictions_d, predicted), 0)
        all_predictions_probabilities_d = torch.cat((all_predictions_probabilities_d, predicted_probability), 0)
        for l in range(0, cls_num):
            correct_class[l] += (predicted[labels == l] == labels[labels == l]).sum()
        
    all_output_d = np.concatenate(all_output_d)
    y_true = all_labels_d.cpu()
    y_predicted = all_predictions_d.cpu()
    all_output_softmax = torch.softmax(torch.tensor(all_output_d), dim=1).numpy() 
    
    # 这里假设y_true是类别标签的一维数组
    # 如果你的任务是二分类，确保y_true和all_output_softmax的形状是兼容的
    num_classes = len(np.unique(y_true))
    if num_classes == 2:
        y_true_one_hot = np.eye(num_classes)[y_true]  # Convert to one-hot encoding for binary classification
    else:
        y_true_one_hot = y_true

   

    
    # all_output_d = np.concatenate(all_output_d)
    # y_true = all_labels_d.cpu()
    # y_predicted = all_predictions_d.cpu()  # to('cpu')
    valset_predicted_probabilites = all_predictions_probabilities_d.cpu()  # to('cpu')
    # ����QWKָ��
    qwk_score = cohen_kappa_score(y_true, y_predicted, weights='quadratic')
    micro_precision = mtc.precision_score(y_true, y_predicted, average="micro")
    micro_recall = mtc.recall_score(y_true, y_predicted, average="micro")
    micro_f1 = mtc.f1_score(y_true, y_predicted, average="micro")

    macro_precision = mtc.precision_score(y_true, y_predicted, average="macro")
    macro_recall = mtc.recall_score(y_true, y_predicted, average="macro")
    macro_f1 = mtc.f1_score(y_true, y_predicted, average="macro")
    mcc = mtc.matthews_corrcoef(y_true, y_predicted)
    y_true = y_true.detach().numpy()
    acc = getACC(y_true, all_output_softmax, task=test_generator.dataset.task)
    if test_generator.dataset.task == 'binary-class':
        all_output_softmax = np.max(all_output_softmax, axis=1)
    auc = getAUC(y_true, all_output_softmax, task=test_generator.dataset.task)
    many_shot_overall, median_shot_overall, low_shot_overall = shot_acc(
        cls_num_list_train, cls_num_list_test, correct_class,
        many_shot_thr=many_shot_thr, low_shot_thr=low_shot_thr)

    result = 'test The many shot accuracy: %.2f. The median shot accuracy: %.2f. The low shot accuracy: %.2f.' % (
         many_shot_overall * 100, median_shot_overall * 100, low_shot_overall * 100)
    print(result)

    test_metrics['acc'] = acc
    test_metrics['auc'] = auc
    test_metrics['many_shot_overall'] = many_shot_overall * 100
    test_metrics['median_shot_overall'] = median_shot_overall * 100
    test_metrics['low_shot_overall'] = low_shot_overall * 100
    test_metrics['micro_precision'] = micro_precision
    test_metrics['micro_recall'] = micro_recall
    test_metrics['micro_f1'] = micro_f1
    test_metrics['macro_precision'] = macro_precision
    test_metrics['macro_recall'] = macro_recall
    test_metrics['macro_f1'] = macro_f1
    test_metrics['mcc'] = mcc
    test_metrics['qwk'] = qwk_score
    
    
    x = all_output_d
    y = all_labels_d.cpu().numpy()
    id_to_name = test_generator.dataset.id_to_name
    id_to_name = {int(k): v for k, v in id_to_name.items()}
    # y = [id_to_name[str(i)] for i in y]
    n_components = 2
    tsne = TSNE(
        n_components=n_components,
        # init='pca',
        perplexity=50,
        n_iter=500,
        metric="euclidean",
        # callbacks=ErrorLogger(),
        n_jobs=8,
        random_state=42,
    )
    print("fit tsne")
    embedding = tsne.fit_transform(x)
    tsneutil.plot(embedding, y, colors=tsneutil.MOUSE_10X_COLORS, save_path="tsne.png",label_order = list(id_to_name.keys()))
    wandb.log({f"tsne": wandb.Image("tsne.png", caption="")},
              commit=False)
    cm = confusion_matrix(y_true, y_predicted)  # confusion matrix

    print('Accuracy of the network on the %d test images: %f %%' % (num_steps, (100.0 * correct / num_steps)))

    print(cm)

    print("taking class names to plot CM")

    class_names = test_generator.dataset.class_names # test_datasets.classes  # taking class names for plotting confusion matrix

    print("Generating confution matrix")

    plot_confusion_matrix(cm, classes=class_names, title='my confusion matrix')
    print(test_metrics)
    print_metrics(test_metrics, args.max_epochs)

    ##################################################################
    # classification report
    #################################################################
    classification_report_str = classification_report(y_true, y_predicted, target_names=class_names)
    print(classification_report_str)
    os.makedirs('classification_report', exist_ok=True)
    with open('classification_report/classification_report.txt', 'w') as f:
        f.write(classification_report_str)
    # artifact = wandb.Artifact(name="classification_report", type="dataset")
    # artifact.add_dir(local_path="classification_report")

    return (test_loss / num_steps), test_metrics, num_steps
