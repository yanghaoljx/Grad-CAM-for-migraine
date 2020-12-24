import torch
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score,roc_auc_score, roc_curve,classification_report,auc 
import torch.backends.cudnn as cudnn
from mri_dataset import MriDataset
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix
from scipy import interp
from itertools import cycle
import seaborn as sns
from models import *

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = VGG('VGG19')
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    checkpoint = torch.load('./checkpoint/ckpt4.pth')
    net.load_state_dict(checkpoint['net'])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    testset = MriDataset(mode='val',transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=1)
    net.eval()
    outputs = np.zeros((1,3))
    print(outputs)
    targets_all = np.zeros((1,1))
    predicts_all = np.zeros((1,1))
    for batch_idx, (inputs, targets) in enumerate(testloader):
        targets_all = np.append(targets_all,targets.numpy().reshape((-1,1)))
        inputs, targets = inputs.to(device), targets.to(device)
        results = net(inputs)
        _,predicted = results.max(1)
        predicted = predicted.cpu()
        predicts_all = np.append(predicts_all,predicted.numpy().reshape((-1,1)))
        results = results.cpu()
        results = results.detach().numpy()
        results = results.reshape((-1,3))
        outputs = np.append(outputs,results,axis=0)
        # _, predicted = outputs.max(1)
    # outputs = np.array(outputs)
    final_outputs = outputs[1:]
    final_targets_all = targets_all[1:]
    final_predicts_all = predicts_all[1:].reshape((-1,1))
    final_targets_all = final_targets_all.reshape((-1,1))
    print(final_outputs.shape)
    print(final_targets_all.shape)
    print(final_predicts_all.shape)

    y_test_non_category = final_targets_all
    y_predict_non_category = final_predicts_all

    n_classes = 3 
    lw = 2
    enc = OneHotEncoder(categories='auto')
    enc.fit([[0],[1],[2]])
    y_test_oh = enc.transform(final_targets_all).toarray()
    # print(final_targets_all)
    fpr = dict()
    tpr = dict() 
    roc_auc = dict() 

    y_proba = final_outputs

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_oh[:, i], y_proba[:,i])
        roc_auc[i] = roc_auc_score(y_test_oh[:, i], y_proba[:,i])
        
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_oh.ravel(), y_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
            label='micro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["micro"]),
            color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
            label='macro-average ROC curve (area = {0:0.2f})'
                ''.format(roc_auc["macro"]),
            color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                label='ROC curve of class {0} (area = {1:0.2f})'
                ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig('final_roc3.jpg')

    class_names = ['HC','MWA','MWoA']

    conf_mat = confusion_matrix(y_test_non_category, y_predict_non_category)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(conf_mat, annot=True, fmt='d', annot_kws={'size':15}, xticklabels=class_names, yticklabels=class_names,cmap="YlGnBu")
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.setp(ax.get_yticklabels(), rotation=45)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

    plt.savefig('confusion-matrix3.jpg')