import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from denoising_diffusion_pytorch_main.denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
import utils
from ema import EMA
import classifier2
import os
import argparse
import torch.optim as optim
from torchvision import datasets, transforms
from einops import rearrange, reduce, repeat
from torch.autograd import Variable
import clip

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = (test_label == i)
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
    print(acc_per_class)
    return acc_per_class.mean() # , acc_per_class

def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    acc_per_class = 0
    for i in target_classes:
        idx = (test_label == i)
        class_acc = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
        print('accuracy of class %d is %.4f' % (i, class_acc))
        acc_per_class += class_acc
    acc_per_class /= target_classes.size(0)
    return acc_per_class


def sort_feat_by_label(feature, labels, classes):
    output = []
    for i in range(len(classes)):
        iclass = classes[i]
        idx = labels.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = feature[idx][0:8]
        output.append(iclass_feature)
    output = torch.cat(output, dim=0)
    return output

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AWA2', help='FLO')
parser.add_argument('--dataroot', default='data', help='path to dataset')
parser.add_argument('--image_embedding', default='res101')
parser.add_argument('--class_embedding', default='att')
parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
parser.add_argument('--batch_size', default='32', help='batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
parser.add_argument('--attSize', type=int, default=85, help='size of semantic features')
parser.add_argument('--nclass_all', type=int, default=50, help='number of all classes')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate to train diffusion model')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
opt = parser.parse_args()

data = utils.DATA_LOADER(opt)
dataset = data.train_feature
unseen_class = data.unseenclasses.size()[0]
seen_class = data.seenclasses.size()[0]
num = opt.syn_num
syn_feature = torch.FloatTensor(unseen_class * num, 2048)
syn_class_label = torch.LongTensor(num)
syn_class_att = torch.FloatTensor(num, opt.attSize)
syn_label_zsl = torch.LongTensor(unseen_class * num)
syn_label_gzsl = torch.LongTensor(unseen_class * num)
syn_label_seen = torch.LongTensor(seen_class * num)
syn_feature_seen = torch.FloatTensor(seen_class * num, opt.resSize)
syn_class_label_seen = torch.LongTensor(num)
syn_class_att_seen = torch.FloatTensor(num, opt.attSize)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

model = Unet1D(dim = 256, dim_mults = (2, 4, 8), num_classes = 50, att_size = 85, channels = 32).to(device)
diffusion = GaussianDiffusion1D(model, seq_length = 64, timesteps = 100, objective = 'pred_x0').to(device)
optimizer = optim.Adam(diffusion.parameters(), lr=opt.lr, betas = (0.9, 0.99))

train_X = data.train_feature.to(device)
train_Y = data.train_label.to(device)
test_seen_feature = data.test_seen_feature.to(device)
test_unseen_feature = data.test_unseen_feature.to(device)
all_attribute = data.attribute
all_class_feature = sort_feat_by_label(train_X, train_Y, data.allclasses)
map_attribute = diffusion.model.classes_mlp(all_attribute.cuda())


batch_size = opt.batch_size
criterion = nn.CrossEntropyLoss()

for epoch in range(5001):
    train_loss = 0
    diffusion.train()
    for i in range(0, train_X.size()[0], batch_size):
        permutation = torch.randperm(train_X.size()[0])
        indices = permutation[i:i + batch_size]
        batch_x = train_X[indices].to(device)  # bs x 2048
        iclass = train_Y[indices]
        batch_att = data.attribute[iclass].to(device)
        optimizer.zero_grad()
        xing_diff, loss = diffusion(batch_x, batch_att)
        loss.backward()
        optimizer.step()
        train_loss += loss
        print('current out epoch=%d, total loss=%.3f' % (iter_d, train_loss))

        if epoch % 10 == 0:
            for j in range(unseen_class):
                jclass = data.unseenclasses[j]
                j_att = data.attribute[jclass]
                syn_class_att = syn_class_att.copy_(j_att).cuda()
                syn_class_label = syn_class_label.fill_(jclass).cuda()
                sampled_feature = diffusion_2.sample(att=syn_class_att, cond_scale=1.)
                sampled_feature = (sampled_feature).view(num, 2048)
                syn_feature.narrow(0, j * num, num).copy_(sampled_feature.data.cpu())
                syn_label_zsl.narrow(0, j * num, num).fill_(j)
                syn_label_gzsl.narrow(0, j * num, num).fill_(jclass)

            train_X = torch.cat((data.train_feature, syn_feature), 0)
            train_Y = torch.cat((data.train_label, syn_label_gzsl), 0)
            one = torch.ones(size=data.train_label.size())
            zero = torch.zeros(size=syn_label_gzsl.size())
            train_U = torch.cat((one, zero), dim=0)

            classifier = classifier_cascade.CLASSIFIER(train_X, train_Y, train_U, data, nclass, opt.classifier_lr, 0.5, 25, opt.syn_num, True)
            seen_acc = classifier.val_gzsl_seen(data.test_seen_feature, data.test_seen_label, data.seenclasses)
            unseen_acc = classifier.val_gzsl_unseen(data.test_unseen_feature.float(), data.test_unseen_label, data.unseenclasses)
            H = 2 * seen_acc * unseen_acc / (seen_acc + unseen_acc)
            retrain_cls2_zsl = classifier2.CLASSIFIER(syn_feature, syn_label_zsl, train_U, data, unseen_class, opt.classifier_lr, 0.5, 25, opt.syn_num, False)
            acc, acc_perclass_unseen = retrain_cls2_zsl.val(data.test_unseen_feature.float(),
                                                    map_label(data.test_unseen_label, data.unseenclasses),
                                                    data.unseenclasses)

            message = "unseen_class_accuracy: %.4f\n" % (acc)
            print(message)
            message_gzsl = 'seen acc: %.4f, unseen acc: %.4f, H: %.4f\n' % (seen_acc, unseen_acc, H)
            print(message_gzsl)



