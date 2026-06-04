import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import os
import torch.nn.functional as F

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, _train_X, _train_Y, _train_U, data_loader, _nclass, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, generalized=True):
        self.train_X =  _train_X
        self.train_Y = _train_Y
        self.train_U = _train_U
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label 
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label 
        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        self.att = data_loader.attribute.cuda()
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = _train_X.size(1)
        self.model = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass)
        self.model.apply(util.weights_init)
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()
        
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size)
        self.seenorunseen = torch.LongTensor(_batch_size)
        
        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.model.cuda()
        self.criterion.cuda()
        self.input = self.input.cuda()
        self.label = self.label.cuda()
        self.seenorunseen = self.seenorunseen.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        self.log_softmax_func = nn.LogSoftmax(dim=1)

        if generalized:
            # self.acc_seen, self.acc_unseen, self.H = self.fit()
            self.fit()
            #print('Final: acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (self.acc_seen, self.acc_unseen, self.H))
        else:
            self.acc = self.fit_zsl() 
            #print('acc=%.4f' % (self.acc))

    
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8 
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label, batch_U = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                output, _ = self.model(inputv)
                # logits = torch.einsum('ba,ca->bc', output, self.att[self.unseenclasses, :])
                # Prob = self.log_softmax_func(logits)

                # loss = -torch.einsum('bc,b->b', Prob, labelv.cuda())
                # loss = torch.mean(loss)
                loss = self.criterion(output, labelv.cuda())
                # 原代码报错
                # mean_loss += loss.data[0]
                mean_loss += loss
                loss.backward()
                self.optimizer.step()
                # print('Training classifier loss= ', loss.item())
            # acc = self.val(self.test_seen_feature, self.map_label(self.test_seen_label, self.seenclasses), self.seenclasses)
            # print('epoch=%.4f, acc=%.4f' % (epoch, acc))
            # if acc > best_acc:
            #     best_acc = acc
        # print(best_acc)
        # return best_acc

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain, self.batch_size):      
                self.model.zero_grad()
                batch_input, batch_label, batch_U = self.next_batch(self.batch_size)
                self.input.copy_(batch_input)
                self.label.copy_(batch_label)
                self.seenorunseen.copy_(batch_U)
                   
                inputv = Variable(self.input)
                labelv = Variable(self.label)
                seenorunseenv = Variable(self.seenorunseen)
                output, output_ = self.model(inputv)
                loss = self.criterion(output, labelv.cuda())
                Prob_all = F.softmax(output, dim=-1)
                Prob_unseen = Prob_all[:, self.unseenclasses]
                Prob_seen = Prob_all[:, self.seenclasses]
                assert Prob_unseen.size(1) == len(self.unseenclasses)
                assert Prob_seen.size(1) == len(self.seenclasses)
                mass_unseen = torch.sum(Prob_unseen, dim=1)
                mass_seen = torch.sum(Prob_seen, dim=1)
                loss_unseen_cal = -torch.log(torch.mean(mass_unseen))
                loss_seen_cal = -torch.log(torch.mean(mass_seen))
                loss = loss + loss_unseen_cal + 4 * loss_seen_cal
                loss_ = self.criterion(output_, seenorunseenv)
                loss += 0.5 * loss_
                loss.backward()
                self.optimizer.step()
                # print('Training classifier loss= ', loss.data[0])
            acc_seen = 0
            acc_unseen = 0
            # acc_seen = self.val_gzsl(self.test_seen_feature, self.test_seen_label, self.seenclasses)
            # acc_unseen = self.val_gzsl(self.test_unseen_feature, self.test_unseen_label, self.unseenclasses)
            # H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
            # print('epoch=%.4f, acc_seen=%.4f, acc_unseen=%.4f, h=%.4f' % (epoch, acc_seen, acc_unseen, H))
            # if H > best_H:
            #     best_seen = acc_seen
            #     best_unseen = acc_unseen
            #     best_H = H
        # return best_seen, best_unseen, best_H
                     
    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.train_U = self.train_U[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
                U_rest_part = self.train_U[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            self.train_U = self.train_U[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            U_new_part = self.train_U[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0), torch.cat((U_rest_part, U_new_part), 0)
            else:
                return X_new_part, Y_new_part, U_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end], self.train_U[start:end]


    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            output, _ = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)

            # print('predict label: ', predicted_label[start:end])
            # print('ground truth: ', test_label[start:end])
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        class_acc = 0
        for i in target_classes:
            idx = (test_label == i)
            class_acc = torch.sum(test_label[idx]==predicted_label[idx]) / torch.sum(idx)
            print('accuracy of class %d is %.4f' % (i, class_acc))
            acc_per_class += class_acc
        acc_per_class /= target_classes.size(0)
        return acc_per_class


    # test_label is integer 
    def val(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_prob = torch.FloatTensor(test_X.size()[0], 10)
        # print(test_label)
        # print(util.map_label(test_label, target_classes))
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            output, _ = self.model(Variable(test_X[start:end].cuda(), volatile=True))
            # logits = torch.einsum('ba,ca->bc', output, self.att[self.unseenclasses, :])
            # predicted_prob[start:end] = F.softmax(output, dim=-1)
            _, predicted_label[start:end] = torch.max(output.data, dim=1)
            # print('predict label: ', predicted_label[start:end])
            # print('ground truth: ', test_label[start:end])
            start = end


        acc, acc_per_class = self.compute_per_class_acc(test_label, predicted_label, target_classes.size(0))
        # acc = torch.eq(predicted_label, util.map_label(test_label, target_classes)).sum().item()
        # acc = acc / len(self.test_unseen_feature)
        return acc, acc_per_class
        # return predicted_prob

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        print(acc_per_class)
        return acc_per_class.mean(), acc_per_class

    def map_label(self, label, classes):
        mapped_label = torch.LongTensor(label.size())
        for i in range(classes.size(0)):
            mapped_label[label == classes[i]] = i

        return mapped_label

class LINEAR_LOGSOFTMAX(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX, self).__init__()
        # self.fc = nn.Linear(input_dim, natt)
        self.fc = nn.Linear(input_dim, nclass)
        self.fc_ = nn.Linear(input_dim, 2)
        # self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        # o = self.logic(self.fc(x))
        o = self.fc(x)
        o_ = self.fc_(x)
        return o, o_
