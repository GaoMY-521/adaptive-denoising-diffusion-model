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
    def __init__(self, _train_X_zsl, _train_Y_zsl, _train_U_zsl, _train_X_gzsl, _train_Y_gzsl, _train_U_gzsl,
                 data_loader, _nclass_zsl, _nclass_gzsl, _lr=0.001, _beta1=0.5, _nepoch=20,
                 _batch_size=100):
        self.train_X_zsl = _train_X_zsl
        self.train_Y_zsl = _train_Y_zsl
        self.train_U_zsl = _train_U_zsl
        self.train_X_gzsl = _train_X_gzsl
        self.train_Y_gzsl = _train_Y_gzsl
        self.train_U_gzsl = _train_U_gzsl
        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label
        self.seenclasses = data_loader.seenclasses.cuda()
        self.unseenclasses = data_loader.unseenclasses.cuda()
        self.att = data_loader.attribute.cuda()
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass_zsl = _nclass_zsl
        self.nclass_gzsl = _nclass_gzsl
        self.input_dim = _train_X_zsl.size(1)
        self.model_zsl = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass_zsl)
        self.model_gzsl = LINEAR_LOGSOFTMAX(self.input_dim, self.nclass_gzsl)
        self.model_zsl.apply(util.weights_init)
        self.model_gzsl.apply(util.weights_init)
        # self.criterion = nn.NLLLoss()
        self.criterion = nn.CrossEntropyLoss()

        self.input_zsl = torch.FloatTensor(_batch_size, self.input_dim)
        self.label_zsl = torch.LongTensor(_batch_size)
        self.seenorunseen_zsl = torch.LongTensor(_batch_size)
        self.input_gzsl = torch.FloatTensor(_batch_size, self.input_dim)
        self.label_gzsl = torch.LongTensor(_batch_size)
        self.seenorunseen_gzsl = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        # setup optimizer
        self.optimizer_zsl = optim.Adam(self.model_zsl.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.optimizer_gzsl = optim.Adam(self.model_gzsl.parameters(), lr=_lr, betas=(_beta1, 0.999))

        self.model_zsl.cuda()
        self.model_gzsl.cuda()
        self.criterion.cuda()
        self.input_zsl = self.input_zsl.cuda()
        self.label_zsl = self.label_zsl.cuda()
        self.seenorunseen_zsl = self.seenorunseen_zsl.cuda()
        self.input_gzsl = self.input_gzsl.cuda()
        self.label_gzsl = self.label_gzsl.cuda()
        self.seenorunseen_gzsl = self.seenorunseen_gzsl.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain_zsl = self.train_X_zsl.size()[0]
        self.ntrain_gzsl = self.train_X_gzsl.size()[0]

        self.log_softmax_func = nn.LogSoftmax(dim=1)

        self.fit()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.fit_zsl()

    def fit_zsl(self):
        mean_loss = 0
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain_zsl, self.batch_size):
                self.model_zsl.zero_grad()
                batch_input, batch_label, batch_U = self.next_batch(self.batch_size, self.ntrain_zsl, self.train_X_zsl,
                                                                    self.train_Y_zsl, self.train_U_zsl)
                self.input_zsl.copy_(batch_input)
                self.label_zsl.copy_(batch_label)

                inputv_zsl = Variable(self.input_zsl)
                labelv_zsl = Variable(self.label_zsl)
                output_zsl, _ = self.model_zsl(inputv_zsl)

                loss = self.criterion(output_zsl, labelv_zsl.cuda())
                # 原代码报错
                # mean_loss += loss.data[0]
                mean_loss += loss
                loss.backward()
                self.optimizer_zsl.step()

    def fit(self):
        for epoch in range(self.nepoch):
            for i in range(0, self.ntrain_gzsl, self.batch_size):
                self.model_gzsl.zero_grad()
                batch_input, batch_label, batch_U = self.next_batch(self.batch_size, self.ntrain_gzsl,
                                                                    self.train_X_gzsl, self.train_Y_gzsl,
                                                                    self.train_U_gzsl)
                self.input_gzsl.copy_(batch_input)
                self.label_gzsl.copy_(batch_label)
                self.seenorunseen_gzsl.copy_(batch_U)

                inputv_gzsl = Variable(self.input_gzsl)
                labelv_gzsl = Variable(self.label_gzsl)
                seenorunseenv_gzsl = Variable(self.seenorunseen_gzsl)
                output_gzsl, output_gzsl_ = self.model_gzsl(inputv_gzsl)
                loss_gzsl = self.criterion(output_gzsl, labelv_gzsl.cuda())
                Prob_all = F.softmax(output_gzsl, dim=-1)
                Prob_unseen = Prob_all[:, self.unseenclasses]
                Prob_seen = Prob_all[:, self.seenclasses]
                assert Prob_unseen.size(1) == len(self.unseenclasses)
                assert Prob_seen.size(1) == len(self.seenclasses)
                mass_unseen = torch.sum(Prob_unseen, dim=1)
                mass_seen = torch.sum(Prob_seen, dim=1)
                loss_unseen_cal = -torch.log(torch.mean(mass_unseen))
                loss_seen_cal = -torch.log(torch.mean(mass_seen))
                loss_gzsl = loss_gzsl + 0.25 * loss_unseen_cal + 2 * loss_seen_cal
                # loss_ = self.criterion(output_gzsl_, seenorunseenv_gzsl)
                # loss_gzsl += 0.5 * (loss_)
                loss_gzsl.backward()
                self.optimizer_gzsl.step()

    def next_batch(self, batch_size, ntrain, train_X, train_Y, train_U):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(ntrain)
            train_X = train_X[perm]
            train_Y = train_Y[perm]
            train_U = train_U[perm]
        # the last batch
        if start + batch_size > ntrain:
            self.epochs_completed += 1
            rest_num_examples = ntrain - start
            if rest_num_examples > 0:
                X_rest_part = train_X[start:ntrain]
                Y_rest_part = train_Y[start:ntrain]
                U_rest_part = train_U[start:ntrain]
            # shuffle the data
            perm = torch.randperm(ntrain)
            train_X = train_X[perm]
            train_Y = train_Y[perm]
            train_U = train_U[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = train_X[start:end]
            Y_new_part = train_Y[start:end]
            U_new_part = train_U[start:end]
            # print(start, end)
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0), torch.cat(
                    (U_rest_part, U_new_part), 0)
            else:
                return X_new_part, Y_new_part, U_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # print(start, end)
            # from index start to index end-1
            return train_X[start:end], train_Y[start:end], train_U[start:end]

    def val(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_prob = torch.FloatTensor(test_X.size()[0], 10)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            output, _ = self.model_zsl(Variable(test_X[start:end].cuda(), volatile=True))
            # logits = torch.einsum('ba,ca->bc', output, self.att[self.unseenclasses, :])

            # _, predicted_label[start:end] = torch.max(output.data, dim=1)
            predicted_prob[start:end] = F.softmax(output, dim=-1)
            start = end

        # acc, acc_per_class = self.compute_per_class_acc(test_label, predicted_label, target_classes.size(0))
        # acc = torch.eq(predicted_label, util.map_label(test_label, target_classes)).sum().item()
        # acc = acc / len(self.test_unseen_feature)
        # return acc, acc_per_class
        return predicted_prob

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
        print(acc_per_class)
        return acc_per_class.mean(), acc_per_class

    def val_gzsl_seen(self, test_X, test_label, target_classes):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        predicted_prob = torch.FloatTensor(test_X.size()[0], 50)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            output, _ = self.model_gzsl(Variable(test_X[start:end].cuda(), volatile=True))
            _, predicted_label[start:end] = torch.max(output.data, 1)
            # predicted_prob[start:end] = F.softmax(output, dim=-1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc
        # return predicted_prob

    def val_gzsl_unseen(self, test_X, test_label, target_classes, seen_classes):
        device = 'cuda:0'
        start = 0
        ntest = test_X.size()[0]
        # predicted_label = torch.LongTensor(test_label.size()).to(device)
        new_predicted_prob = torch.FloatTensor(test_X.size()[0], 50)
        new_predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            test_X_batch = test_X[start:end].cuda()
            output, output_ = self.model_gzsl(Variable(test_X_batch, volatile=True))
            _, predicted_label = torch.max(output.data, 1)
            # predicted_prob = F.softmax(output, dim=-1)
            softmax_ = F.softmax(output_, dim=-1)
            softmax_1 = softmax_[:, 1]
            unseen_idx = softmax_1 != 1
            unseen_idx = unseen_idx.to(device)
            unseen_sample = test_X_batch[unseen_idx]
            if unseen_sample.size()[0] != 0:
                output_zsl, _ = self.model_zsl(Variable(unseen_sample, volatile=True))
                # zeros = torch.zeros(size=[unseen_idx.size()[0], 40]).to(device)
                # predicted_prob[start:end][unseen_idx] = torch.FloatTensor(unseen_idx.size()[0], 50).to(device)
                _, predicted_zsl = torch.max(output_zsl.data, 1)
                predicted_zsl = self.unseenclasses[predicted_zsl]
                unseen_idx = unseen_idx.data.cpu()
                new_predicted_label[start:end][unseen_idx] = predicted_zsl.data.cpu()
                # 111predicted_zsl_prob = F.softmax(output_zsl, dim=-1)
                # predicted_prob[start:end][unseen_idx][:, seen_classes] = zeros
                # predicted_prob.data[start:end][unseen_idx][:, target_classes].copy_(predicted_zsl_prob.data)
                # new_predicted_prob[start:end].copy_(predicted_prob.data.cpu())
                # 111unseen_idx = unseen_idx.data.cpu()
                # 111target_classes = target_classes.data.cpu()
                # print(new_predicted_prob[start:end][unseen_idx][:, target_classes])

                # 111new_predicted_prob[start:end][unseen_idx, target_classes] = predicted_zsl_prob.data.cpu()
                # print(predicted_prob[start:end][unseen_idx][0])
                # print(predicted_prob[start:end][unseen_idx][:, target_classes])
                # print(predicted_zsl_prob[0])

            # print('predict label: ', predicted_label[start:end])
            # print('ground truth: ', test_label[start:end])
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, new_predicted_label, target_classes)
        return acc
        # return new_predicted_prob

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        class_acc = 0
        for i in target_classes:
            idx = (test_label == i)
            class_acc = torch.sum(test_label[idx] == predicted_label[idx]) / torch.sum(idx)
            print('accuracy of class %d is %.4f' % (i, class_acc))
            acc_per_class += class_acc
        acc_per_class /= target_classes.size(0)
        return acc_per_class


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