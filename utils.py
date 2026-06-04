# Loss Function for Diffusion Model
# Original Source: https://github.com/acids-ircam/diffusion_models

import torch
import numpy as np
import sys
import scipy.io as sio
from sklearn import preprocessing
import torch.nn as nn
from torch.autograd import Variable
import pickle

def make_beta_schedule(schedule='linear', n_timesteps=1000, start=1e-5, end=1e-2):
    if schedule == 'linear':
        betas = torch.linspace(start, end, n_timesteps)
    elif schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
    elif schedule == "sigmoid":
        betas = torch.linspace(-6, 6, n_timesteps)
        betas = torch.sigmoid(betas) * (end - start) + start
    return betas

def extract(input, t, x):
    shape = x.shape
    out = torch.gather(input, 0, t.to(input.device))
    reshape = [t.shape[0]] + [1] * (len(shape) - 1)
    return out.reshape(*reshape)

def q_posterior_mean_variance(x_0, x_t, t,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped):
    coef_1 = extract(posterior_mean_coef_1, t, x_0)
    coef_2 = extract(posterior_mean_coef_2, t, x_0)
    mean = coef_1 * x_0 + coef_2 * x_t
    var = extract(posterior_log_variance_clipped, t, x_0)
    return mean, var

def p_mean_variance(model, x, t):
    # Go through model
    out = model(x.cuda(), t.cuda())
    # Extract the mean and variance
    mean, log_var = torch.split(out, 2048, dim=-1)
    var = torch.exp(log_var)
    return mean, log_var

def p_sample(model, cls, x, label, t, alphas, betas, one_minus_alphas_bar_sqrt):
    t = torch.tensor([t])
    CE = nn.CrossEntropyLoss()
    # Factor to the model output
    eps_factor = ((1 - extract(alphas, t, x)) / extract(one_minus_alphas_bar_sqrt, t, x))
    # Model output
    eps_theta = model(x.cuda(), t.cuda())
    predict = cls(x.float().cuda())
    ce = CE(predict, label)
    ce.backward(retain_graph=True)
    grad = x.grad
    eps_theta = eps_theta.cpu().data
    # Final values
    mean = (1 / extract(alphas, t, x).sqrt()) * (x - (eps_factor * eps_theta))
    # Generate z
    z = torch.randn_like(x)
    # Fixed sigma
    sigma_t = extract(betas, t, x).sqrt()
    if t == 0:
        # print("time step over")
        z = 0
    guidance = grad * sigma_t
    sample = mean + 10 * guidance + sigma_t * z
    return sample, predict

def p_sample_distribute(model, x, t):
    mean, log_var = p_mean_variance(model, x, torch.tensor(t))
    noise = torch.randn_like(x).cuda()
    shape = [x.shape[0]] + [1] * (x.ndimension() - 1)
    nonzero_mask = (1 - (t == 0))
    sample = mean + torch.exp(0.5 * log_var) * noise
    return (sample)

def p_sample_loop(model, cls, label, shape, n_steps, alphas, betas, one_minus_alphas_bar_sqrt):
    cur_x = torch.randn(shape)
    # x_seq = [cur_x]
    for i in reversed(range(n_steps)):
        # print(cur_x)
        cur_x = Variable(cur_x, requires_grad=True)
        cur_x, predict = p_sample(model, cls, cur_x, label, i, alphas, betas, one_minus_alphas_bar_sqrt)
        # cur_x = p_sample_distribute(model, cur_x, i)
        # x_seq.append(cur_x)
    return cur_x, predict

def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(torch.tensor(np.sqrt(2.0 / np.pi)) * (x + 0.044715 * torch.pow(x, 3))))

def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 1.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 1.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.clamp(cdf_plus, min=1e-12))
    log_one_minus_cdf_min = torch.log(torch.clamp(1 - cdf_min, min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(x < -0.999, log_cdf_plus, torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(torch.clamp(cdf_delta, min=1e-12))))
    return log_probs

def normal_kl(mean1, logvar1, mean2, logvar2):
    kl = 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(-logvar2))
    return kl

def entropy(val):
    return (0.5 * (1 + np.log(2. * np.pi))) + 0.5 * np.log(val)

def q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt,noise=None):
    if noise is None:
        noise = torch.randn_like(x_0).cuda()
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    return (alphas_1_m_t * noise + alphas_t * x_0)


def loss_variational(model, x_0,alphas_bar_sqrt, one_minus_alphas_bar_sqrt,posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped,n_steps):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # Perform diffusion for step t
    x_t = q_sample(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt).cuda()
    # Compute the true mean and variance
    true_mean, true_var = q_posterior_mean_variance(x_0.cuda(), x_t, t.cuda(),posterior_mean_coef_1.cuda(),posterior_mean_coef_2.cuda(),posterior_log_variance_clipped.cuda())
    # Infer the mean and variance with our model
    model_mean, model_var = p_mean_variance(model, x_t, t.cuda())
    # Compute the KL loss
    kl = normal_kl(true_mean, true_var, model_mean, model_var)
    kl = torch.mean(kl.view(batch_size, -1), dim=1) / np.log(2.)
    # NLL of the decoder
    # decoder_nll = -discretized_gaussian_log_likelihood(x_0.cuda(), means=model_mean.cuda(), log_scales=0.5 * model_var.cuda())
    # decoder_nll = torch.mean(decoder_nll.view(batch_size, -1), dim=1) / np.log(2.)
    # At the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
    # kl = torch.tensor(kl, dtype=torch.float32)
    # theta = torch.tensor(0)
    # output = torch.where(t.cuda() == theta.cuda(), decoder_nll.cuda(), kl.cuda())
    output = kl.cuda()
    return output.mean(-1)

def q_sample_distribute(x_0, t, att, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)
    alphas_t = extract(alphas_bar_sqrt, t, x_0)
    alphas_1_m_t = extract(one_minus_alphas_bar_sqrt, t, x_0)
    # return (alphas_t * x_0 + alphas_1_m_t * noise)
    return (alphas_t * x_0 + alphas_1_m_t * att)

def compute_loss(true_mean, true_var, model_mean, model_var, betas, alphas, n_steps):
    # the KL divergence between model transition and posterior from data
    KL = normal_kl(true_mean, true_var, model_mean, model_var).float()
    # conditional entropies H_q(x^T|x^0) and H_q(x^1|x^0)
    H_start = entropy(betas[0].float()).float()
    beta_full_trajectory = 1. - torch.exp(torch.sum(torch.log(alphas))).float()
    H_end = entropy(beta_full_trajectory.float()).float()
    H_prior = entropy(torch.tensor([1.])).float()
    n_steps = torch.tensor([n_steps]).cuda()
    negL_bound = KL * n_steps + H_start.cuda() - H_end.cuda() + H_prior.cuda()
    # the negL_bound if this was an isotropic Gaussian model of the data
    negL_gauss = entropy(torch.tensor([1.])).float()
    negL_diff = negL_bound - negL_gauss.cuda()
    L_diff_bits = negL_diff / np.log(2.)
    L_diff_bits_avg = L_diff_bits.mean()
    return L_diff_bits_avg

def loss_likelihood_bound(model, x_0, n_steps, betas, alphas, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, posterior_mean_coef_1, posterior_mean_coef_2, posterior_variance, posterior_log_variance_clipped):
    batch_size = x_0.shape[0]
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # Perform diffusion for step t
    x_t = q_sample_distribute(x_0, t, alphas_bar_sqrt, one_minus_alphas_bar_sqrt)
    # Compute the true mean and variance
    true_mean, true_var = q_posterior_mean_variance(x_0, x_t, t, posterior_mean_coef_1,posterior_mean_coef_2,posterior_log_variance_clipped)
    # Infer the mean and variance with our model
    model_mean, model_var = p_mean_variance(model, x_t.cuda(), t.cuda())
    noise = torch.randn_like(x_t).cuda()
    sample = model_mean + torch.exp(0.5 * model_var) * noise
    print("sample: ", sample)
    # Compute the loss
    return compute_loss(true_mean.cuda(), true_var.cuda(), model_mean, model_var, betas, alphas, n_steps)

def noise_estimation_loss(model, x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, one_minus_alphas_sqrt, n_steps):
    batch_size = x_0.shape[0]
    # c_0 = c_0.to(torch.float32).cuda()
    # Select a random step for each example
    t = torch.randint(0, n_steps, size=(batch_size // 2 + 1,))
    t = torch.cat([t, n_steps - t - 1], dim=0)[:batch_size].long()
    # x0 multiplier
    a = extract(alphas_bar_sqrt, t, x_0).cuda()
    # eps multiplier
    am1 = extract(one_minus_alphas_bar_sqrt, t, x_0).cuda()
    am1_t = extract(one_minus_alphas_sqrt, t, x_0).cuda()
    e = torch.randn_like(x_0).cuda()
    # model input
    x = x_0 * a + e * am1
    output = model(x, t.cuda())
    return (e - output).square().mean()

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class DATA_LOADER(object):
    def __init__(self, opt):

        self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def read_matdataset(self, opt):
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        feature = matcontent['features'].T
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        # numpy array index starts from 0, matlab starts from 1
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1

        self.attribute = torch.from_numpy(matcontent['att'].T).float()

        # attribute_path = f'w2v/AWA2_attribute.pkl'
        # with open(attribute_path, 'rb') as f:
        #     w2v_att = pickle.load(f)
        # assert w2v_att.shape == (85, 300)
        # self.w2v_att = torch.from_numpy(w2v_att).float().cuda()
        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()

                _train_feature = scaler.fit_transform(feature[trainval_loc])
                # _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1 / mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1 / mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float()
                self.test_seen_feature.mul_(1 / mx)
                # self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
            else:
                self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
                self.train_label = torch.from_numpy(label[trainval_loc]).long()
                self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long()
                self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
                self.test_seen_label = torch.from_numpy(label[test_seen_loc]).long()
        else:
            self.train_feature = torch.from_numpy(feature[train_loc]).float()
            self.train_label = torch.from_numpy(label[train_loc]).long()
            self.test_unseen_feature = torch.from_numpy(feature[val_unseen_loc]).float()
            self.test_unseen_label = torch.from_numpy(label[val_unseen_loc]).long()

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))
        self.ntrain = self.train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class + self.ntest_class).long()

        self.train_mapped_label = map_label(self.train_label, self.seenclasses)

    def next_batch_one_class(self, batch_size):
        if self.index_in_epoch == self.ntrain_class:
            self.index_in_epoch = 0
            perm = torch.randperm(self.ntrain_class)
            self.train_class[perm] = self.train_class[perm]

        iclass = self.train_class[self.index_in_epoch]
        idx = self.train_label.eq(iclass).nonzero().squeeze()
        perm = torch.randperm(idx.size(0))
        idx = idx[perm]
        iclass_feature = self.train_feature[idx]
        iclass_label = self.train_label[idx]
        self.index_in_epoch += 1
        return iclass_feature[0:batch_size], iclass_label[0:batch_size], self.attribute[iclass_label[0:batch_size]]

    def next_batch(self, batch_size):
        idx = torch.randperm(self.ntrain)[0:batch_size]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        return batch_feature, batch_label, batch_att

    # select batch samples by randomly drawing batch_size classes
    def next_batch_uniform_class(self, batch_size):
        batch_class = torch.LongTensor(batch_size)
        for i in range(batch_size):
            idx = torch.randperm(self.ntrain_class)[0]
            batch_class[i] = self.train_class[idx]

        batch_feature = torch.FloatTensor(batch_size, self.train_feature.size(1))
        batch_label = torch.LongTensor(batch_size)
        batch_att = torch.FloatTensor(batch_size, self.attribute.size(1))
        for i in range(batch_size):
            iclass = batch_class[i]
            idx_iclass = self.train_label.eq(iclass).nonzero().squeeze()
            idx_in_iclass = torch.randperm(idx_iclass.size(0))[0]
            idx_file = idx_iclass[idx_in_iclass]
            batch_feature[i] = self.train_feature[idx_file]
            batch_label[i] = self.train_label[idx_file]
            batch_att[i] = self.attribute[batch_label[i]]
        return batch_feature, batch_label, batch_att


class align_Embedding(nn.Module):
    def __init__(self, att_in, feature_in, num_out):
        super(align_Embedding, self).__init__()
        self.att_channel = att_in
        self.feature_channel = feature_in
        self.out_channel = num_out
        self.encoder_att = nn.Sequential(nn.Linear(self.att_channel, self.out_channel),
                                         nn.GELU(),
                                         nn.Linear(self.out_channel, 2 * self.out_channel))
        self.encoder_feature = nn.Sequential(nn.Linear(self.feature_channel, self.out_channel),
                                             nn.ReLU(),
                                             nn.Linear(self.out_channel, 2 * self.out_channel))
        self.decoder_att = nn.Sequential(nn.Linear(self.out_channel, self.feature_channel),
                                         nn.GELU(),
                                         nn.Linear(self.feature_channel, self.feature_channel))
        self.decoder_feature = nn.Sequential(nn.Linear(self.out_channel, self.att_channel),
                                             nn.ReLU(),
                                             nn.Linear(self.att_channel, self.att_channel))

    def forward(self, att, f):
        noise = torch.randn([att.size()[0], self.out_channel]).cuda()
        normal_mean = torch.ones([att.size()[0], self.out_channel]).cuda()
        normal_var = torch.zeros([att.size()[0], self.out_channel]).cuda()
        latent_att = self.encoder_att(att)
        latent_att_mean, latent_att_var = torch.split(latent_att, self.out_channel, dim=-1)
        if f != None:
            latent_feature = self.encoder_feature(f)
            latent_feature_mean, latent_feature_var = torch.split(latent_feature, self.out_channel, dim=-1)
            kl_att = normal_kl(normal_mean, normal_var, latent_att_mean, latent_att_var)
            kl_feature = normal_kl(normal_mean, normal_var, latent_feature_mean, latent_feature_var)
            kl_common = normal_kl(latent_att_mean, latent_att_var, latent_feature_mean, latent_feature_var)
            kl_sum = kl_att + kl_feature + kl_common
        latent_att = latent_att_mean + torch.exp(0.5 * latent_att_var) * noise
        out_f_from_att = self.decoder_att(latent_att)
        if f != None:
            latent_feature = latent_feature_mean + torch.exp(0.5 * latent_feature_var) * noise
            out_att_from_f = self.decoder_feature(latent_feature)
            # return out_att_from_f, out_f_from_att, kl_sum.mean(), latent_att, latent_feature
        return out_f_from_att


class CLIP_guidance(nn.Module):
    def __init__(self, att_channel, feature_channel):
        super(CLIP_guidance, self).__init__()
        self.embed = nn.Linear(att_channel, 512)
        self.activate = nn.ReLU()
        self.embed_2 = nn.Linear(512, feature_channel)

    def forward(self, att):
        out = self.embed_2(self.activate(self.embed(att))) # bs x 2048
        out_min = torch.min(out, dim=1)[0].T.repeat(2048, 1).T
        out_max = torch.max(out, dim=1)[0].T.repeat(2048, 1).T
        out_norm = (out - out_min) / (out_max - out_min)
        return out_norm
