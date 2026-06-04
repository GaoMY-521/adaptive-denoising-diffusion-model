
import os, sys, torch
import numpy as np
import scipy.io as sio
# import pandas as pd
from PIL import Image
from sklearn import preprocessing
from torchvision import datasets, transforms
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import models
import torch.nn as nn


os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class BaseDataset(Dataset):
    def __init__(self, dataset_path, image_files, labels, transform=None):
        super(BaseDataset, self).__init__()
        self.dataset_path = dataset_path
        self.image_files = image_files
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image_file = self.image_files[idx]
        image_file = os.path.join(self.dataset_path, image_file)
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class UNIDataloader():
    def __init__(self, config):
        self.config = config
        with open(config.pkl_path, 'rb') as f:
            self.info = pickle.load(f)

        self.seenclasses = self.info['seenclasses'].to(config.device)
        self.unseenclasses = self.info['unseenclasses'].to(config.device)

        (self.train_set,
         self.test_seen_set,
         self.test_unseen_set) = self.torch_dataset()

        self.train_loader = DataLoader(self.train_set,
                                       batch_size=config.batch_size,
                                       shuffle=True,
                                       num_workers=config.num_workers)
        self.test_seen_loader = DataLoader(self.test_seen_set,
                                           batch_size=config.batch_size,
                                           shuffle=False,
                                           num_workers=config.num_workers)
        self.test_unseen_loader = DataLoader(self.test_unseen_set,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             num_workers=config.num_workers)

    def torch_dataset(self):
        data_transforms = transforms.Compose([
            transforms.Resize(self.config.img_size),
            transforms.CenterCrop(self.config.img_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        baseset = BaseDataset(self.config.dataset_path,
                              self.info['image_files'],
                              self.info['labels'],
                              data_transforms)

        train_set = Subset(baseset, self.info['trainval_loc'])
        test_seen_set = Subset(baseset, self.info['test_seen_loc'])
        test_unseen_set = Subset(baseset, self.info['test_unseen_loc'])

        return train_set, test_seen_set, test_unseen_set

class CustomedDataset(Dataset):
    def __init__(self, dataset, img_dir, file_paths, transform=None):
        self.dataset = dataset
        self.matcontent = sio.loadmat(file_paths) # resnet101.mat
        self.image_files = np.squeeze(self.matcontent['image_files']) # 取出所有图像的文件名
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx][0]
        if self.dataset == 'CUB':
            split_idx = 6
        elif self.dataset == 'SUN':
            split_idx = 7
        elif self.dataset == 'AWA2':
            split_idx = 5
        image_file = os.path.join(self.img_dir,
                                    '/'.join(image_file.split('/')[split_idx:]))
        image = Image.open(image_file)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label==classes[i]] = i

    return mapped_label

class CUBDataLoader():
    def __init__(self, data_path, is_scale=False,
                 is_unsupervised_attr=False, is_balance=True):
        print(data_path)
        sys.path.append(data_path)
        self.data_path = data_path
        self.dataset = 'CUB'
        # print('$'*30)
        # print(self.dataset)
        # print('$'*30)
        self.datadir = os.path.join(self.data_path, 'data/{}/'.format(self.dataset))
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.all_features, self.labels, self.trainval_loc, self.test_seen_loc, self.test_unseen_loc = self.preprocessing()
        self.read_matdataset()
        self.get_idx_classes()

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].cuda()
        batch_label =  self.data['train_seen']['labels'][idx].cuda()
        # batch_att = self.att[batch_label].cuda()
        return batch_label, batch_feature # , batch_att
    
    def get_idx_classes(self):
        print('get_idx_classes')
        n_classes = self.seenclasses.size(0) # 全部可见类个数
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            # idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy() # 返回非零元素索引，也就是返回标签为第i个可见类的样本索引
            idx_c = torch.nonzero(train_label == i).cpu().numpy()  # 返回非零元素索引，也就是返回标签为第i个可见类的样本索引
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c) # 按照类别顺序排列样本索引
        return self.idxs_list

    def preprocessing(self):
        img_dir = f'link/data/standard_CUB/CUB_200_2011'
        file_paths = f'link/data/{self.dataset}/res101.mat'
        # attribute_path = f'w2v/{self.config.dataset}_attribute.pkl'

        model = models.resnet101(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-2])  # 删除最后两层（全局平均池化和先行层），并验证
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        model = model.cuda()

        # model = SwinTransformer(img_size=224)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        # model = model.cuda()
        # weights = torch.load('swin_tiny_patch4_window7_224.pth')
        # model.module.load_state_dict(weights['model'], strict=True)

        # model = SwinTransformerModel(num_class=312)
        # model = torch.nn.DataParallel(model, device_ids=[0, 1])
        # model = model.cuda()
        # weights = torch.load('checkpoint/CUB_zsl_swin/100.pth')
        # weights['patch_embedding_weight'] = nn.Parameter(torch.randn(48, 8))
        # model.module.load_state_dict(weights, strict=True)

        data_transforms = transforms.Compose([
            transforms.Resize(512),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

        Dataset = CustomedDataset(self.dataset, img_dir, file_paths, data_transforms)  # 读取图片
        dataset_loader = torch.utils.data.DataLoader(Dataset,
                                                     batch_size=128,
                                                     shuffle=False,
                                                     num_workers=8)

        with torch.no_grad():
            all_features = []
            for i, imgs in enumerate(dataset_loader):
                # imgs = imgs.to(config.device)
                imgs = imgs.cuda()
                features = model(imgs)  # 提取特征，
                all_features.append(features.cpu().numpy())
            all_features = np.concatenate(all_features, axis=0)

        matcontent = Dataset.matcontent
        labels = matcontent['labels'].astype(int).squeeze() - 1  # 取出每个样本对应标签，取值范围0-199

        split_path = os.path.join(f'link/data/{self.dataset}/att_splits.mat')
        matcontent = sio.loadmat(split_path)
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        # train_loc = matcontent['train_loc'].squeeze() - 1
        # val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1
        # att = matcontent['att'].T
        # original_att = matcontent['original_att'].T

        # construct attribute w2v
        # with open(attribute_path, 'rb') as f:
        #     w2v_att = pickle.load(f)
        # if self.config.dataset == 'CUB':
        #     assert w2v_att.shape == (312, 300)
        # elif self.config.dataset == 'SUN':
        #     assert w2v_att.shape == (102, 300)
        #  elif self.config.dataset == 'AWA2':
        #     assert w2v_att.shape == (85, 300)

        return all_features, labels, trainval_loc, test_seen_loc, test_unseen_loc


    def read_matdataset(self):
        path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        ## tic = time.time()
        ## hf = h5py.File(path, 'r')
        # features = np.array(hf.get('feature_map'))
        print('load resnet-101 prepocessed features')
        features = self.all_features
        ## shape = features.shape
        ## features = features.reshape(shape[0],shape[1],shape[2]*shape[3])
        ## pdb.set_trace()
        # labels = np.array(hf.get('labels'))
        labels = self.labels
        # trainval_loc = np.array(hf.get('trainval_loc'))
        trainval_loc = self.trainval_loc
        ## train_loc = np.array(hf.get('train_loc')) #--> train_feature = TRAIN SEEN
        ## val_unseen_loc = np.array(hf.get('val_unseen_loc')) #--> test_unseen_feature = TEST UNSEEN
        #test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_seen_loc = self.test_seen_loc
        # test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        test_unseen_loc = self.test_unseen_loc

        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).cuda() # 全部可见类类别
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).cuda() # 全部未见类类别
        self.ntrain = train_feature.size()[0] # 训练样本数量
        self.ntrain_class = self.seenclasses.size(0) # 可见类类别总数（即训练类别总数）
        self.ntest_class = self.unseenclasses.size(0) # 未见类类别总数
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long() # 全部类别数量
        train_label = map_label(train_label.cuda(), self.seenclasses.cuda())

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label


class SUNDataLoader():
    def __init__(self, data_path, device, is_scale=False,
                 is_unsupervised_attr=False, is_balance=True):
        print(data_path)
        sys.path.append(data_path)
        self.data_path = data_path
        self.device = device
        self.dataset = 'SUN'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = os.path.join(self.data_path, 'data/{}/'.format(self.dataset))
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()
        self.I = torch.eye(self.allclasses.size(0)).to(device)

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label =  self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list
        
    def read_matdataset(self):
        path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)

        print('_____')
        print(path)
        # tic = time.time()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            assert w2v_class.shape == (50,300)
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).to(self.device)
            self.att = torch.mm(U,torch.diag(s)).to(self.device)
            self.normalize_att = torch.mm(U,torch.diag(s)).to(self.device)
            
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            self.att = torch.from_numpy(att).float().to(self.device)
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().to(self.device)
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            
            self.normalize_att = self.original_att/100
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label


class AWA2DataLoader():
    def __init__(self, data_path, device, is_scale=False,
                 is_unsupervised_attr=False, is_balance=True):
        print(data_path)
        sys.path.append(data_path)
        self.data_path = data_path
        self.device = device
        self.dataset = 'AWA2'
        print('$'*30)
        print(self.dataset)
        print('$'*30)
        self.datadir = os.path.join(self.data_path, 'data/{}/'.format(self.dataset))
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.is_scale = is_scale
        self.is_balance = is_balance
        if self.is_balance:
            print('Balance dataloader')
        self.is_unsupervised_attr = is_unsupervised_attr
        self.read_matdataset()
        self.get_idx_classes()

    def next_batch(self, batch_size):
        if self.is_balance:
            idx = []
            n_samples_class = max(batch_size //self.ntrain_class,1)
            sampled_idx_c = np.random.choice(np.arange(self.ntrain_class),min(self.ntrain_class,batch_size),replace=False).tolist()
            for i_c in sampled_idx_c:
                idxs = self.idxs_list[i_c]
                idx.append(np.random.choice(idxs,n_samples_class))
            idx = np.concatenate(idx)
            idx = torch.from_numpy(idx)
        else:
            idx = torch.randperm(self.ntrain)[0:batch_size]
    
        batch_feature = self.data['train_seen']['resnet_features'][idx].to(self.device)
        batch_label =  self.data['train_seen']['labels'][idx].to(self.device)
        batch_att = self.att[batch_label].to(self.device)
        return batch_label, batch_feature, batch_att
    
    def get_idx_classes(self):
        n_classes = self.seenclasses.size(0)
        self.idxs_list = []
        train_label = self.data['train_seen']['labels']
        for i in range(n_classes):
            idx_c = torch.nonzero(train_label == self.seenclasses[i].cpu()).cpu().numpy()
            idx_c = np.squeeze(idx_c)
            self.idxs_list.append(idx_c)
        return self.idxs_list

    def read_matdataset(self):
        path= self.datadir + 'feature_map_ResNet_101_{}.hdf5'.format(self.dataset)
        print('_____')
        print(path)
        # tic = time.clock()
        hf = h5py.File(path, 'r')
        features = np.array(hf.get('feature_map'))
        labels = np.array(hf.get('labels'))
        trainval_loc = np.array(hf.get('trainval_loc'))
        test_seen_loc = np.array(hf.get('test_seen_loc'))
        test_unseen_loc = np.array(hf.get('test_unseen_loc'))
        
        if self.is_unsupervised_attr:
            print('Unsupervised Attr')
            class_path = './w2v/{}_class.pkl'.format(self.dataset)
            with open(class_path,'rb') as f:
                w2v_class = pickle.load(f)
            assert w2v_class.shape == (50,300)
            w2v_class = torch.tensor(w2v_class).float()
            
            U, s, V = torch.svd(w2v_class)
            reconstruct = torch.mm(torch.mm(U,torch.diag(s)),torch.transpose(V,1,0))
            print('sanity check: {}'.format(torch.norm(reconstruct-w2v_class).item()))
            
            print('shape U:{} V:{}'.format(U.size(),V.size()))
            print('s: {}'.format(s))
            
            self.w2v_att = torch.transpose(V,1,0).to(self.device)
            self.att = torch.mm(U,torch.diag(s)).to(self.device)
            self.normalize_att = torch.mm(U,torch.diag(s)).to(self.device)
        else:
            print('Expert Attr')
            att = np.array(hf.get('att'))
            
            print("threshold at zero attribute with negative value")
            att[att<0]=0
            
            self.att = torch.from_numpy(att).float().to(self.device)
            
            original_att = np.array(hf.get('original_att'))
            self.original_att = torch.from_numpy(original_att).float().to(self.device)
            
            w2v_att = np.array(hf.get('w2v_att'))
            self.w2v_att = torch.from_numpy(w2v_att).float().to(self.device)
            
            self.normalize_att = self.original_att/100
        
        train_feature = features[trainval_loc]
        test_seen_feature = features[test_seen_loc]
        test_unseen_feature = features[test_unseen_loc]
        if self.is_scale:
            scaler = preprocessing.MinMaxScaler()
    
            train_feature = scaler.fit_transform(train_feature)
            test_seen_feature = scaler.fit_transform(test_seen_feature)
            test_unseen_feature = scaler.fit_transform(test_unseen_feature)

        train_feature = torch.from_numpy(train_feature).float() #.to(self.device)
        test_seen_feature = torch.from_numpy(test_seen_feature) #.float().to(self.device)
        test_unseen_feature = torch.from_numpy(test_unseen_feature) #.float().to(self.device)

        train_label = torch.from_numpy(labels[trainval_loc]).long() #.to(self.device)
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]) #.long().to(self.device)
        test_seen_label = torch.from_numpy(labels[test_seen_loc]) #.long().to(self.device)

        self.seenclasses = torch.from_numpy(np.unique(train_label.cpu().numpy())).to(self.device)
        self.unseenclasses = torch.from_numpy(np.unique(test_unseen_label.cpu().numpy())).to(self.device)
        self.ntrain = train_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()

        self.data = {}
        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels']= train_label

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen']['labels'] = test_unseen_label