import h5py
import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
#from sklearn.metrics.pairwise import cosine_similarity
import sys
import pdb
import h5py
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

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
        if opt.vit:
            path = opt.dataroot+"/"+opt.dataset+"/"+"feature_"+opt.dataset+"_VIT_224.hdf5"
            hf = h5py.File(path, 'r')
            feature = np.array(hf.get('feature_map')).squeeze()
            label = np.array(hf.get('labels'))
        else:
            if opt.size == 448:
                path = opt.dataroot+"/"+opt.dataset+"/"+"feature_"+opt.dataset+"_448.hdf5"
                hf = h5py.File(path, 'r')
                feature = np.array(hf.get('feature_map')).squeeze()
                label = np.array(hf.get('labels'))
            else:
                matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
                feature = matcontent['features'].T
                label = matcontent['labels'].astype(int).squeeze() - 1
        # matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.image_embedding + ".mat")
        # feature = matcontent['features'].T
        # label = matcontent['labels'].astype(int).squeeze() - 1
        matcontent = sio.loadmat(opt.dataroot + "/" + opt.dataset + "/" + opt.class_embedding + "_splits.mat")
        trainval_loc = matcontent['trainval_loc'].squeeze() - 1
        train_loc = matcontent['train_loc'].squeeze() - 1
        val_unseen_loc = matcontent['val_loc'].squeeze() - 1
        test_seen_loc = matcontent['test_seen_loc'].squeeze() - 1
        test_unseen_loc = matcontent['test_unseen_loc'].squeeze() - 1 
        self.allclasses_name = matcontent['allclasses_names']

        self.feature = feature
        self.label = label 

        # self.attribute = torch.from_numpy(matcontent['att'].T).float()
        # self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))

        #visual_pro
        # path_vis_pro = "/home/hwj/CVPR_CODE/data/SUN/SUN_visual_protopyte.hdf5"
        # #path_vis_pro = opt.dataroot+"/"+opt.dataset+"/"+opt.dataset+"_visual_prototype.hdf5"
        # hf_vis = h5py.File(path_vis_pro, 'r') 
        # self.visual_prototype_norm = torch.tensor(hf_vis.get('visual_prototype_norm')).float()
        # self.visual_prototype = torch.tensor(hf_vis.get('visual_prototype')).float()

        #vgse
        # self.vgse_feature = torch.from_numpy(sio.loadmat("/home/hwj/CVPR_CODE/"+opt.dataroot+"/"+opt.dataset+"/"+"VGSE_SMO_splits.mat")['att'].T).float()
        # self.vgse_feature /= self.vgse_feature.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.vgse_feature.size(0),self.vgse_feature.size(1)) #[200,469]
        #print("ok")
        if opt.dataset == 'CUB':
            self.update_att1 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_CUB_1024_size_448_seed_0.mat')['att']).float()
            self.update_att2 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_CUB_1024_size_448_seed_250.mat')['att']).float()
            self.update_att3 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_CUB_1024_size_448_seed_1000.mat')['att']).float()
            self.attribute = (self.update_att1+self.update_att3)/2.0
        # self.update_att1 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_cub_1024_seed_0.mat')['att']).float()
        # self.update_att2 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_cub_1024_seed_250.mat')['att']).float()
        # self.update_att3 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/CUB/att_cub_1024_seed_1000.mat')['att']).float()
        if opt.dataset == 'SUN':
            self.update_att1 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/SUN/att_SUN_seed_0.mat')['att']).float()
            self.update_att2 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/SUN/att_SUN_seed_250.mat')['att']).float()
            self.update_att3 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/SUN/att_SUN_seed_1000.mat')['att']).float()
            self.attribute = (self.update_att1+self.update_att2+self.update_att3)/3.0
        if opt.dataset == 'AWA2':
            self.update_att1 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/AWA2/att_AWA2_seed_0.mat')['att']).float()
            self.update_att2 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/AWA2/att_AWA2_seed_250.mat')['att']).float()
            self.update_att3 = torch.from_numpy(sio.loadmat('/home/hwj/CVPR_CODE/data/AWA2/att_AWA2_seed_1000.mat')['att']).float()
            self.attribute = (self.update_att1+self.update_att2+self.update_att3)/3.0
       
        self.attribute /= self.attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(self.attribute.size(0),self.attribute.size(1))
        self.update_att = (self.update_att1+self.update_att3)/2.0
        #self.attribute = 0.5*self.attribute+0.5*self.update_att
        #self.attribute = self.update_att3

        #cos = torch.cosine_similarity(self.attribute1,self.attribute,dim=0,eps=1e-08)
        #cos = torch.mm(self.attribute,self.attribute1.T)
        #cos  = cosine_similarity(np.array(self.attribute1),np.array(self.attribute))
       # import numpy as np
        
        #x = np.random.rand(100).reshape(10,10)
        # x = plt.imshow(cos[0:10,0:10], cmap=plt.cm.hot, vmin=0, vmax=1)
        # plt.colorbar()
        # plt.savefig('./update_att_att.png')
        # plt.show()


        # print(cos)



        if not opt.validation:
            if opt.preprocessing:
                if opt.standardization:
                    print('standardization...')
                    scaler = preprocessing.StandardScaler()
                else:
                    scaler = preprocessing.MinMaxScaler()
                
                _train_feature = scaler.fit_transform(feature[trainval_loc])
                _test_seen_feature = scaler.transform(feature[test_seen_loc])
                _test_unseen_feature = scaler.transform(feature[test_unseen_loc])
                self.train_feature = torch.from_numpy(_train_feature).float()
                mx = self.train_feature.max()
                self.train_feature.mul_(1/mx)
                self.train_label = torch.from_numpy(label[trainval_loc]).long() 
                self.test_unseen_feature = torch.from_numpy(_test_unseen_feature).float()
                self.test_unseen_feature.mul_(1/mx)
                self.test_unseen_label = torch.from_numpy(label[test_unseen_loc]).long() 
                self.test_seen_feature = torch.from_numpy(_test_seen_feature).float() 
                self.test_seen_feature.mul_(1/mx)
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
        
        #####few-shot setting
        # ph = [[] for _ in range(250)]
        # for fi in range(len(self.train_feature)):
            # ph[self.train_label[fi]].append(fi)
        # ph = [i for i in ph if i !=[]]
        # training=True
        # ph = ph[0:self.seenclasses.size(0)]

        # feature = []
        
        # for fi in range(len(np.unique(self.train_label))):
            # g = ph[fi][0:10]
            # feature = np.concatenate((feature, g))
        # print("feature:", feature.shape)
        # self.train_feature_new = np.concatenate(np.expand_dims(self.train_feature[feature.astype(int)], axis=1))
        # self.train_feature = torch.from_numpy(self.train_feature_new)
        # self.train_label = self.train_label[feature.astype(int)]

        
        self.ntrain = self.train_feature.size()[0]
        self.ntest_seen = self.test_seen_feature.size()[0]
        self.ntest_unseen = self.test_unseen_feature.size()[0]
        self.ntrain_class = self.seenclasses.size(0)
        self.ntest_class = self.unseenclasses.size(0)
        self.train_class = self.seenclasses.clone()
        self.allclasses = torch.arange(0, self.ntrain_class+self.ntest_class).long()
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
        #print("***********",self.ntrain_class,self.ntest_class)

    def next_seen_batch(self, seen_batch):
        idx = torch.randperm(self.ntrain)[0:seen_batch]
        batch_feature = self.train_feature[idx]
        batch_label = self.train_label[idx]
        batch_att = self.attribute[batch_label]
        #batch_att = torch.cat([self.attribute[batch_label],self.vgse_feature[batch_label]],dim=1)
        #batch_att = self.update_att[batch_label]
        return batch_feature, batch_label, batch_att

    # def next_seen_batch(self, seen_batch):
    #     idx = torch.randperm(self.ntrain)[0:seen_batch]
    #     batch_feature = self.train_feature[idx]
    #     batch_label = self.train_label[idx]
    #     batch_att = torch.cat([self.attribute[batch_label],self.update_att[batch_label]],dim=1)
    #     return batch_feature, batch_label, batch_att