import pickle
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch import nn 
import torch.nn.functional as Fun
import random


print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.is_available())

USE_GPU=False

class FeatureDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, filename, K=20, is_train=True, transform=None):
        pickle_in = open(filename,"rb")
        ps = pickle.load(pickle_in)

        image_mats = list()
        for p in ps:
            image_mats.append(np.asarray(p, dtype=np.float32))
     
        self.transform = transform
        self.is_train = is_train
        self.N = len(image_mats)
        self.K = K

        self.image_feats_train = list()
        self.image_feat_imagepool = list()
        
        for i in range(len(image_mats)):
            # self.image_feats_train.append(torch.from_numpy(image_mats[i][0:-1,:]).cuda())
            self.image_feats_train.append((image_mats[i][0:-1,:]))
            self.image_feat_imagepool.append(image_mats[i][-1,:].tolist())

        self.user_num = len(self.image_feat_imagepool)
        self.image_dim = image_mats[0].shape[1]
        # self.image_feat_imagepool =  torch.from_numpy(np.asarray(self.image_feat_imagepool, dtype=np.float32)).cuda()
        self.image_feat_imagepool =  (np.asarray(self.image_feat_imagepool, dtype=np.float32))

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        permute_list = np.random.permutation(self.image_feats_train[idx].shape[0])
        input_feat_idx = permute_list[0:(self.K)]    #np.random.randint(self.image_feats_train[idx].shape[0], size=self.K+1)
        pos_feat_idx = permute_list[self.K+1]

        neg_idx = random.choice(list(range(0,idx))+list(range(idx+1,self.N)))
        neg_feat_idx = random.choice(list(range(self.image_feats_train[neg_idx].shape[0])))

        # print(idx, pos_feat_idx, input_feat_idx)
        # print(neg_idx, neg_feat_idx)

        Xs = self.image_feats_train[idx][input_feat_idx,:]
        x_pos = self.image_feats_train[idx][pos_feat_idx,:]
        x_neg = self.image_feats_train[neg_idx][neg_feat_idx,:]
        return Xs, x_pos, x_neg

class Net1(nn.Module):
    def __init__(self, image_dim):
        super(Net1, self).__init__()
        self.embedding_dim = 128
        self.embedding_mlp = nn.Sequential(
            nn.Linear(image_dim, 256),
            nn.LeakyReLU(inplace=True),
            nn.Linear(256, self.embedding_dim),
            nn.LeakyReLU(inplace=True)
        )

        # self.mlp2 = nn.Sequential(
        #     nn.Linear(image_dim, 256),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(256, 128),
        #     nn.LeakyReLU(inplace=True)
        # )
        self.attention = nn.Sequential(
            nn.Linear(self.embedding_dim*3, 1),
            nn.LeakyReLU(inplace=True),
        )

        self.ATTENTION = True
    def forward(self, x, x_p=None, x_n=None, is_train=True, test_mode='single'):

        if is_train:
            # get embedding
            N, K, F = x.size()
            # x
            x = x.view(N*K, F)
            x = self.embedding_mlp(x)
            x = x.view(N, K, 128)
            # x_p
            x_p = self.embedding_mlp(x_p.view(N,1,F))
            # x_n
            x_n = self.embedding_mlp(x_n.view(N,1,F))

            ### averaging x
            if not self.ATTENTION:
                x = torch.mean(x, 1, True)
                x_aggregated = x.view(N,1,128)
            else:
                x_max = x.max(1)[0].view(N,1,self.embedding_dim).expand_as(x)
                x_avg = torch.mean(x,1,True).expand_as(x)
                # print(x.size(), x_max.size(), x_avg.size())
                x_all = torch.cat((x,x_max,x_avg), 1)
                x_all = x_all.view(N*K, 3*self.embedding_dim)
                x_attention = self.attention(x_all).view(N,K,1)

                x_attention = Fun.softmax(x_attention, dim=1)
                x_aggregated = x*x_attention
                x_aggregated = x_aggregated.sum(1,keepdim=True)

            ### loss1 contrasive loss
            # positive_distance
            dist_p = torch.sum((x_aggregated-x_p)*(x_aggregated-x_p), 2)
            # negtive_distance
            dist_n = torch.sum((x_aggregated-x_n)*(x_aggregated-x_n), 2)
            # loss
            margin=0.5
            loss = dist_p + torch.clamp(margin-dist_n, min=0.0, max=10000.0)
            loss = torch.mean(loss)

            return loss
        else:
            if test_mode=='average':
                K,F = x.size()
                x = x.view(1,K,F)
                x = self.embedding_mlp(x)
                if not self.ATTENTION:
                    x = torch.mean(x, 1, True)
                    x = x.view(1,128)
                else:
                    N=1
                    x_max = x.max(1)[0].view(N,1,self.embedding_dim).expand_as(x)
                    x_avg = torch.mean(x,1,True).expand_as(x)
                    # print(x.size(), x_max.size(), x_avg.size())
                    x_all = torch.cat((x,x_max,x_avg), 1)
                    x_all = x_all.view(N*K, 3*self.embedding_dim)
                    x_attention = self.attention(x_all).view(N,K,1)
                    x_attention = Fun.softmax(x_attention, dim=1)

                    x = x*x_attention
                    x = x.sum(1,keepdim=True)
                    x = x.view(1,128)

                return x
            elif test_mode=='single':
                x = self.embedding_mlp(x)
                return x

# class PCA(nn.Module):
#     def __init__(self, image_dim):
#         super(PCA, self).__init__()
#         self.fc1 = nn.Linear(image_dim, 256)
#         self.fc2 = nn.Linear(256, image_dim)

#     def forward(self, x):
#         N, K, F = x.size()
#         x = x.view(N*K, F)
#         coding = F.ReLU(self.fc1(x), inplace=True)
#         x_hat = self.fc2(x)
#         x = x.view(N,K,F)
#         return x

def nearest_search(imagepool, feat, K=1):
    assert K==1
    nearest_idx = -1
    nearest_dist = 10000000
    for gt_idx, gt_feat in enumerate(dataset.image_feat_imagepool):
        dist = np.mean(np.abs(gt_feat-feat))
        if dist<nearest_dist:
            nearest_dist = dist
            nearest_idx = gt_idx
    return nearest_idx, nearest_dist

def nearest_search_matrix(imagepool, feat, K=1):
    assert K==1
    dist = np.mean(np.abs(imagepool-feat), axis=1)
    nearest_idx = np.argmin(dist)
    nearest_dist = dist[nearest_idx]
    return nearest_idx, nearest_dist

transform = transforms.Compose([transforms.ToTensor()])
dataset = FeatureDataset("final_gather_with_title_hashtag.pkl",K=10, is_train=True, transform=transform)
# print(dataset[0][0].shape, dataset[0][1].shape, dataset[0][2].shape)
trainloader = DataLoader(dataset, batch_size=16,shuffle=True, num_workers=0)
trainloader_test = DataLoader(dataset, batch_size=1,shuffle=False, num_workers=0)

net = Net1(image_dim=dataset.image_dim)
if USE_GPU:
    net = net.cuda()
# net_pca = PCA(image_dim=dataset.image_dim).cuda()

# criterion = torch.nn.MSELoss(reduction='sum')
criterion = torch.nn.L1Loss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.95)
scheduler = StepLR(optimizer, step_size = 100, gamma = 0.1)
loss_list = []
acc_list = []
# PCA = False
# if PCA:
#     for epoch in range(200):  # loop over the dataset multiple times
#         running_loss = 0.0
#         ## training
#         for i, data in enumerate(trainloader, 0):
#             # get the inputs; data is a list of [inputs, labels]
#             inputs, gt = data
#             inputs, gt = inputs.cuda(), gt.cuda()
#             # zero the parameter gradients
#             optimizer.zero_grad()
#             # forward + backward + optimize
#             outputs = net(inputs)
#             loss = criterion(outputs, gt)
#             loss.backward()
#             optimizer.step()
#             # print statistics
#             running_loss += loss.item()

for epoch in range(500):  # loop over the dataset multiple times
    running_loss = 0.0
    scheduler.step()
    ## training
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        input_feat, pos_feat, neg_feat = data
        if USE_GPU:
            input_feat, pos_feat, neg_feat = input_feat.cuda(), pos_feat.cuda(), neg_feat.cuda()
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        loss = net(input_feat,pos_feat,neg_feat)
        # loss = criterion(outputs, gt)
        # loss = torch.sum(outputs)
        # print(loss)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()

    ## print training result
    if epoch % 1 == 0:    # print every 2000 mini-batches
        print('epoch %d, loss: %.6f , LR: %.10f' % (epoch + 1, running_loss / (len(trainloader)), scheduler.get_lr()[0]))
        loss_list.append(running_loss / (len(trainloader)))
    # test accuracy:

    TEST_EPOCH = 1
    if epoch % TEST_EPOCH == TEST_EPOCH-1:    # print every 2000 mini-batches
        with torch.no_grad():
            image_feat_imagepool = torch.from_numpy(dataset.image_feat_imagepool)
            if USE_GPU:
                image_feat_imagepool = image_feat_imagepool.cuda()
            # print(type(image_feat_imagepool), image_feat_imagepool.size())
            image_feat_mlp_imagepool = net(image_feat_imagepool, is_train=False, test_mode='single').cpu().detach().numpy()
            # print(image_feat_mlp_imagepool.shape)
        # exit()
        TEST_NUM = 1
        oo=0.
        total=0.
        with torch.no_grad():
            for _ in range(TEST_NUM):
                # for sample_idx, data in enumerate(trainloader_test, 0):
                for sample_idx, sample in enumerate(dataset):
                    input_feat, pos_feat, neg_feat = sample
                    input_feat, pos_feat, neg_feat = torch.from_numpy(input_feat), torch.from_numpy(pos_feat), torch.from_numpy(neg_feat)
                    if USE_GPU:
                        input_feat, pos_feat, neg_feat = input_feat.cuda(), pos_feat.cuda(), neg_feat.cuda() 
                    output = net(input_feat, is_train=False, test_mode='average').cpu().detach().numpy()
                    # print(output.shape, image_feat_mlp_imagepool.shape)
                    # do KNN
                    nearest_index, nearest_dist = nearest_search_matrix(image_feat_mlp_imagepool, output)
                    # print(sample_idx, nearest_index, nearest_dist)
                    oo += 1 if (nearest_index==sample_idx) else 0
                    total += 1
                    # print statistics
        # print(oo,  total, 1, float(dataset.user_num))
        print('epoch %d, acc: %.6f, random guess: %.6f' % (epoch + 1, oo / total, 1/float(dataset.user_num)))
        
        acc_list.append(oo / total)

print('Finished Training')