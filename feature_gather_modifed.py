# from models.networks.base_network import BaseNetwork
# from models.networks.normalization import get_nonspade_norm_layer#, GAPConcat
# from models.networks.architecture import ResnetBlock as ResnetBlock
# #from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock, SPADEAttentionResnetBlock, ModifiedSPADEResnetBlock
# from models.networks.architecture import VGG19
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset
import torch
# import models.networks as networks
# import util.util as util
import cv2
import torchvision
import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torchvision.models import vgg19
from collections import namedtuple
from torchvision import transforms, datasets

import pickle

# with open('parrot.pkl', 'wb') as f:
#     pickle.dump(mylist, f)


#initialization
data_transform = transforms.Compose([
        transforms.Resize((224,224)),
#         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#getting data dir
path = './datasets_resized/'

#excluded 0,0,0, which is backgorund
label_colours = [(128,0,0),(255,0,0),(0,85,0),(170,0,51),(255,85,0)
                # 1=Hat,  2=Hair,    3=Glove, 4=Sunglasses, 5=UpperClothes
                ,(0,0,85),(0,119,221),(85,85,0),(0,85,85),(85,51,0)
                # 6=Dress, 7=Coat, 8=Socks, 9=Pants, 10=Jumpsuits
                ,(52,86,128),(0,128,0),(0,0,255),(51,170,221),(0,255,255)
                # 11=Scarf, 12=Skirt, 13=Face, 14=LeftArm, 15=RightArm
                 ,(85,255,170),(170,255,85),(255,255,0),(255,170,0)]
                # 16=LeftLeg, 17=RightLeg, 18=LeftShoe, 19=RightShoe
label_colours = [list(x) for x in label_colours]


class Vgg19(torch.nn.Module):
    def __init__(self):
        super(Vgg19, self).__init__()
        features = list(vgg19(pretrained=True).features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 17, 26}: #, 10, 12, 14, 16 19, 21, 23, 25}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['conv1_64', 'conv2_128', 
                                                'conv3_256','conv4_512'])#,
                                               #'conv3_1','conv3_2','conv3_3','conv3_4',
                                               #'conv4_1','conv4_2','conv4_3',])
        return vgg_outputs(*results)

class Gather(torch.nn.Module):
    ''' 
        gather
    '''
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, F, K, use_mask=False):
        super().__init__()
        self.K = K
        self.F = F
        self.softmax_2 = nn.Softmax(dim=2)
        self.use_mask = use_mask

        if use_mask:
            self.a = nn.Parameter(torch.randn(F, K), requires_grad=True)

            # self.register_parameter('a', torch.nn.Parameter(data=torch.zeros(F, K), requires_grad=True))

    '''image_codeing [N,F,H,W]
        s_att (spatial attention): [N,K,H,W]     K-how many segment     s_att(0 or 50)
        feature: [N,K,F]
    ''' 
    def forward(self, image_coding, s_att, att_mask=None):
        '''
            x: b*m*h*w
            c_att: b*K*h*w
        '''
        b, F, h, w = image_coding.size()
        b_, K, h_, w_ = s_att.size()
        assert (b == b_ and h == h_ and w == w_ and self.K == K and self.F == F)
        if self.use_mask:
            b__, K__ = att_mask.size()
            assert (b == b__) and (self.K == K__)

        # gather feature
        s_att_new = self.softmax_2(s_att.view(b, K, h*w)).permute(0, 2, 1)  # b*hw*K
        gather_result = torch.bmm(image_coding.view(b, F, h*w), s_att_new)   # b*F*K = b*F*hw  X b*hw*K
        if self.use_mask:
            att_mask_new = att_mask.view(b__, 1, K__).expand_as(gather_result)
            # gather_result
            # print(att_mask_new.get_device(), gather_result.get_device(), self.a.get_device())
            gather_result = att_mask_new * gather_result + (1-att_mask_new) * self.a.view(1, F, K).expand_as(gather_result)

        return gather_result

def process_binary_mask(im_seg,label_colours):
    output_mask = np.zeros((len(label_colours),224,224))
    for i in range(len(label_colours)):
        k = label_colours[i]
        temp_mask = (im_seg[:,:,0] == k[0]) & (im_seg[:,:,1] == k[1]) & (im_seg[:,:,2] == k[2])
        temp_mask = temp_mask * 50
        output_mask[i] = temp_mask
    return output_mask




def main():
	
    #main functino
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #Initialize Vgg
    vgg_19 = Vgg19().to(device)

    userlist = os.listdir(path)

    g_64  = Gather(F = 64, K = 19)
    g_128 = Gather(F = 128, K = 19)
    g_256 = Gather(F = 256, K = 19)
    g_512 = Gather(F = 512, K = 19)

    for username in userlist:
        print('User: ',username)
        filelist = os.listdir(path+username+'/out/')
        if 'gather.pkl' in filelist:
            l = [filelist.index(i) for i in filelist if 'gather.pkl' in i]
            filelist.pop(l[0])
        filelist = sorted(filelist,key=lambda x: int(os.path.splitext(x)[0]))

        seg_list = []
        img_list = []
        for file in filelist:
            if file.endswith('png'):
                seg_list.append(file)
            elif file.endswith('jpg'):
                img_list.append(file)
        seg_list = sorted(seg_list,key=lambda x: int(os.path.splitext(x)[0]))
        img_list = sorted(img_list,key=lambda x: int(os.path.splitext(x)[0]))

        # exit()
        if len(seg_list)!=0:
            seg_data = []
            im_data = []
            for fname in img_list:
                img_name =  path+username+'/out/'+fname
                seg_name = img_name.replace('jpg','png')
                print(img_name, seg_name)

                im_seg = np.array(Image.open(seg_name).resize((224, 224)))
                binary_mask = torch.tensor(process_binary_mask(im_seg,label_colours)).type('torch.FloatTensor')
                seg_data.append(binary_mask)

                im_ori = data_transform(Image.open(img_name))
                im_data.append(im_ori)

            # predict
            img_loader = torch.utils.data.DataLoader(im_data, shuffle=False, batch_size=1)
            seg_loader = torch.utils.data.DataLoader(seg_data, shuffle=False, batch_size=1)

            

            gathered_feature = {}

            with torch.no_grad():
                for i, data in enumerate(zip(img_loader,seg_loader)):
                    print('img:', i)
                    try:
                        temp = []
                        im_batch = data[0].to(device)
                        seg_batch = data[1].to(device)

                        output = vgg_19.forward(im_batch)

                        gathered_64  = g_64.forward(output[0],seg_batch)
                        gathered_128 = g_128.forward(output[1],F.interpolate(seg_batch,size =(112,112)))
                        gathered_256 = g_256.forward(output[2],F.interpolate(seg_batch,size =(56,56)))
                        gathered_512 = g_512.forward(output[3],F.interpolate(seg_batch,size =(28,28)))
                        temp = [gathered_64.detach().cpu().numpy(), gathered_128.detach().cpu().numpy(), gathered_256.detach().cpu().numpy(), gathered_512.cpu().detach().numpy()]
                        gathered_feature[img_list[i]] = temp
                    except:
                        print('Grayscale image detected', img_list[i])
    #                 gathered_feature.append(temp)

    #         print(len(gathered_feature))
    #         print(gathered_feature[0][0].shape)
    #         print(gathered_feature[0][1].shape)
    #         print(gathered_feature[0][2].shape)
    #         print(gathered_feature[0][3].shape)

            ##TODO: concatenate feature and save pickle file
            pickle_fname = path+username+'/out/gather.pkl'
            with open(pickle_fname, 'wb') as f:
                pickle.dump(gathered_feature, f)
    #         exit()


if __name__ == '__main__':
    main()
