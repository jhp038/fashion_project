# import torch
import random
import pickle as pkl
import numpy as np
import os
from sys import getsizeof


#initialization
def main():
    path = './datasets_resized/'
    userlist = os.listdir(path)
    output = []
    for username in userlist:
        pkl_name =  path+username+'/out/gather.pkl'
        print(username)
        with open(pkl_name,'rb') as pkl_file:
            data = pkl.load(pkl_file)
            pkl_file.close()
            user_data = []
            index = 0
            for k,v in data.items():
    #             print(k)
                data_64 = np.reshape(v[0], v[0].shape[1] * v[0].shape[2])
                data_128 = np.reshape(v[1], v[1].shape[1] * v[1].shape[2])
                data_256 = np.reshape(v[2], v[2].shape[1] * v[2].shape[2])
                data_512 = np.reshape(v[3], v[3].shape[1] * v[3].shape[2])
                temp = np.concatenate((data_64,data_128,data_256,data_512))
                user_data.append(temp)
                index +=1
            pkl_file.close()
        output.append(user_data)
    with open("total_img_feature.pkl", "wb") as fout:
        pkl.dump(output, fout, protocol=pkl.HIGHEST_PROTOCOL)
    print(getsizeof(output))

if __name__ == '__main__':
    main()
