import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import copy


#Code to generate Synthetic dataset
def self_implemented_synthetic_data(train_num = 100000, test_num = 10000, dim_num = 50, linear_num = 1, slab3_num = 1, slab5_num = 1, slab7_num = 1, 
                                    linear_margin = 0.4, slab3_margin = 0.1, slab5_margin = 0.1, slab7_margin = 0.1, width = 1.0, center = 0.0, 
                                    linear_p = 0.0, attack_p = 0.0, batch_size = 256):
    left_bound = center - width
    right_bound = center + width
    total_num = train_num + test_num

    if attack_p == 0 :
        Y1 = torch.zeros(int(total_num/2))
        Y2 = torch.ones(int(total_num/2))
        Y = torch.cat([Y1, Y2], dim = 0)
    else:
        Y1 = torch.zeros(int(total_num/2) - int(total_num/2*attack_p))
        Y3 = torch.ones(int(total_num/2*attack_p))
        Y2 = torch.ones(int(total_num/2) - int(total_num/2*attack_p))
        Y4 = torch.zeros(int(total_num/2*attack_p))
        Y = torch.cat([Y1, Y3, Y2, Y4], dim = 0)

    combined = Y.reshape(-1, 1)

    for i in range(linear_num):
        # set linear feature
        if isinstance(linear_margin, float):
            X1 = torch.from_numpy(np.random.uniform(left_bound, center-linear_margin/2, int(total_num/2)))
            X2 = torch.from_numpy(np.random.uniform(center+linear_margin/2, right_bound, int(total_num/2)))
            X = torch.cat([X1, X2], dim = 0)
            #If there's corruption area in linear dimension
            if isinstance(linear_p, float):
                if linear_p != 0:
                    idx = np.random.choice(X.shape[0], int(X.shape[0]*linear_p), replace=False)
                    X[idx] = torch.from_numpy(np.random.uniform(center-linear_margin/2, center+linear_margin/2, int(X.shape[0]*linear_p)))
            else:
                if linear_p[i] != 0:
                    idx = np.random.choice(X.shape[0], int(X.shape[0]*linear_p[i]), replace=False)
                    X[idx] = torch.from_numpy(np.random.uniform(center-linear_margin/2, center+linear_margin/2, int(X.shape[0]*linear_p[i])))
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
        else:
            X1 = torch.from_numpy(np.random.uniform(left_bound, center-linear_margin[i]/2, int(total_num/2)))
            X2 = torch.from_numpy(np.random.uniform(center+linear_margin[i]/2, right_bound, int(total_num/2)))
            X = torch.cat([X1, X2], dim = 0)
            #If there's corruption area in linear dimension
            if isinstance(linear_p, float):
                if linear_p != 0:
                    idx = np.random.choice(X.shape[0], X.shape[0]*linear_p, replace=False)
                    X[idx] = torch.from_numpy(np.random.uniform(center-linear_margin[i]/2, center+linear_margin[i]/2, X.shape[0]*linear_p))
            else:
                if linear_p[i] != 0:
                    idx = np.random.choice(X.shape[0], X.shape[0]*linear_p[i], replace=False)
                    X[idx] = torch.from_numpy(np.random.uniform(center-linear_margin[i]/2, center+linear_margin[i]/2, X.shape[0]*linear_p[i]))
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
            
    
    for i in range(slab3_num):
        combined_1 = combined[0:int(total_num/2), :]
        combined_2 = combined[int(total_num/2):total_num, :]
        combined_1 = combined_1[torch.randperm(combined_1.shape[0])].view(combined_1.size())
        combined_2 = combined_2[torch.randperm(combined_2.shape[0])].view(combined_2.size())
        combined = torch.cat([combined_1, combined_2], dim = 0)

        if isinstance(slab3_margin, float):
            length = (right_bound - left_bound - slab3_margin*2)/3
            bounds = [left_bound, left_bound+length, center-length/2, center+length/2, right_bound-length, right_bound]
        else:
            length = (right_bound - left_bound - slab3_margin[i]*2)/3
            bounds = [left_bound, left_bound+length, center-length/2, center+length/2, right_bound-length, right_bound]

        #Zero label in the middle
        if 0==0:#np.random.randint(0,2) == 0:
            X1 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], combined_1.shape[0]))
            X_2_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_2.shape[0]/2)))
            X_2_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], combined_2.shape[0] - int(combined_2.shape[0]/2)))
            X2 = torch.cat([X_2_1, X_2_2], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)

        #label one in the middle
        else:
            X2 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], combined_2.shape[0]))
            X_1_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_1.shape[0]/2)))
            X_1_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], combined_1.shape[0] - int(combined_1.shape[0]/2)))
            X1 = torch.cat([X_1_1, X_1_2], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
        
    
    for i in range(slab5_num):
        combined_1 = combined[0:int(total_num/2), :]
        combined_2 = combined[int(total_num/2):total_num, :]
        combined_1 = combined_1[torch.randperm(combined_1.shape[0])].view(combined_1.size())
        combined_2 = combined_2[torch.randperm(combined_2.shape[0])].view(combined_2.size())
        combined = torch.cat([combined_1, combined_2], dim = 0)

        if isinstance(slab5_margin, float):
            length = (right_bound - left_bound - slab5_margin*4)/5
            bounds = [left_bound, left_bound+length, left_bound+length+slab5_margin, left_bound+2*length+slab5_margin, center-length/2, center+length/2, right_bound-2*length-slab5_margin, right_bound-length-slab5_margin, right_bound-length, right_bound]
        else:
            length = (right_bound - left_bound - slab5_margin[i]*4)/5
            bounds = [left_bound, left_bound+length, left_bound+length+slab5_margin[i], left_bound+2*length+slab5_margin[i], center-length/2, center+length/2, right_bound-2*length-slab5_margin[i], right_bound-length-slab5_margin[i], right_bound-length, right_bound]

        #Zero label in the middle
        if 0 == 0:#np.random.randint(0,2) == 0:
            X_1_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_1.shape[0]/3)))
            X_1_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], int(combined_1.shape[0]/3)))
            X_1_3 = torch.from_numpy(np.random.uniform(bounds[8], bounds[9], combined_1.shape[0]-2*int(combined_1.shape[0]/3)))
            X_2_1 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], int(combined_2.shape[0]/2)))
            X_2_2 = torch.from_numpy(np.random.uniform(bounds[6], bounds[7], combined_2.shape[0] - int(combined_2.shape[0]/2)))
            X2 = torch.cat([X_2_1, X_2_2], dim = 0)
            X1 = torch.cat([X_1_1, X_1_2, X_1_3], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)

        #label one in the middle
        else:
            X_1_1 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], int(combined_1.shape[0]/2)))
            X_1_2 = torch.from_numpy(np.random.uniform(bounds[6], bounds[7], combined_1.shape[0] - int(combined_1.shape[0]/2)))
            X_2_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_2.shape[0]/3)))
            X_2_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], int(combined_2.shape[0]/3)))
            X_2_3 = torch.from_numpy(np.random.uniform(bounds[8], bounds[9], combined_2.shape[0] - 2*int(combined_2.shape[0]/3)))
            X1 = torch.cat([X_1_1, X_1_2], dim = 0)
            X2 = torch.cat([X_2_1, X_2_2, X_2_3], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
    
    for i in range(slab7_num):
        combined_1 = combined[0:int(total_num/2), :]
        combined_2 = combined[int(total_num/2):total_num, :]
        combined_1 = combined_1[torch.randperm(combined_1.shape[0])].view(combined_1.size())
        combined_2 = combined_2[torch.randperm(combined_2.shape[0])].view(combined_2.size())
        combined = torch.cat([combined_1, combined_2], dim = 0)

        if isinstance(slab7_margin, float):
            length = (right_bound - left_bound - slab7_margin*6)/7
            bounds = [left_bound, left_bound+length, left_bound+length+slab5_margin, left_bound+2*length+slab5_margin, left_bound+2*length+2*slab5_margin, left_bound+3*length+2*slab5_margin, center-length/2, center+length/2, right_bound-3*length-2*slab5_margin, right_bound-2*length-2*slab5_margin, right_bound-2*length-slab5_margin, right_bound-length-slab5_margin, right_bound-length, right_bound]
        else:
            length = (right_bound - left_bound - slab7_margin[i]*6)/7
            bounds = [left_bound, left_bound+length, left_bound+length+slab5_margin[i], left_bound+2*length+slab5_margin[i], left_bound+2*length+2*slab5_margin[i], left_bound+3*length+2*slab5_margin[i], center-length/2, center+length/2, right_bound-3*length-2*slab5_margin[i], right_bound-2*length-2*slab5_margin[i], right_bound-2*length-slab5_margin[i], right_bound-length-slab5_margin[i], right_bound-length, right_bound]

        #Zero label in the middle
        if 0==0:#np.random.randint(0,2) == 0:
            X_2_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_2.shape[0]/4)))
            X_2_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], int(combined_2.shape[0]/4)))
            X_2_3 = torch.from_numpy(np.random.uniform(bounds[8], bounds[9], int(combined_2.shape[0]/4)))
            X_2_4 = torch.from_numpy(np.random.uniform(bounds[12], bounds[13], combined_2.shape[0] - 3*int(combined_2.shape[0]/4)))
            X_1_1 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], int(combined_1.shape[0]/3)))
            X_1_2 = torch.from_numpy(np.random.uniform(bounds[6], bounds[7], int(combined_1.shape[0]/3)))
            X_1_3 = torch.from_numpy(np.random.uniform(bounds[10], bounds[11], combined_1.shape[0] - 2*int(combined_1.shape[0]/3)))
            X2 = torch.cat([X_2_1, X_2_2, X_2_3, X_2_4], dim = 0)
            X1 = torch.cat([X_1_1, X_1_2, X_1_3], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)

        #label one in the middle
        else:
            X_1_1 = torch.from_numpy(np.random.uniform(bounds[0], bounds[1], int(combined_1.shape[0]/4)))
            X_1_2 = torch.from_numpy(np.random.uniform(bounds[4], bounds[5], int(combined_1.shape[0]/4)))
            X_1_3 = torch.from_numpy(np.random.uniform(bounds[8], bounds[9], int(combined_1.shape[0]/4)))
            X_1_4 = torch.from_numpy(np.random.uniform(bounds[12], bounds[13], combined_1.shape[0] - 3*int(combined_1.shape[0]/4)))
            X_2_1 = torch.from_numpy(np.random.uniform(bounds[2], bounds[3], int(combined_2.shape[0]/3)))
            X_2_2 = torch.from_numpy(np.random.uniform(bounds[6], bounds[7], int(combined_2.shape[0]/3)))
            X_2_3 = torch.from_numpy(np.random.uniform(bounds[10], bounds[11], combined_2.shape[0] - 2*int(combined_2.shape[0]/3)))
            X2 = torch.cat([X_2_1, X_2_2, X_2_3], dim = 0)
            X1 = torch.cat([X_1_1, X_1_2, X_1_3, X_1_4], dim = 0)
            X = torch.cat([X1, X2], dim = 0)
            combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
    
    for i in range(dim_num-linear_num-slab3_num-slab5_num-slab7_num):
        X = torch.from_numpy(np.random.uniform(left_bound, right_bound, combined.shape[0]))
        combined = torch.cat([combined, X.reshape(-1, 1)], dim = 1)
    
    idx = torch.randperm(combined.shape[0])
    combined = combined[idx].view(combined.size())
    Y = combined[:, 0].reshape(-1)
    X = combined[:, 1:]
    X_tr, Y_tr = X[:train_num,:], Y[:train_num]
    X_te, Y_te = X[train_num:,:], Y[train_num:]
    Y_te, Y_tr = map(lambda Z: Z.long(), [Y_te, Y_tr])
    X_te, X_tr = map(lambda Z: Z.float(), [X_te, X_tr])

    tr_dl = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)
    te_dl = DataLoader(TensorDataset(X_te, Y_te), batch_size=batch_size, shuffle=False)


    return {
        'X': X,
        'Y': Y,
        'train_dl': tr_dl,
        'test_dl': te_dl,
    }

#to decode the configuration dictionary
def get_data(**c):
    data = self_implemented_synthetic_data(c['train_num'], c['test_num'], c['dim_num'], c['linear_num'], c['slab3_num'], c['slab5_num'], c['slab7_num'],
                                           c['linear_margin'], c['slab3_margin'], c['slab5_margin'], c['slab7_margin'], c['width'], c['center'], c['linear_p'], c['attack_p'], c['batch_size'])
    return data

#Randomize certain dimension of the synthetic dataset
def randomized_loader(dl, r_list):
    X, Y = map(copy.deepcopy, dl.dataset.tensors)
    for i, val in enumerate(r_list):
        X[:, val] = torch.from_numpy(np.random.uniform(torch.min(X[:, val]), torch.max(X[:, val]), X.shape[0]))
    return DataLoader(TensorDataset(X, Y), batch_size=dl.batch_size, shuffle=True)
