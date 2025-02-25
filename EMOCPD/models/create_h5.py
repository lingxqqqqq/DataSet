'''
create a demo for training 7 channel cnn model
'''
import tables as tb
import h5py
import random
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold
from tqdm import tqdm

res_label_dict = {'HIS': 0, 'LYS': 1, 'ARG': 2, 'ASP': 3, 'GLU': 4, 'SER': 5, 'THR': 6, 'ASN': 7, 'GLN': 8, 'ALA': 9,
                  'VAL': 10, 'LEU': 11, 'ILE': 12, 'MET': 13, 'PHE': 14, 'TYR': 15, 'TRP': 16, 'PRO': 17, 'GLY': 18,
                  'CYS': 19}

label_res_dict = {0: 'HIS', 1: 'LYS', 2: 'ARG', 3: 'ASP', 4: 'GLU', 5: 'SER', 6: 'THR', 7: 'ASN', 8: 'GLN', 9: 'ALA',
                  10: 'VAL', 11: 'LEU', 12: 'ILE', 13: 'MET', 14: 'PHE', 15: 'TYR', 16: 'TRP', 17: 'PRO', 18: 'GLY',
                  19: 'CYS'}

data_root = '/data_for_msa/raw_data/train_new'
files = os.listdir(data_root)
# print(len(files))
random.shuffle(files)
label = [res_label_dict[file[:3]] for file in files]
kfold = StratifiedKFold(n_splits=20)
k_fold_indexes = []
for _, test_index in kfold.split(files, label):
    # labels_i = [labels[i] for i in test_index]
    # print(labels_i)
    k_fold_indexes.append(test_index)
    print([label[i] for i in test_index])

exit(111)
for k, index in enumerate(k_fold_indexes):
    datas = []
    labels = []
    print('fold: ', k)
    f = open('/data_for_msa/train_data/' + 'fold_' + str(k) + '.txt', 'w')
    for i in tqdm(index):
        f.write(files[i] + '\n')
        dat_file = os.path.join(data_root, files[i])
        data_np = np.load(dat_file)['arr']
        label_np = np.ones(data_np.shape[0]) * label[i]
        datas.append(data_np)
        labels.append(label_np)
    f.close()
    datas = np.vstack(datas)
    labels = np.hstack(labels)
    datas_saved_path = os.path.join('/data_for_msa/train_data', 'data_' + str(k) + '.npz')
    np.savez_compressed(datas_saved_path, datas)

    labels_saved_path = os.path.join('/data_for_msa/train_data', 'label_' + str(k) + '.npz')
    np.savez_compressed(labels_saved_path, labels)


