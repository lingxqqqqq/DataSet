import torch
import numpy as np
import os
from model import EMOCPD
from tqdm import tqdm
import torch.utils.data as Data
from datasets import ValDataset
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    test_data_dir = '/home/lingxq/datasets/data_for_CPD/TS50_data'
    out_path = '/home/lingxq/datasets/data_for_CPD/TS50_out'

    data_files = os.listdir(test_data_dir)
    model_path = '/home/lingxq/programs/EMOCPD/models/model_save/EMOCPD/best_6071model.pth.tar'
    # model = MutComputeX()
    model = EMOCPD()
    # model = VitCPD()
    check = torch.load(model_path)
    model.load_state_dict(check['state_dict'])
    model.to(device)
    model.eval()

    correct = 0
    total = 0
    y_true = []
    y_predict = []

    for test_data_file in data_files:
        test_data_set = ValDataset(os.path.join(test_data_dir, test_data_file))

        test_loader = Data.DataLoader(
            # 从数据库中每次抽出batch size个样本
            dataset=test_data_set,
            batch_size=50,
            pin_memory=True,
            shuffle=False,
            num_workers=5,
        )

        for x, y in tqdm(test_loader):
            x, y = x.to(device), y.to(device)
            y = y.long()
            outputs_ = model(x.float())
            _, predicted_ = torch.max(outputs_.data, 1)
            total += y.size(0)
            correct += (predicted_ == y).sum().item()
            y_predict.extend(outputs_.data.tolist())
            y_true.extend(y.data.tolist())

        del test_data_set, test_loader
        gc.collect()

    print("acc = ", correct / total)

    y_predict = np.array(y_predict)
    y_true = np.array(y_true)

    np.save(os.path.join(out_path, "p_values_emo1.npy"), y_predict)
    np.save(os.path.join(out_path, "y_true_emo1.npy"), y_true)
