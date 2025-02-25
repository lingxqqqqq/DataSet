import csv
import torch
import torch.nn as nn
import torch.optim as optim
from model import EMOCPD
import numpy as np
from datasets import NpzDataset
import torch.utils.data as Data
import os
from torch.optim import lr_scheduler
from tqdm import tqdm
import gc


def test_model(loader_, model_, device_):
    print("testing...")
    model_.eval()
    with torch.no_grad():
        correct_ = 0
        total_ = 0
        for data_ in tqdm(loader_):
            inputs_, labels_ = data_
            inputs_, labels_ = inputs_.to(device_), labels_.to(device_)
            labels_ = labels_.long()
            outputs_ = model_(inputs_.float())
            _, predicted_ = torch.max(outputs_.data, 1)
            total_ += labels_.size(0)
            correct_ += (predicted_ == labels_).sum().item()
    return 100 * correct_ / total_


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 150
    # 实例化模型和损失函数、优化器
    model = EMOCPD()
    model.to(device)
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5, weight_decay=0.001)

    epochs = 8
    iteration = 0
    # data_root = '/data_for_msa/mutecompute/train_data/'

    test_data_file = '/home/lingxq/datasets/data_for_CPD/raw_data/test_data/data.npz'
    test_label_file = '/home/lingxq/datasets/data_for_CPD/raw_data/test_data/label.npz'

    test_data_set = NpzDataset(test_data_file, test_label_file)

    test_loader = Data.DataLoader(
        # 从数据库中每次抽出batch size个样本
        dataset=test_data_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
    )

    best_acc = 0
    epoch = 0

    while epoch < epochs:
        print(f'training epoch {epoch + 1}' + '\n')
        k = 0
        while k < 10:

            running_loss = 0.0
            correct = 0
            total = 0
            patience = 0

            data_file = '/home/lingxq/datasets/data_for_CPD/raw_data/train_data_10/data_' + str(k) + '.npz'
            label_file = '/home/lingxq/datasets/data_for_CPD/raw_data/train_data_10/label_' + str(k) + '.npz'

            data_set = NpzDataset(data_file, label_file)

            loader = Data.DataLoader(
                dataset=data_set,
                batch_size=batch_size,
                shuffle=True,
                num_workers=5,
            )
            for data in tqdm(loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.long()

                optimizer.zero_grad()
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                iteration += 1
                if iteration % 100 == 0:
                    acc_temp = test_model(test_loader, model, device)
                    print("test acc: ", acc_temp)
                    with open('./model_save/EMOCPD/train_logs.csv', 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([iteration + 1, running_loss / total, correct / total, acc_temp])

                    if acc_temp > best_acc:
                        best_acc = acc_temp
                        state = {
                            'epoch': epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'best_acc': best_acc,
                            'k': k
                        }
                        torch.save(state, './model_save/EMOCPD/best_' + str(int(best_acc * 100)) + 'model.pth.tar')
                        print('model saved' + ", acc = " + str(best_acc))
            del loader, data_set
            gc.collect()
            k += 1
        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_acc': best_acc,
            'k': k
        }
        torch.save(state, './model_save/EMOCPD/train_epoch_' + str(int(epoch + 1)) + '_model.pth.tar')
        epoch += 1
