import numpy as np
from sklearn.metrics import confusion_matrix


def get_top_k_result(logits, k=3, sorted=True):
    indices = np.argsort(logits, axis=-1)[:, -k:]  # 取概率最大的前K个所对应的预测标签
    if sorted:  # np.argsort 默认返回的顺序是从小到达，sorted=True可以返回从大到小
        tmp = []
        for item in indices:
            tmp.append(item[::-1])
        indices = np.array(tmp)
    values = []
    for idx, item in zip(indices, logits):  # 取所有预测值所对应的概率值
        p = item.reshape(1, -1)[:, idx].reshape(-1)
        values.append(p)
    values = np.array(values)
    return values, indices


def calculate_top_k_accuracy(logits, targets, k=2):
    values, indices = get_top_k_result(logits, k=k, sorted=False)
    y = np.reshape(targets, [-1, 1])
    correct = (y == indices) * 1.  # 对比预测的K个值中是否包含有正确标签中的结果
    top_k_accuracy = np.mean(correct) * k  # 计算最后的准确率
    return top_k_accuracy


if __name__ == '__main__':
    p_file = "/data_for_msa/mutecompute/testset/TS500/out_path/p_values_emo.npy"
    y_file = "/data_for_msa/mutecompute/testset/TS500/out_path/y_true_emo.npy"

    p = np.load(p_file)
    y = np.load(y_file)
    print(p.shape)
    print(y.shape)
    # top_k = []
    #
    # for i in range(1, 21):
    #     top_i = calculate_top_k_accuracy(p, y, i)
    #     top_k.append(top_i)
    #
    # print(top_k)
    y_p = np.argmax(p, 1)
    print(y_p[:10])
    print(y_p.shape)

    cm = confusion_matrix(y, y_p)

    print(cm)

    TP = np.diag(cm)  # 真正样本数
    FP = cm.sum(axis=0) - np.diag(cm)  # 假正样本数
    FN = cm.sum(axis=1) - np.diag(cm)  # 假负样本数

    PRECISION = TP / (TP + FP)  # 查准率，又名准确率
    RECALL = TP / (TP + FN)  # 查全率，又名召回率

    print(PRECISION)
    print(RECALL)
