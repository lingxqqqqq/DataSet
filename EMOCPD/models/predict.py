import torch
import os
import numpy as np
from model import EMOCPD
from data_utils import prepareBox
from generate_full_sidechain_box_20A import grab_PDB_csv
from calc_metrics import get_top_k_result
from atom_res_dict import label_res_dict, abrev

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

if __name__ == '__main__':
    model_path = '/data_for_msa/scripts/models/model_save/EMOCPD/best6225model.pth.tar'
    # protein_file = "/data_for_msa/case/PET/csv/6ij6.csv"
    protein_file = "/data_for_msa/case/Thaumatin/csv/1rqw.csv"
    out_file = "/data_for_msa/case/Thaumatin/out/1rqw.txt"
    sites = [121, 140, 186, 224, 233, 280]
    chain = 'A'
    k = 5

    model = EMOCPD()
    check = torch.load(model_path)
    model.load_state_dict(check['state_dict'])
    model.eval()

    PROTEIN = grab_PDB_csv(protein_file)
    positions = None
    if len(sites) > 0:
        positions = [(chain, str(site)) for site in sites]

    boxes, labels, valid_aa, _ = prepareBox(PROTEIN, atom_density=0.01, v=False, check=False, positions=positions)
    input = torch.tensor(np.vstack(boxes), dtype=torch.float32)
    output = model(input)
    output = torch.nn.Softmax(dim=1)(output)

    values, indices = get_top_k_result(output.data, k=k, sorted=False)

    for i, site in enumerate(sites):
        candidate_aa = [abrev[label_res_dict[int(j)]] for j in indices[i]]
        candidate_aa.reverse()
        print(abrev[label_res_dict[labels[i]]] + ":" + str(site) + ":", candidate_aa)

    # p_predict, predicted = torch.max(output.data, 1)
    # orgin_seq = [abrev[label_res_dict[i]] for i in labels]
    # design_seq = [abrev[label_res_dict[i]] for i in predicted.tolist()]
    # mutations = []
    # for i, aa_id in enumerate(valid_aa):
    #     if orgin_seq[i] != design_seq[i]:
    #         mutations.append(orgin_seq[i] + str(aa_id[1]) + design_seq[i])
    #
    # print(mutations)







