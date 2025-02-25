import os
import csv
from tqdm import tqdm
from atom_res_dict import *
from generate_full_sidechain_box_20A import get_position_dict, pts_to_Xsmooth, grab_PDB_csv


def seperate_chains(_file, _target_path):
    pdb_id = _file[-8:-4]
    # print(pdb_id)
    with open(_file, 'r') as _f:
        line = " "
        chain_id = ""
        while line:
            line = _f.readline()
            if line.startswith('ATOM'):
                chain_now = line.split()[4][0]
                if chain_id != chain_now:
                    if chain_id == "":
                        _t = open(os.path.join(_target_path, pdb_id + chain_now + ".pdb"), 'w')
                    else:
                        _t.close()
                    chain_id = chain_now
                    _t = open(os.path.join(_target_path, pdb_id + chain_now + ".pdb"), 'w')

                _t.writelines(line)
        _t.close()


def get_data_from_file(_file, start=0):
    _data = []
    with open(_file, "r") as _f:
        line = " "
        while line:
            line = _f.readline()
            if line.startswith('ATOM'):
                if start != -1:
                    data_row = line.split()[start:]
                else:
                    data_row = line.split()[start]
                _data.append(data_row)

    return _data


# def get_data_from_file_1(_file, start=0):
#     _data = []
#     with open(_file, "r") as _f:
#         line = " "
#         while line:
#             line = _f.readline()
#             if line.startswith('ATOM'):
#                 if start != -1:
#                     data_row = line.split()[start:]
#                 else:
#                     data_row = line.split()[start]
#                 _data.append(data_row)
#
#     return _data


# protein_path = "/data_for_msa/testset/TS50/"


def process_pdb2csv(_target_path, _output_path):
    # file_list = os.listdir(protein_path)
    file_list1 = os.listdir(_target_path)

    # for file in file_list:
    #     if file.endswith(".csv"):
    #         protein_file = os.path.join(protein_path, file)
    #         os.remove(protein_file)


    error_file = []
    head = ["atom", "serial", "name", "resname", "resid", "x", "y", "z", "charges",	"radius", "sasa"]
    for file in file_list1:
        if not file.endswith(".pdb"):
            continue
        pdb_chain_id = file[:-4]
        csv_file = os.path.join(_target_path, "csv", pdb_chain_id + ".csv")
        if os.path.exists(csv_file):
            continue
        # print(pdb_chain_id)
        pqr_output_file = os.path.join(_output_path, pdb_chain_id + "_pqr.pdb")
        protein_file = os.path.join(_target_path, file)
        os.system('/opt/anaconda3/envs/EMOCPD/bin/pdb2pqr --ff CHARMM ' + protein_file + " " + pqr_output_file)
        if os.path.exists(pqr_output_file):
            freesasa_output_file = os.path.join(_output_path, pdb_chain_id + "_sasa.pdb")
            os.system('freesasa --format=pdb --depth=atom --output=' + freesasa_output_file + " " + pqr_output_file)
        else:
            error_file.append(pdb_chain_id)
            print("pdb2pqr fails on" + pdb_chain_id)
            continue
        # pqr_f = open(pqr_output_file, "r")
        base_rows = get_data_from_file(pqr_output_file, start=0)
        sasa_list = get_data_from_file(freesasa_output_file, start=-1)
        with open(csv_file, 'a', newline='') as csv_f:
            writer = csv.writer(csv_f)
            writer.writerow(head)
            if len(base_rows) == len(sasa_list):
                for i, row in enumerate(base_rows):
                    row.append(sasa_list[i])
                    writer.writerow(row)
            else:
                error_file.append(pdb_chain_id)
    print("error files:")
    print(error_file)
    return csv_file


def prepareBox(PROTEIN, channels=7, atom_density=0.01, box_size=20, pixel_size=1, v=True, check=True, positions=None):
    '''
        PROTEIN:蛋白质数据文件路径
        channels:通道数，默认为7维（C、N、O、S、H、partial_charge、sasa）
        atom_density: 盒子里的原子密度，控制一个盒子中最少有多少原子, num_of_atom / box volume
        box_size: 盒子尺寸
        pixel_size: 分辨率，默认为1埃
    '''
    ID_dict, _, _, _, _, _, _ = PROTEIN

    boxes = []
    labels = []
    valid_aa = []
    skip = []
    keys = list(ID_dict.keys())

    if positions is not None:
        keys = positions

    for chain_ID in keys:
        res_atoms = ID_dict[chain_ID]
        res = res_atoms[0].res
        if res in list(res_label_dict.keys()):
            label = res_label_dict[res]
            get_position = get_position_dict(res_atoms)
            if "CA" in get_position.keys():
                ctr = get_position["CA"]
                pts = [ctr, chain_ID, label]
                X_smooth, label, _, _, _, valid_box = pts_to_Xsmooth(PROTEIN, pts, atom_density,
                                                                     channels, pixel_size, box_size, check=check)
                if valid_box:
                    boxes.append([X_smooth])
                    labels.append(label)
                    valid_aa.append(chain_ID)
                else:
                    if v:
                        print(f'{chain_ID[0]}链的第{int(float(chain_ID[1]))}号氨基酸采样的原子过少，被忽略')
                    skip.append(chain_ID)
            else:
                if v:
                    print(f'{chain_ID[0]}链的第{int(float(chain_ID[1]))}号氨基酸没有CA原子，被忽略')
                skip.append(chain_ID)
    return boxes, labels, valid_aa, skip



