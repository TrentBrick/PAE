# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import torch
import torch.utils.data
import torch.nn.utils.rnn as rnn_utils
import h5py
import PeptideBuilder
import Bio.PDB
import math
from datetime import datetime
import numpy as np
import pnerf.pnerf as pnerf
import platform
#from TorchProteinLibrary import RMSD

from torch.utils.data import Sampler, Dataset
from collections import OrderedDict
from random import shuffle

# have to set something for the padding value of 0 in case it is predicted and plotted!! 
# I ALSO HAVE TO MODIFY THIS INSIDE OF protein_id_to_dict AS IT MAKES AN INVERSE
# DICTIONARY AND CANT HAVE A DIFFERENT VALUE FOR 
'''AA_ID_DICT = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9,
              'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17,
              'V': 18, 'W': 19,'Y': 20}'''

#Where padding cannot be predicted!! 
AA_ID_DICT = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8,
'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16,
'V': 17, 'W': 18,'Y': 19}

def contruct_dataloader_from_disk(filename, minibatch_size):
    # may want to pre generate ind_n_len with another function and then feed this into the BatchSampler direct
    ind_n_len = H5PytorchDataset(filename).sequences()   
    bucket_batch_sampler = BucketBatchSampler(ind_n_len, minibatch_size) # <-- does not store X
    if platform.system() is 'Windows':
        num_workers = 0
    else:
        num_workers = 8
    return torch.utils.data.DataLoader(H5PytorchDataset(filename), batch_size=1, batch_sampler= bucket_batch_sampler,
                                       shuffle=False, num_workers=num_workers, 
                                       collate_fn=H5PytorchDataset.sort_samples_in_minibatch,
                                       drop_last=False)

class H5PytorchDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        super(H5PytorchDataset, self).__init__()

        self.h5pyfile = h5py.File(filename, 'r')
        self.num_proteins, self.max_sequence_len = self.h5pyfile['primary'].shape
        print('number of proteins in this dataset', filename, self.num_proteins, self.max_sequence_len)

    def __getitem__(self, index):
        # this mask gets rid of the padding that is added during the preprocessing step.
        padding_mask = torch.Tensor(self.h5pyfile['padding_mask'][index,:]).type(dtype=torch.uint8) 
        # below mask has the actual amino acids without good angle data. 
        mask = torch.masked_select(torch.Tensor(self.h5pyfile['mask'][index,:]).type(dtype=torch.uint8), padding_mask)
        # I need to apply the mask select here but only to the end padding!
        prim = torch.masked_select(torch.Tensor(self.h5pyfile['primary'][index,:]).type(dtype=torch.long), padding_mask)
        tertiary = torch.Tensor(self.h5pyfile['tertiary'][index][:int(padding_mask.sum())]) # max length x 9
        '''if torch.isnan(tertiary).sum() >0:
            print('there is a nan in dataloader!!!', tertiary)
            print('the mask', mask)'''
        return  prim, tertiary, mask

    def __len__(self):
        return self.num_proteins

    def sequences(self):
        padding_mask = torch.Tensor(self.h5pyfile['padding_mask']).type(dtype=torch.uint8) 
        lens = padding_mask.sum(dim=1)
        ind_n_len = []
        for i, p in enumerate(lens):
            ind_n_len.append((i, p.item()))
        return ind_n_len
       
    def sort_samples_in_minibatch(samples):
        samples_list = []
        for s in samples:
            samples_list.append(s)
        # sort according to length of aa sequence
        samples_list.sort(key=lambda x: len(x[0]), reverse=True)
        return zip(*samples_list)

class BucketBatchSampler(Sampler):
    # want inputs to be an array
    def __init__(self, ind_n_len, batch_size):
        self.batch_size = batch_size
        self.ind_n_len = ind_n_len
        self.batch_list = self._generate_batch_map()
        self.num_batches = len(self.batch_list)

    def _generate_batch_map(self, equal_length=False):
        # shuffle all of the indices first so they are put into buckets differently
        shuffle(self.ind_n_len)
        # Organize lengths, e.g., batch_map[10] = [30, 124, 203, ...] <= indices of sequences of length 10
        batch_map = OrderedDict()
        for idx, length in self.ind_n_len:
            if length not in batch_map:
                batch_map[length] = [idx]
            else:
                batch_map[length].append(idx)
        # Use batch_map to split indices into batches of equal size
        # e.g., for batch_size=3, batch_list = [[23,45,47], [49,50,62], [63,65,66], ...]
        if equal_length:
            batch_list = []
            for length, indices in batch_map.items():
                for group in [indices[i:(i+self.batch_size)] for i in range(0, len(indices), self.batch_size)]:
                    batch_list.append(group)
        else: # mimic torchtexts' BucketIterator
            indices = []
            [ indices.extend(v) for v in batch_map.values() ]
            batch_list = [indices[i:(i+self.batch_size)] for i in range(0, len(indices), self.batch_size)]

        return batch_list

    def batch_count(self):
        return self.num_batches

    def __len__(self):
        return len(self.ind_n_len)

    def __iter__(self):
        self.batch_list = self._generate_batch_map()
        # shuffle all the batches so they arent ordered by bucket size
        shuffle(self.batch_list)
        for i in self.batch_list:
            yield i

def set_experiment_id(save_name, learning_rate, minibatch_size, store_date_time_etc=False):
    if store_date_time_etc:
        output_string = datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        output_string += "-" + save_name
        output_string += "LR" + str(learning_rate).replace(".","_")
        output_string += "-MB" + str(minibatch_size)
    else:
        output_string = save_name
    globals().__setitem__("experiment_id",output_string)
    return output_string

def write_out(*args, end='\n'):
    output_string = datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": " + str.join(" ", [str(a) for a in args]) + end
    if globals().get("experiment_id") is not None:
        with open("output/"+globals().get("experiment_id")+".txt", "a+") as output_file:
            output_file.write(output_string)
            output_file.flush()
    print(output_string, end="")

'''def evaluate_model(data_loader, model):
    loss = 0
    data_total = []
    dRMSD_list = []
    RMSD_list = []
    for i, data in enumerate(data_loader, 0):
        primary_sequence, tertiary_positions, mask = data
        start = time.time()
        predicted_angles, backbone_atoms, batch_sizes = model(primary_sequence)
        write_out("Apply model to validation minibatch:", time.time() - start)
        cpu_predicted_angles = predicted_angles.transpose(0,1).cpu().detach()
        cpu_predicted_backbone_atoms = backbone_atoms.transpose(0,1).cpu().detach()
        minibatch_data = list(zip(primary_sequence,
                                  tertiary_positions,
                                  cpu_predicted_angles,
                                  cpu_predicted_backbone_atoms))
        data_total.extend(minibatch_data)
        start = time.time()
        for primary_sequence, tertiary_positions,predicted_pos, predicted_backbone_atoms in minibatch_data:
            actual_coords = tertiary_positions.transpose(0,1).contiguous().view(-1,3)
            predicted_coords = predicted_backbone_atoms[:len(primary_sequence)].transpose(0,1).contiguous().view(-1,3).detach()
            rmsd = calc_rmsd(predicted_coords, actual_coords)
            drmsd = calc_drmsd(predicted_coords, actual_coords)
            RMSD_list.append(rmsd)
            dRMSD_list.append(drmsd)
            error = rmsd
            loss += error
            end = time.time()
        write_out("Calculate validation loss for minibatch took:", end - start)
    loss /= data_loader.dataset.__len__()
    return (loss, data_total, float(torch.Tensor(RMSD_list).mean()), float(torch.Tensor(dRMSD_list).mean()))
'''
'''def write_model_to_disk(model):
    path = "output/models/"+globals().get("experiment_id")+".model"
    torch.save(model,path)
    return path'''

def draw_plot(fig, plt, validation_dataset_size, sample_num, train_loss_values,
              validation_loss_values):
    def draw_with_vars():
        ax = fig.gca()
        ax2 = ax.twinx()
        plt.grid(True)
        plt.title("Training progress (" + str(validation_dataset_size) + " samples in validation set)")
        train_loss_plot, = ax.plot(sample_num, train_loss_values)
        ax.set_ylabel('Train Negative log likelihood')
        ax.yaxis.labelpad = 0
        validation_loss_plot, = ax2.plot(sample_num, validation_loss_values, color='black')
        ax2.set_ylabel('Validation loss')
        ax2.set_ylim(bottom=0)
        plt.legend([train_loss_plot, validation_loss_plot],
                   ['Train loss on last batch', 'Validation loss'])
        ax.set_xlabel('Minibatches processed (=network updates)', color='black')
    return draw_with_vars

def draw_ramachandran_plot(fig, plt, phi, psi):
    def draw_with_vars():
        ax = fig.gca()
        plt.grid(True)
        plt.title("Ramachandran plot")
        train_loss_plot, = ax.plot(phi, psi)
        ax.set_ylabel('Psi')
        ax.yaxis.labelpad = 0
        plt.legend([train_loss_plot],
                   ['Phi psi'])
        ax.set_xlabel('Phi', color='black')
    return draw_with_vars

def write_result_summary(accuracy):
    output_string = globals().get("experiment_id") + ": " + str(accuracy) + "\n"
    with open("output/result_summary.txt", "a+") as output_file:
        output_file.write(output_string)
        output_file.flush()
    print(output_string, end="")

def calculate_dihedral_angles_over_minibatch(atomic_coords_padded, batch_sizes, device, is_padded=True):
    angles = []
    if is_padded:
        atomic_coords = atomic_coords_padded.transpose(0,1)
        for idx, _ in enumerate(batch_sizes): # WHY IS THERE A FOR LOOP HERE??
            angles.append(calculate_dihedral_angles(atomic_coords[idx][:batch_sizes[idx]], device))
        return torch.nn.utils.rnn.pad_packed_sequence(
            torch.nn.utils.rnn.pack_sequence(angles))
    else: 
        for atomic_c in atomic_coords_padded: # WHY IS THERE A FOR LOOP HERE??
            angles.append(calculate_dihedral_angles(atomic_c.to(device), device))
        return angles
    

def calculate_dihedral_angles(atomic_coords, device):
    assert int(atomic_coords.shape[1]) == 9
    atomic_coords = atomic_coords.contiguous().view(-1,3)

    zero_tensor = torch.tensor(0.0).to(device)

    dihedral_list = [zero_tensor,zero_tensor]
    dihedral_list.extend(compute_dihedral_list(atomic_coords))
    dihedral_list.append(zero_tensor)
    angles = torch.tensor(dihedral_list).view(-1,3)
    return angles

def compute_dihedral_list(atomic_coords):
    # atomic_coords is -1 x 3
    # https://math.stackexchange.com/questions/47059/how-do-i-calculate-a-dihedral-angle-given-cartesian-coordinates 
    ba = atomic_coords[1:] - atomic_coords[:-1]
    ba /= ba.norm(dim=1).unsqueeze(1)
    ba_neg = -1 * ba

    n1_vec = torch.cross(ba[:-2], ba_neg[1:-1], dim=1)
    n2_vec = torch.cross(ba_neg[1:-1], ba[2:], dim=1)
    n1_vec /= n1_vec.norm(dim=1).unsqueeze(1)
    n2_vec /= n2_vec.norm(dim=1).unsqueeze(1)

    m1_vec = torch.cross(n1_vec, ba_neg[1:-1], dim=1)

    x = torch.sum(n1_vec*n2_vec,dim=1)
    y = torch.sum(m1_vec*n2_vec,dim=1)

    return torch.atan2(y,x)

def get_structure_from_angles(aa_list_encoded, angles):
    aa_list = protein_id_to_str(aa_list_encoded)
    omega_list = angles[1:,0]
    phi_list = angles[1:,1]
    psi_list = angles[:-1,2]
    assert len(aa_list) == len(phi_list)+1 == len(psi_list)+1 == len(omega_list)+1
    structure = PeptideBuilder.make_structure(aa_list,
                                              list(map(lambda x: math.degrees(x), phi_list)),
                                              list(map(lambda x: math.degrees(x), psi_list)),
                                              list(map(lambda x: math.degrees(x), omega_list)))
    return structure

def write_to_pdb(structure, prot_id):
    out = Bio.PDB.PDBIO()
    out.set_structure(structure)
    out.save("output/protein_" + str(prot_id) + ".pdb")

def calc_pairwise_distances(chain_a, chain_b, device):
    distance_matrix = torch.Tensor(chain_a.size()[0], chain_b.size()[0]).type(torch.float).to(device)
    # add small epsilon to avoid boundary issues
    epsilon = 10 ** (-4) * torch.ones(chain_a.size(0), chain_b.size(0))
    epsilon = epsilon.to(device)

    for i, row in enumerate(chain_a.split(1)):
        distance_matrix[i] = torch.sum((row.expand_as(chain_b) - chain_b) ** 2, 1).view(1, -1)

    return torch.sqrt(distance_matrix + epsilon)

def calc_drmsd(chain_a, chain_b, device):
    assert len(chain_a) == len(chain_b)
    distance_matrix_a = calc_pairwise_distances(chain_a, chain_a, device)
    distance_matrix_b = calc_pairwise_distances(chain_b, chain_b, device)
    return torch.norm(distance_matrix_a - distance_matrix_b, 2) \
            / math.sqrt((len(chain_a) * (len(chain_a) - 1)))

'''def least_rmsd(src, ref, num_atoms):
    rmsd = RMSD.Coords2RMSD().cuda()
    return rmsd(src, ref, num_atoms)'''

# method for translating a point cloud to its center of mass
def transpose_atoms_to_center_of_mass(x):
    # calculate com by summing x, y and z respectively
    # and dividing by the number of points
    centerOfMass = np.matrix([[x[0, :].sum() / x.shape[1]],
                    [x[1, :].sum() / x.shape[1]],
                    [x[2, :].sum() / x.shape[1]]])
    # translate points to com and return
    return x - centerOfMass

def calc_rmsd(chain_a, chain_b):
    # move to center of mass
    a = chain_a.cpu().numpy().transpose()
    b = chain_b.cpu().numpy().transpose()
    X = transpose_atoms_to_center_of_mass(a)
    Y = transpose_atoms_to_center_of_mass(b)

    R = Y * X.transpose()
    # extract the singular values
    _, S, _ = np.linalg.svd(R)
    # compute RMSD using the formular
    E0 = sum(list(np.linalg.norm(x) ** 2 for x in X.transpose())
             + list(np.linalg.norm(x) ** 2 for x in Y.transpose()))
    TraceS = sum(S)
    RMSD = np.sqrt((1 / len(X.transpose())) * (E0 - 2 * TraceS))
    return RMSD

def calc_angular_difference(a1, a2):
    # MSE between the angles. Need to look into the min equation... 
    sum = torch.sqrt(torch.mean(
            torch.min(torch.abs(a1 - a2),
                      2 * math.pi - torch.abs(a2 - a1)
                      ) ** 2))
    '''a1 = a1.transpose(0,1).contiguous()
    a2 = a2.transpose(0,1).contiguous()
    sum = 0
    for idx, _ in enumerate(a1):
        assert a1[idx].shape[1] == 3
        assert a2[idx].shape[1] == 3
        a1_element = a1[idx].view(-1, 1)
        a2_element = a2[idx].view(-1, 1)
        sum += torch.sqrt(torch.mean(
            torch.min(torch.abs(a2_element - a1_element),
                      2 * math.pi - torch.abs(a2_element - a1_element)
                      ) ** 2))'''
    return sum #/ a1.shape[0]

def structures_to_backbone_atoms_padded(structures):
    backbone_atoms_list = []
    for structure in structures:
        backbone_atoms_list.append(structure_to_backbone_atoms(structure))
    backbone_atoms_padded, batch_sizes_backbone = torch.nn.utils.rnn.pad_packed_sequence(
        torch.nn.utils.rnn.pack_sequence(backbone_atoms_list))
    return backbone_atoms_padded, batch_sizes_backbone

def structure_to_backbone_atoms(structure):
    predicted_coords = []
    for res in structure.get_residues():
        predicted_coords.append(torch.Tensor(res["N"].get_coord()))
        predicted_coords.append(torch.Tensor(res["CA"].get_coord()))
        predicted_coords.append(torch.Tensor(res["C"].get_coord()))
    return torch.stack(predicted_coords).view(-1,9)

def get_backbone_positions_from_angular_prediction(angular_emissions, batch_sizes, device):
    # angular_emissions -1 x minibatch size x 3 (omega, phi, psi)
    points = pnerf.dihedral_to_point(angular_emissions, device)
    coordinates = pnerf.point_to_coordinate(points, device) / 100 # devide by 100 to angstrom unit
    return coordinates.transpose(0,1).contiguous().view(batch_sizes,-1,9).transpose(0,1), batch_sizes


'''def calc_avg_drmsd_over_minibatch(backbone_atoms_padded, actual_coords_padded, batch_sizes):
    backbone_atoms_list = list(
        [backbone_atoms_padded[:batch_sizes[i], i] for i in range(int(backbone_atoms_padded.size(1)))])
    actual_coords_list = list(
        [actual_coords_padded[:batch_sizes[i], i] for i in range(int(actual_coords_padded.size(1)))])
    drmsd_avg = 0
    for idx, backbone_atoms in enumerate(backbone_atoms_list):
        actual_coords = actual_coords_list[idx].transpose(0, 1).contiguous().view(-1, 3)
        drmsd_avg += calc_drmsd(backbone_atoms.transpose(0, 1).contiguous().view(-1, 3), actual_coords)
    return drmsd_avg / len(backbone_atoms_list)
'''
def encode_primary_string(primary):
    return list([AA_ID_DICT[aa] for aa in primary])

def protein_id_to_str(protein_id_list):
    _aa_dict_inverse = {v: k for k, v in AA_ID_DICT.items()}
    aa_list = []
    for a in protein_id_list:
        aa_symbol = _aa_dict_inverse[int(a)]
        aa_list.append(aa_symbol)
    return aa_list

import time
import math
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


'''def intial_pos_from_aa_string(batch_aa_string):
    structures = []
    for aa_string in batch_aa_string:
        structure = get_structure_from_angles(aa_string,
                                              np.repeat([-120], len(aa_string)-1),
                                              np.repeat([140], len(aa_string)-1),
                                              np.repeat([-370], len(aa_string)-1))
        structures.append(structure)
    return structures

def pass_messages(aa_features, message_transformation, device):
    # aa_features (#aa, #features) - each row represents the amino acid type (embedding) and the positions of the backbone atoms
    # message_transformation: (-1 * 2 * feature_size) -> (-1 * output message size)
    feature_size = aa_features.size(1)
    aa_count = aa_features.size(0)
    eye = torch.eye(aa_count,dtype=torch.uint8).view(-1).expand(2,feature_size,-1).transpose(1,2).transpose(0,1)
    eye_inverted = torch.ones(eye.size(),dtype=torch.uint8) - eye
    eye_inverted = eye_inverted.to(device)
    features_repeated = aa_features.repeat((aa_count,1)).view((aa_count,aa_count,feature_size))
    aa_messages = torch.stack((features_repeated.transpose(0,1), features_repeated)).transpose(0,1).transpose(1,2).view(-1,2,feature_size)
    aa_msg_pairs = torch.masked_select(aa_messages,eye_inverted).view(-1,2,feature_size) # (aa_count^2 - aa_count) x 2 x aa_features     (all pairs except for reflexive connections)
    transformed = message_transformation(aa_msg_pairs).view(aa_count, aa_count - 1, -1)
    transformed_sum = transformed.sum(dim=1) # aa_count x output message size
    return transformed_sum'''