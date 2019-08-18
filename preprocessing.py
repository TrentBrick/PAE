# This file is part of the OpenProtein project.
#
# @author Jeppe Hallgren
#
# For license information, please see the LICENSE file in the root directory.

import glob
import os.path
import os
import numpy as np
import h5py
from util import AA_ID_DICT, calculate_dihedral_angles, protein_id_to_str, get_structure_from_angles, \
    structure_to_backbone_atoms, write_to_pdb, calculate_dihedral_angles_over_minibatch, \
    get_backbone_positions_from_angular_prediction, encode_primary_string
import torch

MAX_SEQUENCE_LENGTH = 750

def process_raw_data(device, force_pre_processing_overwrite=True, want_trimmed=False, want_pure=False):
    print("Starting pre-processing of raw data...")
    input_files = glob.glob("data/raw/*")
    #input_files = [input_files[0]]
    input_files_filtered = filter_input_files(input_files)
    for file_path in input_files_filtered:
        filename = file_path.split('/')[-1]
        print('file name is:', filename)
        if want_trimmed:
            preprocessed_file_name = "data/preprocessed/"+filename+"_trimmed" +".hdf5"
        elif want_pure: 
            preprocessed_file_name = "data/preprocessed/"+filename+"_pure" +".hdf5"
        else:
            preprocessed_file_name = "data/preprocessed/"+filename+".hdf5"

        # check if we should remove the any previously processed files
        if os.path.isfile(preprocessed_file_name):
            print("Preprocessed file for " + filename + " already exists.")
            if force_pre_processing_overwrite:
                print("force_pre_processing_overwrite flag set to True, overwriting old file...")
                os.remove(preprocessed_file_name)
            else:
                print("Skipping pre-processing for this file...")

        if not os.path.isfile(preprocessed_file_name):
            process_file(filename, preprocessed_file_name, device, want_trimmed, want_pure)
    print("Completed pre-processing.")

def read_protein_from_file(file_pointer):

        dict_ = {}
        _dssp_dict = {'L': 0, 'H': 1, 'B': 2, 'E': 3, 'G': 4, 'I': 5, 'T': 6, 'S': 7}
        _mask_dict = {'-': 0, '+': 1}

        while True:
            next_line = file_pointer.readline()
            if next_line == '[ID]\n':
                id_ = file_pointer.readline()[:-1]
                dict_.update({'id': id_})
            elif next_line == '[PRIMARY]\n':
                primary = encode_primary_string(file_pointer.readline()[:-1])
                dict_.update({'primary': primary})
            elif next_line == '[EVOLUTIONARY]\n':
                evolutionary = []
                for residue in range(21): evolutionary.append(
                    [float(step) for step in file_pointer.readline().split()])
                dict_.update({'evolutionary': evolutionary})
            elif next_line == '[SECONDARY]\n':
                secondary = list([_dssp_dict[dssp] for dssp in file_pointer.readline()[:-1]])
                dict_.update({'secondary': secondary})
            elif next_line == '[TERTIARY]\n':
                tertiary = []
                # 3 dimension
                for axis in range(3): tertiary.append(
                    [float(coord) for coord in file_pointer.readline().split()])
                dict_.update({'tertiary': tertiary})
            elif next_line == '[MASK]\n':
                # IS THE LAST AMINO ACID TRIMMED FROM EVERY PROTEIN? Not it gets rid of the newline char 
                mask = list([_mask_dict[aa] for aa in file_pointer.readline()[:-1]])
                dict_.update({'mask': mask})
            elif next_line == '\n':
                return dict_
            elif next_line == '':
                return None


def process_file(input_file, output_file, device, want_trimmed, want_pure):
    print("Processing raw data file", input_file)

    # create output file
    f = h5py.File(output_file, 'w')
    current_buffer_size = 1
    current_buffer_allocation = 0
    dset1 = f.create_dataset('primary',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='int32')
    dset2 = f.create_dataset('tertiary',(current_buffer_size,MAX_SEQUENCE_LENGTH,9),maxshape=(None,MAX_SEQUENCE_LENGTH, 9),dtype='float')
    dset3 = f.create_dataset('mask',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='uint8')
    dset4 = f.create_dataset('padding_mask',(current_buffer_size,MAX_SEQUENCE_LENGTH),maxshape=(None,MAX_SEQUENCE_LENGTH),dtype='uint8')
    input_file_pointer = open("data/raw/" + input_file, "r")

    while True:
        # while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein is None:
            break
        
        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length too long:", sequence_length)
            continue

        if want_pure: 
            unpadded_mask = torch.Tensor(next_protein['mask']).type(dtype=torch.uint8)
            if unpadded_mask.sum() != unpadded_mask.shape[0]:
                print('dropping protein, mask has holes')
                continue

        elif want_trimmed:
            s = [str(i) for i in next_protein['mask']]
            s.append('0') # needed for those that dont end with a mask spot!. 
            res = "".join(s)
            if len(res.split('10')) > 2:
                print('dropping protein, mask isnt just on edges')
                continue

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((9, MAX_SEQUENCE_LENGTH))
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)

        # masking and padding here happens so that the stored dataset is of the same size. 
        # when the data is loaded in this padding is removed again. 
        primary_padded[:sequence_length] = next_protein['primary']
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        t_reshaped = np.reshape(t_transposed, (sequence_length,9)).T

        tertiary_padded[:,:sequence_length] = t_reshaped
        mask_padded[:sequence_length] = next_protein['mask']
        mask = torch.Tensor(mask_padded).type(dtype=torch.uint8)
        
        prim = torch.masked_select(torch.Tensor(primary_padded).type(dtype=torch.long), mask)
        pos = torch.masked_select(torch.Tensor(tertiary_padded), mask).view(9, -1).transpose(0, 1).unsqueeze(1) / 100

        pos = pos.to(device)

        angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos, [len(prim)], device)
        # this must be what is creating the nans!! Not clear to me why...
        tertiary, _ = get_backbone_positions_from_angular_prediction(angles, batch_sizes, device)
        tertiary = tertiary.squeeze(1)

        if torch.isnan(tertiary).sum() >0:
            print('there is a nan in tertiary! Dropping and printing mask')
            print(next_protein['mask'])
            continue

        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

        length_after_mask_removed = len(prim)

        if sequence_length == 0: 
            print('sequence length is zero after mask was applied. Dropping! ==========')
            continue

        #print('final size', length_after_mask_removed)
        #print('tertiary ', tertiary)

        primary_padded[:length_after_mask_removed] = prim.data.cpu().numpy()
        tertiary_padded[:length_after_mask_removed, :] = tertiary.data.cpu().numpy()
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        padding_mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        # this mask has masking for both the padding and the AAs without angle data!
        # # THIS HAS BECOME COMPLETELY IRRELEVANT NOW THAT I AM GETTING RID OF ANY MISSING THINGS FIRST! 
        padding_mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size,MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset4.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))

        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        dset4[current_buffer_allocation] = padding_mask_padded
        current_buffer_allocation += 1

    print("Wrote output to", current_buffer_allocation, "proteins to", output_file)


def filter_input_files(input_files):
    disallowed_file_endings = (".gitignore", ".DS_Store")
    return list(filter(lambda x: not x.endswith(disallowed_file_endings), input_files))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

process_raw_data(device, force_pre_processing_overwrite=True, want_trimmed=False, want_pure=True)

'''
# while there's more proteins to process
        next_protein = read_protein_from_file(input_file_pointer)
        if next_protein is None:
            break
        if current_buffer_allocation >= current_buffer_size:
            current_buffer_size = current_buffer_size + 1
            dset1.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset2.resize((current_buffer_size,MAX_SEQUENCE_LENGTH, 9))
            dset3.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))
            dset4.resize((current_buffer_size,MAX_SEQUENCE_LENGTH))

        sequence_length = len(next_protein['primary'])

        if sequence_length > MAX_SEQUENCE_LENGTH:
            print("Dropping protein as length too long:", sequence_length)
            continue

        # masking and padding here happens so that the stored dataset is of the same size. 
        # when the data is loaded in this padding is removed again.
        # 
        # # I dont want to have the masking applied here. Only in the loss function!! 
        # # I also dont get why padding was added BEFORE calculating the angles? 
        # # would make a lot more sense to only add it afterwards.  
        
        mask = next_protein['mask']
        mask = torch.Tensor(mask).type(dtype=torch.uint8)

        #                           does it go from a 1 to a 0 more than once?? 
        # or mask.split('01').shape>1
        if mask.sum() == mask.shape[0] :
            print('dropping protein, mask is present')
            continue

        primary = next_protein['primary']
        print('org shape', np.array(next_protein['tertiary']).shape)
        t_transposed = np.ravel(np.array(next_protein['tertiary']).T)
        print('trans', t_transposed.shape)
        t_reshaped = np.reshape(t_transposed, (sequence_length,9)).T
        tertiary = t_reshaped
        prim = torch.Tensor(primary).type(dtype=torch.long)

        print('tertiary shape', tertiary.shape)

        pos = torch.Tensor(tertiary).view(9, -1).transpose(0, 1).unsqueeze(1) / 100
        print('pos shape', tertiary.shape)
        pos = pos.to(device)

        # why am I getting nans here? 
        #print('before angles', pos)
        angles, batch_sizes = calculate_dihedral_angles_over_minibatch(pos, [len(prim)], device)
        #print('angles', angles)
        tertiary, _ = get_backbone_positions_from_angular_prediction(angles, batch_sizes, device)
        tertiary = tertiary.squeeze(1)
        #print('backbone positions again',tertiary)
        primary_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        tertiary_padded = np.zeros((MAX_SEQUENCE_LENGTH, 9))

        protein_length = len(prim)

        primary_padded[:protein_length] = prim.data.cpu().numpy()
        tertiary_padded[:protein_length, :] = tertiary.data.cpu().numpy()
        mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        # this mask has masking for both the padding and the AAs without angle data! 
        mask_padded[:protein_length] = mask.data.cpu().numpy()
        #mask_padded[:length_after_mask_removed] = np.ones(length_after_mask_removed)

        padding_mask_padded = np.zeros(MAX_SEQUENCE_LENGTH)
        # this mask has masking for both the padding and the AAs without angle data! 
        padding_mask_padded[:protein_length] = np.ones(protein_length)

        dset1[current_buffer_allocation] = primary_padded
        dset2[current_buffer_allocation] = tertiary_padded
        dset3[current_buffer_allocation] = mask_padded
        dset4[current_buffer_allocation] = padding_mask_padded
        current_buffer_allocation += 1
'''