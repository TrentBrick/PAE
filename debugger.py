from util import *
import h5py
training_file = "data/preprocessed/sample.txt.hdf5"
print(training_file)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = contruct_dataloader_from_disk(training_file, 32)
for x in train_loader:
    seqs, coords, mask = x
    print(coords)
    break
'''f=h5py.File(training_file, 'r')
#masks = torch.Tensor(f['mask'])

count=0
print(f['mask'].shape)
for index in range(0,10000): #range(f['mask'].shape[0]-50000): 
    padding_mask = torch.Tensor(f['padding_mask'][index,:]).type(dtype=torch.uint8)
    m = torch.masked_select(torch.Tensor(f['mask'][index,:]).type(dtype=torch.uint8), padding_mask)  
    #print(m.sum())
    #print(m.shape)
    if m.sum() == m.shape[0]:
        count+=1
print(count)'''

    #mask = torch.Tensor(mask).to(device)

''' 
What the sample dihedral angles should be looking like! 
tensor([[[ 0.0000,  0.0000,  0.0000,  ..., -0.1526,  2.4556,  0.0000],
         [ 0.3437,  3.5661,  0.5346,  ..., -0.8005,  5.3487, -0.7171],
         [-1.9823,  5.9469, -0.8217,  ..., -2.8805,  8.0277, -1.7792],
         ...,
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.0000,  0.0000,  0.0000,  ...,  0.0000,  0.0000,  0.0000]]])'''