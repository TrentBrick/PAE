from util import *
training_file = "data/preprocessed/testing.hdf5"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader = contruct_dataloader_from_disk(training_file, 32)
for x in train_loader:
    seqs, coords, mask = x
    print(len(mask))
    
    
    #mask = torch.Tensor(mask).to(device)