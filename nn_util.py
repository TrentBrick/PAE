import torch
import torch.nn.utils.rnn as rnn_utils
from util import calc_angular_difference, write_out
import numpy as np
from datetime import datetime
import time

def seq_and_angle_loss(pred_seqs, padded_seqs, pred_dihedrals, padded_dihedrals, mask, VOCAB_SIZE=21, use_mask=False):
    # Everything is padded here! 
    if not use_mask:
        mask = torch.ones(mask.shape).byte()
    # GET TUTORIAL FROM MEDIUM!!
    '''criterion = torch.nn.NLLLoss(size_average=True, ignore_index=-1)
    loss=  criterion(pred_seqs.permute([0,2,1]).contiguous(),padded_seqs.max(dim=2)[1])'''
    #get cross entropy just padding at the end!
    criterion = torch.nn.NLLLoss(size_average=True, ignore_index=0)
    seq_cross_ent_loss = criterion(pred_seqs.permute([0,2,1]).contiguous(),padded_seqs)
    # flatten all the labels
    padded_seqs =padded_seqs.flatten()
    pred_seqs = pred_seqs.view(-1, VOCAB_SIZE)
    seq_mask = (padded_seqs > 0).float() # this is just the padding at the end! 
    nb_tokens = int(torch.sum(seq_mask).item())
    #get accuracy
    top_preds = pred_seqs.max(dim=1)[1]
    seq_acc = torch.sum( torch.eq(top_preds,padded_seqs).type(torch.float)*seq_mask) / nb_tokens
    
    #loss for the angles, apply the mask to padding and uncertain coordinates!
    mask = mask.view(mask.shape[0],mask.shape[1], 1).expand(-1,-1,3)
    angular_loss = calc_angular_difference(torch.masked_select(pred_dihedrals, mask), 
            torch.masked_select(torch.Tensor(padded_dihedrals), mask))

    return seq_cross_ent_loss, seq_acc, angular_loss

def printParamNum(encoder_net, decoder_net):
    params=0
    for net in [encoder_net, decoder_net]:
        model_parameters = filter(lambda p: p.requires_grad, net.parameters())
        params += sum([np.prod(p.size()) for p in net.parameters()])
    print('total model parameters: ',params)

def saveModel(exp_id, encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, train_loss, eval_acc, e):
    for name, net, optim in zip(['encoder_save','decoder_save'],[encoder_net, decoder_net],[encoder_optimizer, decoder_optimizer]):
        path = "output/models/"+exp_id+".tar"
        torch.save({
                    'epoch': e,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                    'last_loss': train_loss,
                    'eval_accuracy':eval_acc
                    }, path)
    print('saveModel worked')
    return path

def embed(data, batch_sizes, device):
    
    # one-hot encoding
    start_compute_embed = time.time()
    prot_aa_list = data.unsqueeze(1)
    embed_tensor = torch.zeros(prot_aa_list.size(0), 21, prot_aa_list.size(2)).to(device) # 21 classes
    #prot_aa_list.to(device) #should already be embedded. 
    input_sequences = embed_tensor.scatter_(1, prot_aa_list.data, 1).transpose(1,2)
    end = time.time()
    write_out("Embed time:", end - start_compute_embed)
    packed_input_sequences = rnn_utils.pack_padded_sequence(input_sequences, batch_sizes)
    return packed_input_sequences


def loadModel(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name, ignore_optim=False):
    #ignore optim is for when I am loading in a model to assess predictions and not training it anymore. 
    for name, net, optim in zip(['encoder_save','decoder_save'],[encoder_net, decoder_net],[encoder_optimizer, decoder_optimizer] ):
        checkpoint = torch.load(load_name+name+'.tar')
        state = checkpoint['model_state_dict'] #.state_dict()
        #state.update(net.state_dict())
        net.load_state_dict(state)#checkpoint['model_state_dict'])
        if not ignore_optim:
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
        e = checkpoint['epoch']
        loss = checkpoint['last_loss']
        best_eval_acc = checkpoint['eval_accuracy']
    print('loaded in previous model!')
    return encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, e, best_eval_acc

def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)
    elif type(m) == torch.nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight' in name:
                torch.nn.init.xavier_uniform_(param, gain=torch.nn.init.calculate_gain('tanh'))
            if 'bias' in name:
                param = torch.zeros(param.shape)

def save_weights(m):
    save_init_weights = dict()
    for name, param in m.named_parameters():
        save_init_weights[name] = param
    return save_init_weights