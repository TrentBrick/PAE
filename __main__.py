#from preprocessing import process_raw_data
import torch
import torch.utils.data
import h5py
import torch.autograd as autograd
import torch.optim as optim
import argparse
import numpy as np
import time
import requests
import math

from trainer import *
from models import *
from util import *
from nn_util import * 
from RAdam import *

import torch.optim as optim

def main():

    hide_ui = False
    if not hide_ui: 
        from dashboard import start_dashboard_server
        start_dashboard_server()

    mem_pin = False
    BATCH_SIZE = 32
    epochs = 500
    curr_ep = 1 # cant be 0 else later on there is division by zero!
    learning_rate=0.001
    use_DRMSD=False
    clip=30
    encoder_scheduler_on=False

    #LOAD IN EXISTING MODEL? 
    load_model =True
    save_name = 'LRexperiment' 
    load_name = 'LRexperiment'

    # WRONG FILEI FOR TRAINING FOR NOW!! 
    variant = '_trimmed'
    training_file = "data/preprocessed/testing"+variant+".hdf5"
    validation_file = "data/preprocessed/testing"+variant+".hdf5"
    testing_file = "data/preprocessed/testing"+variant+".hdf5"

    ENCODING_LSTM_OUTPUT=600
    META_ENCODING_LSTM_OUTPUT=600
    CODE_LAYER_SIZE=5000
    DECODING_LSTM_OUTPUT=600
    VOCAB_SIZE=21 # 20 amino acids and then the padding value too
    ENCODER_LSTM_NUM_LAYERS=1
    DECODER_LSTM_NUM_LAYERS=1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    readout=False
    allow_teacher_force = False
    teaching_strategy = 'epoch' # can also be 'accuracy'
    want_preds_printed = False

    # could make this more efficient by preventing padding from being predicted and rewriting the loss function for the sequences. 
    encoder_net = EncoderNet(device, ENCODING_LSTM_OUTPUT=ENCODING_LSTM_OUTPUT, META_ENCODING_LSTM_OUTPUT=META_ENCODING_LSTM_OUTPUT, CODE_LAYER_SIZE=CODE_LAYER_SIZE, 
                            VOCAB_SIZE=VOCAB_SIZE, ENCODER_LSTM_NUM_LAYERS=ENCODER_LSTM_NUM_LAYERS).to(device)
    decoder_net = DecoderNet(device, DECODING_LSTM_OUTPUT=DECODING_LSTM_OUTPUT, CODE_LAYER_SIZE=CODE_LAYER_SIZE, 
                            VOCAB_SIZE=VOCAB_SIZE, DECODER_LSTM_NUM_LAYERS=DECODER_LSTM_NUM_LAYERS).to(device)

    encoder_optimizer = RAdam(encoder_net.parameters(), lr=learning_rate)
    decoder_optimizer = RAdam(decoder_net.parameters(), lr=learning_rate) # optim.Adam

    ## Only actually used if encoder_scheduler_on = True. 
    # WATCH OUT FOR MIN VS MAX!!! If it is max then it reduces the LR when the value stops INCREASING. 
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', factor=0.9, patience=5, verbose=True, threshold=0.001, threshold_mode='abs' )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', factor=0.9, patience=5, verbose=True, threshold=0.001, threshold_mode='abs' )

    #encoder_optimizer = optim.SGD(encoder_net.parameters(), lr=learning_rate, momentum =0.9)
    #decoder_optimizer = optim.SGD(decoder_net.parameters(), lr=learning_rate, momentum =0.9)

    # initialize the model weights! 
    '''enc_saved_weights = dict()
    dec_saved_weights = dict()'''
    #nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    #for net in [encoder_net, decoder_net]: #, save_dict in zip([encoder_net, decoder_net], [enc_saved_weights, dec_saved_weights]):
    #    net.apply(init_weights)
        #save_dict = net.apply(save_weights)

    print('All models for this run will be saved under:', save_name)
    if load_model:
        print("LOADING IN A MODEL, load_model=True")
        encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadModel(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

    encoder_net.train()
    decoder_net.train()
  
    fitModel(encoder_net, decoder_net, encoder_optimizer, decoder_optimizer, 
            BATCH_SIZE, epochs, curr_ep, learning_rate, mem_pin, device, 
            save_name, load_name, readout, allow_teacher_force, teaching_strategy, 
            clip, want_preds_printed, encoder_scheduler, decoder_scheduler,
            training_file, validation_file, testing_file, hide_ui, encoder_scheduler_on=encoder_scheduler_on, use_DRMSD=use_DRMSD)

if __name__=='__main__':
    main()
