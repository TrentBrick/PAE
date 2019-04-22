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
from dashboard import start_dashboard_server

from trainer import *
from models import *
from util import *
from nn_util import * 

import torch.optim as optim

def main():

    # need to pass anything I do use here into fitModel at the bottom. 
    '''parser = argparse.ArgumentParser(description = "OpenProtein version 0.1")
    parser.add_argument('--silent', dest='silent', action='store_true',
                        help='Dont print verbose debug statements.')
    parser.add_argument('--hide-ui', dest = 'hide_ui', action = 'store_true',
                        default=True, help='Hide loss graph and visualization UI while training goes on.')
    parser.add_argument('--evaluate-on-test', dest = 'evaluate_on_test', action = 'store_true',
                        default=False, help='Run model of test data.')
    parser.add_argument('--eval-interval', dest = 'eval_interval', type=int,
                        default=5, help='Evaluate model on validation set every n epochs.')
    parser.add_argument('--min-updates', dest = 'minimum_updates', type=int,
                        default=5000, help='Minimum number of minibatch iterations.')
    parser.add_argument('--minibatch-size', dest = 'minibatch_size', type=int,
                        default=1, help='Size of each minibatch.')
    parser.add_argument('--learning-rate', dest = 'learning_rate', type=float,
                        default=0.01, help='Learning rate to use during training.')
    args, unknown = parser.parse_known_args()

    if args.hide_ui:
        write_out("Live plot deactivated, see output folder for plot.")'''

    # start web server ==============

    hide_ui = False
    if not hide_ui: 
        start_dashboard_server()

    # WRONG FILEI FOR TRAINING FOR NOW!! 
    variant = '_trimmed'
    training_file = "data/preprocessed/testing"+variant+".hdf5"
    validation_file = "data/preprocessed/testing"+variant+".hdf5"
    testing_file = "data/preprocessed/testing"+variant+".hdf5"

    ENCODING_LSTM_OUTPUT=300
    CODE_LAYER_SIZE=200
    DECODING_LSTM_OUTPUT=300
    VOCAB_SIZE=21
    ENCODER_LSTM_NUM_LAYERS=1
    DECODER_LSTM_NUM_LAYERS=1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

    mem_pin = False
    BATCH_SIZE = 32
    epochs = 500
    curr_ep = 1 # cant be 0 else later on there is division by zero!
    learning_rate=0.001
    clip=30

    readout=False
    allow_teacher_force = False
    teaching_strategy = 'epoch' # can also be 'accuracy'
    want_preds_printed = False

    encoder_net = EncoderNet(device, ENCODING_LSTM_OUTPUT=ENCODING_LSTM_OUTPUT, CODE_LAYER_SIZE=CODE_LAYER_SIZE, 
                            VOCAB_SIZE=VOCAB_SIZE, ENCODER_LSTM_NUM_LAYERS=ENCODER_LSTM_NUM_LAYERS).to(device)
    decoder_net = DecoderNet(device, DECODING_LSTM_OUTPUT=DECODING_LSTM_OUTPUT, CODE_LAYER_SIZE=CODE_LAYER_SIZE, 
                            VOCAB_SIZE=VOCAB_SIZE, DECODER_LSTM_NUM_LAYERS=DECODER_LSTM_NUM_LAYERS).to(device)

    encoder_optimizer = optim.Adam(encoder_net.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder_net.parameters(), lr=learning_rate)

    #encoder_optimizer = optim.SGD(encoder_net.parameters(), lr=learning_rate, momentum =0.9)
    #decoder_optimizer = optim.SGD(decoder_net.parameters(), lr=learning_rate, momentum =0.9)

    # initialize the model weights! 
    '''enc_saved_weights = dict()
    dec_saved_weights = dict()'''
    #nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))
    for net in [encoder_net, decoder_net]: #, save_dict in zip([encoder_net, decoder_net], [enc_saved_weights, dec_saved_weights]):
        net.apply(init_weights)
        #save_dict = net.apply(save_weights)

    #LOAD IN EXISTING MODEL? 
    load_model =False
    save_name = 'code300_' 
    load_name = 'code300_'

    print('All models for this run will be saved under:', save_name)
    if load_model:
        print("LOADING IN A MODEL, load_model=True")
        encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss, curr_ep, best_eval_acc = loadModel(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, load_name)

    encoder_net.train()
    decoder_net.train()

    # WATCH OUT FOR MIN VS MAX!!!
    encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'max', factor=0.5, patience=10, verbose=True, threshold=0.0001 )
    decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'max', factor=0.5, patience=10, verbose=True, threshold=0.0001  )
        
    fitModel(encoder_net, decoder_net, encoder_optimizer, decoder_optimizer, 
            BATCH_SIZE, epochs, curr_ep, learning_rate, mem_pin, device, 
            save_name, load_name, readout, allow_teacher_force, teaching_strategy, 
            clip, want_preds_printed, encoder_scheduler, decoder_scheduler,
            training_file, validation_file, testing_file, hide_ui)

if __name__=='__main__':
    main()