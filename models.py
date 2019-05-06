import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from util import calculate_dihedral_angles_over_minibatch, get_backbone_positions_from_angular_prediction
from nn_util import embed
class EncoderNet(nn.Module):
    def __init__(self, device, ENCODING_LSTM_OUTPUT=100, META_ENCODING_LSTM_OUTPUT=50 ,CODE_LAYER_SIZE=50, VOCAB_SIZE=21, ENCODER_LSTM_NUM_LAYERS=2 ):
        super(EncoderNet, self).__init__()
        #encoding
        self.LSTM_NUM_LAYERS = ENCODER_LSTM_NUM_LAYERS
        self.ENCODING_LSTM_OUTPUT = ENCODING_LSTM_OUTPUT
        self.META_ENCODING_LSTM_OUTPUT = META_ENCODING_LSTM_OUTPUT
        self.encoder_seq = nn.LSTM(input_size=VOCAB_SIZE, hidden_size= self.ENCODING_LSTM_OUTPUT,num_layers=self.LSTM_NUM_LAYERS,bidirectional=True, batch_first=False)
        self.encoder_tert = nn.LSTM(input_size=3, hidden_size= self.ENCODING_LSTM_OUTPUT,num_layers=self.LSTM_NUM_LAYERS,bidirectional=True, batch_first=False)
        self.encoder_meta = nn.LSTM(input_size=self.ENCODING_LSTM_OUTPUT*4, hidden_size= self.META_ENCODING_LSTM_OUTPUT,num_layers=1,bidirectional=True, batch_first=False)
        self.batchnorm = nn.BatchNorm1d(self.META_ENCODING_LSTM_OUTPUT * 2) # as bidirectional LSTMS, 2 of them. 
        self.dense1_enc = nn.Linear(in_features=(self.META_ENCODING_LSTM_OUTPUT*2 ), out_features=CODE_LAYER_SIZE ) # * self.LSTM_NUM_LAYERS because it is bidirectional
        self.dense2_enc = nn.Linear(in_features=CODE_LAYER_SIZE, out_features=CODE_LAYER_SIZE )
        self.device = device

    def forward(self, seq, batch_sizes, tert):
        packed_input_sequences = embed(seq, batch_sizes, self.device)
        # dealing with the sequences: 
        packed_output, hidden = self.encoder_seq(packed_input_sequences)
        # batch comes second here? so shape[1]
        # commented out so I can add the meta LSTM. need to unpack also!!  
        out_padded_seq, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        #seq_hidden_means = torch.sum(out_padded, dim=0) / lengths.view(-1,1).expand(-1, self.ENCODING_LSTM_OUTPUT*2).type(torch.float)
        #Now dealing with the tertiary structure!! Convert coords to dihedral angles. 
        # None here is because this is not padded and I dont want to give it a batch size.
        #print('pre dihedral', tert)
        tert_angles = calculate_dihedral_angles_over_minibatch(tert, None, self.device, is_padded=False) 
        # convert this into a packed sequence! 
        #print('pre packing', tert_angles)
        packed_tert_angles = torch.nn.utils.rnn.pack_sequence(tert_angles).to(self.device)
        #this is to return for the loss function, the real dihedral angles: 
        padded_real_angles, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_tert_angles)
        # dealing with the sequences: 
        packed_output, hidden = self.encoder_tert(packed_tert_angles)
        # need to unpack and get the means here! 
        out_padded_tert, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_output)
        #tert_hidden_means = torch.sum(out_padded, dim=0) / lengths.view(-1,1).expand(-1, self.ENCODING_LSTM_OUTPUT*2).type(torch.float)
        # get mean of all hidden states. will then concat this with the tertiary and put through dense.        
        
        # meta encoder LSTM: concat all the hidden states from every time step!!  
        res = torch.cat( (out_padded_seq, out_padded_tert ), dim=2 )
        res = torch.nn.utils.rnn.pack_padded_sequence(res, lengths)
        res, hidden = self.encoder_meta(res)
        res, lengths= torch.nn.utils.rnn.pad_packed_sequence(res)
        res = torch.sum(res, dim=0) / lengths.view(-1,1).expand(-1, self.META_ENCODING_LSTM_OUTPUT*2).type(torch.float).to(self.device)
        #res = torch.cat( (seq_hidden_means, tert_hidden_means), dim=1)       
        #Ignore the batchnorm for now! 
        #res = self.batchnorm(res)
        res = self.dense2_enc(F.elu(self.dense1_enc(res))) # used to have F.tanh here!
        # out_padded are the dihedral angles for the structure!! 
        return res, padded_real_angles

class soft_to_angle(nn.Module):
    def __init__(self, mixture_size):
        super(soft_to_angle, self).__init__()
        # Omega intializer
        omega_components1 = np.random.uniform(0, 1, int(mixture_size * 0.1))  # Initialize omega 90/10 pos/neg
        omega_components2 = np.random.uniform(2, math.pi, int(mixture_size * 0.9))
        omega_components = np.concatenate((omega_components1, omega_components2))
        np.random.shuffle(omega_components)

        phi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 1]
        psi_components = np.genfromtxt("data/mixture_model_pfam_"+str(mixture_size)+".txt")[:, 2]

        self.phi_components = nn.Parameter(torch.from_numpy(phi_components).contiguous().view(-1, 1).float())
        self.psi_components = nn.Parameter(torch.from_numpy(psi_components).contiguous().view(-1, 1).float())
        self.omega_components = nn.Parameter(torch.from_numpy(omega_components).view(-1, 1).float())

    def forward(self, x):
        phi_input_sin = torch.matmul(x, torch.sin(self.phi_components))
        phi_input_cos = torch.matmul(x, torch.cos(self.phi_components))
        psi_input_sin = torch.matmul(x, torch.sin(self.psi_components))
        psi_input_cos = torch.matmul(x, torch.cos(self.psi_components))
        omega_input_sin = torch.matmul(x, torch.sin(self.omega_components))
        omega_input_cos = torch.matmul(x, torch.cos(self.omega_components))

        eps = 10 ** (-4)
        phi = torch.atan2(phi_input_sin, phi_input_cos + eps)
        psi = torch.atan2(psi_input_sin, psi_input_cos + eps)
        omega = torch.atan2(omega_input_sin, omega_input_cos + eps)

        return torch.cat((phi, psi, omega), 2)

class DecoderNet(nn.Module):
    def __init__(self, device, DECODING_LSTM_OUTPUT=100, CODE_LAYER_SIZE=50, VOCAB_SIZE=20, DECODER_LSTM_NUM_LAYERS=1 ):
        super(DecoderNet, self).__init__()
        #decode it should be the inverse of the encoder!
        #self.dense1_predecode = nn.Linear(in_features=(CODE_LAYER_SIZE), out_features=50 )
        #self.dense1_pre_dec = nn.Linear(in_features=(CODE_LAYER_SIZE), out_features=200 )
        self.LSTM_OUTPUT_SIZE = DECODING_LSTM_OUTPUT
        self.LSTM_NUM_LAYERS=DECODER_LSTM_NUM_LAYERS
        self.CODE_LAYER_SIZE=CODE_LAYER_SIZE
        self.VOCAB_SIZE=VOCAB_SIZE
        self.decoder = nn.LSTM(input_size=CODE_LAYER_SIZE,bidirectional=True, 
                               hidden_size=self.LSTM_OUTPUT_SIZE,num_layers=self.LSTM_NUM_LAYERS, batch_first=True)
        
        self.latent_to_dihedral1 = nn.Linear(in_features=self.LSTM_OUTPUT_SIZE*2, out_features=50 )
        self.latent_to_dihedral2 = nn.Linear(in_features=50, out_features=3 )
        self.mixture_size = 500
        self.hidden_to_labels = nn.Linear(self.LSTM_OUTPUT_SIZE * 2, self.mixture_size, bias=True) # * 2 for bidirectional
        self.softmax_to_angle = soft_to_angle(self.mixture_size)
        self.soft = nn.LogSoftmax(2)
        self.bn = nn.BatchNorm1d(self.mixture_size)
        self.dense2_post_dec = nn.Linear(in_features=self.LSTM_OUTPUT_SIZE*2, out_features=VOCAB_SIZE )
        #self.dense3_post_dec = nn.Linear(in_features=50, out_features=VOCAB_SIZE )
        self.device = device

    def forward(self, latent_space):
        #decoding. takes in a single latent code from a single part of the batch. 
        # where input is the teacher forcing or the predictions from the previous step.

        # batch_first = True for this. 
        prev_out, hidden = self.decoder(latent_space)
        #predict sequences: 
        pred_seqs = F.log_softmax(self.dense2_post_dec(prev_out), dim=2)

        # alternatives to weird angle mixture model. 
        # # just predicting dihedrals directly. Then try to scale them to be between +/-pi
        # # then try using atan2
            #np.pi* F.tanh(
        output_angles = self.latent_to_dihedral2(F.elu(self.latent_to_dihedral1(prev_out))).permute([1,0,2]) # max size, minibatch size, 3 (angels)
        #print('output angles shape::: ', output_angles.shape)
        ###print('output angles::: ', output_angles)
        # weird angle mixture model thing. 
        '''x = self.hidden_to_labels(prev_out)
        x = self.bn(x.permute([0,2,1])).permute([0,2,1]).contiguous()
        #x = x.transpose(1,2) #(minibatch_size, -1, self.mixture_size)
        p = torch.exp(self.soft(x))
        output_angles = self.softmax_to_angle(p).transpose(0,1) # max size, minibatch size, 3 (angels)'''
        # used to feed in batch sizes here. could do so from encoder. but all I do I take the length of it... 
        backbone_atoms_padded, batch_sizes_backbone = get_backbone_positions_from_angular_prediction(output_angles, 
                                                                        latent_space.shape[0], self.device)     
        
        if torch.isnan(backbone_atoms_padded).sum() >0 :
            print('angles are NOT VALID!!!++++++++++++++++++++++++')

        return pred_seqs, output_angles, backbone_atoms_padded, batch_sizes_backbone
    
    '''def initHidden(self, batch_size):
        hidden_h = torch.empty([self.LSTM_NUM_LAYERS,batch_size,int(self.LSTM_OUTPUT_SIZE)], device=device, requires_grad=True)
        torch.nn.init.orthogonal_(hidden_h, gain=nn.init.calculate_gain('tanh'))
        return (hidden_h,
                torch.zeros([self.LSTM_NUM_LAYERS,batch_size,int(self.LSTM_OUTPUT_SIZE)], device=device, requires_grad=True))'''