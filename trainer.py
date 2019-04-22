# do I need to import everything here too??
import torch 
from util import *
from models import *
from nn_util import * 
import pickle
import requests

def fitModel(encoder_net, decoder_net, encoder_optimizer, 
             decoder_optimizer, BATCH_SIZE, epochs, e, learning_rate, 
             mem_pin, device, save_name, load_name, readout, allow_teacher_force, 
             teaching_strategy, clip, want_preds_printed, encoder_scheduler, decoder_scheduler,
             training_file, validation_file, testing_file, hide_ui):
    
    print('save name for experiment', save_name)
    exp_id = set_experiment_id(save_name, learning_rate, BATCH_SIZE)

    train_loader = contruct_dataloader_from_disk(training_file, BATCH_SIZE)
    validation_loader = contruct_dataloader_from_disk(validation_file, BATCH_SIZE)
    train_dataset_size = train_loader.dataset.__len__()
    validation_dataset_size = validation_loader.dataset.__len__()
    
    print('device being used is: ', device)
    print('teacher forcing?', allow_teacher_force)
    print('teaching strategy', teaching_strategy)

    rmsd_avg_values = list()
    drmsd_avg_values = list()

    start = time.time()
    plot_losses_train = []
    plot_losses_eval = []
    sample_num=[]
    print_loss_total = 0 
    plot_loss_total = 0
    running_loss = 0
    prob_of_teacher_forcing = 0.7
    print_every=1
    plot_every=1
    accuracy_running =0.0
    num_batches_per_epoch = int(train_dataset_size/BATCH_SIZE)
    num_eval_batches_per_epoch = int(validation_dataset_size/BATCH_SIZE)
    accuracy=0.05*num_batches_per_epoch # need it for my teacher forcing calc. 
    best_eval_acc = 0.0
    print('approximate num of train batches per ep', num_batches_per_epoch)
    # how often to print minibatch loss
    print_mini_every = 100

    printParamNum(encoder_net, decoder_net)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    print('mem pinned? ', mem_pin)
    
    mini_batch_iters=0
    
    print('number of epochs', (epochs+e))
    
    init_e = e

    while e < (epochs+init_e):  
        print('Epoch', e)
        ep_loss = 0.0
        accuracy=0.0
        num_batches_per_epoch =0

        # iterate through data
        for x in train_loader:
            num_batches_per_epoch += 1
            start_compute_loss = time.time()

            seqs, coords, mask = x

            #seqs = torch.Tensor(seqs).to(device)
            #coords = torch.Tensor(coords).to(device)
            #mask = torch.Tensor(mask).to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            if allow_teacher_force:
                rand_for_teacher_forcing = np.random.rand(1)[0]
                teacher_force= True if rand_for_teacher_forcing < prob_of_teacher_forcing else False
            else:
                teacher_force= False
                
            seq_cross_ent_loss, seq_acc, angular_loss, drmsd_avg = train_forward(encoder_net, decoder_net, seqs, coords, mask, device, 
                                      teacher_forcing=teacher_force, readout=readout)#, print_preds=want_preds_printed)

            loss = seq_cross_ent_loss+angular_loss+drmsd_avg

            #write_out("Train loss:", float(loss))
            start_compute_grad = time.time()

            loss.backward()

            end = time.time()
            write_out("Loss time:", start_compute_grad-start_compute_loss, "Grad time:", end-start_compute_grad)
            
            # Clip gradients: gradients are modified in place
            encoder_clip = torch.nn.utils.clip_grad_norm_(encoder_net.parameters(), clip)
            if encoder_clip > clip:
                print('clipped encoder gradient with a value of: ', encoder_clip)
            decoder_clip = torch.nn.utils.clip_grad_norm_(decoder_net.parameters(), clip)
            if decoder_clip > clip:
                print('clipped decoder gradient with a value of: ', decoder_clip)

            encoder_optimizer.step()
            decoder_optimizer.step()
            
            accuracy += seq_acc.item()
            accuracy_running += seq_acc.item()

            print_loss_total += loss.item()
            plot_loss_total += loss.item()

            running_loss +=  loss.item()
            ep_loss +=  loss.item()

            mini_batch_iters+=1

            if mini_batch_iters % print_mini_every == (print_mini_every-1):
                print('# of minibatch iters:', (mini_batch_iters+1), 'prints every', print_mini_every, 'loss: ', 
                      round( (running_loss/print_mini_every), 4), 'accuracy', round((accuracy_running/print_mini_every),4) )
                running_loss = 0.0
                accuracy_running =0.0

        print('Total loss for epoch ',e, 'is: ', round(ep_loss ,4))

        if (e % print_every == 0):
            print_loss_avg = (print_loss_total / print_every) / num_batches_per_epoch
            plot_losses_train.append(print_loss_avg)
            print_loss_total = 0
            print('Time passed: %s Time Till Done: (%d %d%%) Loss average: %.4f Accuracy: %.4f' % (timeSince(start, e / epochs), e, e /epochs * 100, print_loss_avg, 
                                                                                                   accuracy/num_batches_per_epoch) )
        # cross_eval of model: 
        with torch.no_grad():
            encoder_net.eval()
            decoder_net.eval()
            tot_eval_loss = 0.0
            tot_eval_acc = 0.0
            num_eval_batches_per_epoch = 0 

            epoch_avg_rmsd = 0.0
            epoch_avg_drmsd = 0.0

            for x in validation_loader:
                num_eval_batches_per_epoch += 1
                
                seqs, coords, mask = x
                # returns different things because this is the evaluation step! 
                rmsd, drmsd, eval_seq_cross_ent_loss, eval_seq_acc, eval_angular_loss, angles, angles_pred, pred_seqs= train_forward(encoder_net, decoder_net, seqs, coords, mask, device, teacher_forcing=False, readout=readout, print_preds=want_preds_printed, eval_mode=True)
                # angles here are all still padded so is the sequence.  
                epoch_avg_rmsd += rmsd
                epoch_avg_drmsd += drmsd
                
                eval_loss = eval_seq_cross_ent_loss+eval_angular_loss
                
                tot_eval_loss+= eval_loss.item()
                tot_eval_acc+= eval_seq_acc.item()

            drmsd_avg_values.append( epoch_avg_rmsd /num_eval_batches_per_epoch )
            rmsd_avg_values.append( epoch_avg_drmsd /num_eval_batches_per_epoch )

            # plot only the last datapoint from the whole eval epoch! seq is not padded
            # I dont need to mask the angles because the first in the batch has no padding!
            rand_protein_to_display = int(np.random.rand(1)[0] * len(seqs))
            s = seqs[rand_protein_to_display].to(device)
            real_angles = angles[:,rand_protein_to_display,:][:s.shape[0]]
            pred_angles = angles_pred[:,rand_protein_to_display,:][:s.shape[0]]
            write_to_pdb(get_structure_from_angles(s, real_angles), "test")
            write_to_pdb(get_structure_from_angles(pred_seqs[rand_protein_to_display][:s.shape[0]].max(dim=1)[1], pred_angles), "test_pred")

            tot_eval_acc = tot_eval_acc/num_eval_batches_per_epoch
            tot_eval_loss = tot_eval_loss/num_eval_batches_per_epoch
            plot_losses_eval.append(tot_eval_loss)
            sample_num.append(mini_batch_iters)
            print('Eval Loss average per batch: %.4f Accuracy: %.4f' % (tot_eval_loss, tot_eval_acc) ) 
            #right now this is actually train accuracy just because I want to overfit!!! 
            if (accuracy/num_batches_per_epoch>best_eval_acc):
                print('new best eval_accuracy! At:', round(tot_eval_acc,4),' Saving model')
                best_eval_acc = accuracy/num_batches_per_epoch
                saveModel(exp_id, encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, loss.item(), tot_eval_acc, e)
        
            if not hide_ui:
                data = {}
                data["pdb_data_pred"] = open("output/protein_test_pred.pdb","r").read()
                data["pdb_data_true"] = open("output/protein_test.pdb","r").read()
                data["validation_dataset_size"] = validation_dataset_size
                data["sample_num"] = sample_num
                data["train_loss_values"] = plot_losses_train
                data["validation_loss_values"] = plot_losses_eval
                data["phi_actual"] = list([math.degrees(float(v)) for v in real_angles[1:,1]])
                data["psi_actual"] = list([math.degrees(float(v)) for v in real_angles[:-1,2]])
                data["phi_predicted"] = list([math.degrees(float(v)) for v in pred_angles[1:,1]])
                data["psi_predicted"] = list([math.degrees(float(v)) for v in pred_angles[:-1,2]])
                data["drmsd_avg"] = drmsd_avg_values
                data["rmsd_avg"] = rmsd_avg_values
                res = requests.post('http://localhost:5000/graph', json=data)
                if res.ok:
                    print(res.json())

        encoder_net.train()
        decoder_net.train()
        
        # determine probability of teacher forcing for the next epoch
        if(allow_teacher_force and teaching_strategy =='accuracy'): 
            prob_of_teacher_forcing = 1-((accuracy/num_batches_per_epoch)+0.3)
            print('prob use teacher forcing this epoch', round(prob_of_teacher_forcing,2))
            
        elif(allow_teacher_force and teaching_strategy=='epoch'):
            prob_of_teacher_forcing = 0.7-(e/100)
            print('prob use teacher forcing this epoch', round(prob_of_teacher_forcing,2))
        
        pickle.dump((plot_losses_train, plot_losses_eval), open(save_name+'list_of_losses_to_plot.pickle', 'wb'))
        
        encoder_scheduler.step(accuracy/num_batches_per_epoch)
        decoder_scheduler.step(accuracy/num_batches_per_epoch)
        
        e +=1

def train_forward(encoder_net, decoder_net, inp_seqs, coords, mask, device, readout=True, teacher_forcing=False, make_preds=False, print_preds=False, eval_mode=False ):

    seqs, lengths = torch.nn.utils.rnn.pad_packed_sequence(
                    torch.nn.utils.rnn.pack_sequence(inp_seqs))

    latent, padded_dihedrals = encoder_net(seqs.to(device), lengths, coords)
    
    batch_size = latent.shape[0]
    max_l = torch.max(lengths) # NEED TO GET THE LONGEST SEQUENCE HERE! 
    #hidden = decoder_net.initHidden(batch_size) put in if want to do teacher forcing. 
    # need to give each of the latents a time step proportional to their length number. 
    # get rid of the time STEPS IF DOING TEACHER FORCING AND INCREMENT THE INPUTS BY ONE!
    if(readout==True): 
        batch_outs = torch.zeros([batch_size,max_l, VOCAB_SIZE], device=device, requires_grad=False)
        if (not teacher_forcing):
            #time_steps=1
            prev_out = torch.zeros([batch_size,1, 1], device=device, requires_grad=False)
            latent = latent.view(latent.shape[0],1,latent.shape[1])
            for t in range(max_l):
                prev_out, hidden = decoder_net(latent, prev_out, hidden)
                batch_outs[:,t,:]= prev_out.squeeze()
                prev_out = prev_out.max(dim=2, keepdim=True)[1].to(torch.float32)
                
        else:
            latent = latent.view(latent.shape[0],1,latent.shape[1]).expand(-1,max_l.item(),-1)
            ground_truth = seqs.max(dim=2, keepdim=True)[1][:,:-1,:]
            #prev_out = input[0].max(dim=2, keepdim=True)[1].to(torch.float32).to(device) # this was just seeing if the model would learn. 
            prev_out = torch.cat( (torch.zeros([ground_truth.shape[0], 1, 1]), ground_truth.to(torch.float32) ) ,1).to(device) #as both tensors should already be in cuda!
            batch_outs, hidden = decoder_net(latent, prev_out, hidden)
        # add zero padding to the end. Dont need to do as I am computing each batch now. 
        #batch_outs[b_ind,:,:].add_( torch.cat( ( seq_outs, torch.zeros([ (max_l-length) ,VOCAB_SIZE] , device=device, requires_grad=False) ) , 0))

    else: 
        #prev_out = torch.zeros([batch_size,max_l.item(), 1], device=device, requires_grad=False)
        # this was for the original keras model where I needed to repeat vector the latent space. 
        pred_seqs, pred_dihedrals, backbone_atoms_padded, batch_sizes_backbone = decoder_net( latent.view(batch_size,1,-1).expand(-1,max_l.item(),-1))

    #only use the backbone_atoms_padded if I want to calc. drmsd. these are the predicted coords! 

    mask, _ = torch.nn.utils.rnn.pad_packed_sequence(
                        torch.nn.utils.rnn.pack_sequence(mask))

    if make_preds:
        return pred_seqs, backbone_atoms_padded
    
    if print_preds:
        # always print a random number.
        rand_ind = int(torch.rand([1]).item()*batch_size)
        s = inp_seqs[rand_ind]
        top_preds = pred_seqs[rand_ind].max(dim=1)[1]  # [:s.shape[0]]
        print('ground truth sequence', s)
        print('predicted values sequence', top_preds)

    if eval_mode:
        dRMSD_list = [] # remove once I parallelize angle calculations!!! 
        RMSD_list = []  
        start = time.time()
        eval_seq_cross_ent_loss, eval_seq_acc, eval_angular_loss = seq_and_angle_loss(pred_seqs, seqs.t(), pred_dihedrals, padded_dihedrals, mask, use_mask=False)
        # how can I parallelize these calculations?? 

        for tertiary_positions, predicted_backbone_atoms in zip(coords, backbone_atoms_padded.permute([1,0,2])):
            to_remove_padding =  tertiary_positions.shape[0]
            actual_coords = tertiary_positions.transpose(0,1).contiguous().view(-1,3)
            predicted_coords = predicted_backbone_atoms[:to_remove_padding].contiguous().view(-1,3).detach()
            rmsd = calc_rmsd(predicted_coords, actual_coords)
            drmsd = calc_drmsd(predicted_coords, actual_coords, device)
            RMSD_list.append(rmsd)
            dRMSD_list.append(drmsd)

        end = time.time()
        write_out("Calculating all validation losses for minibatch took:", end - start)

        return torch.Tensor(RMSD_list).mean().item(), torch.Tensor(dRMSD_list).mean().item(), eval_seq_cross_ent_loss, eval_seq_acc, eval_angular_loss, padded_dihedrals, pred_dihedrals, pred_seqs
    
    # feed in the sequence length for each example and the truth
    seq_cross_ent_loss, seq_acc, angular_loss = seq_and_angle_loss(pred_seqs, seqs.t(), pred_dihedrals, padded_dihedrals, mask, use_mask=False)
    # calc drmsd over minibatch: 
    # same for loop as in the evaluation dataset!! 
    dRMSD_list = []
    for tertiary_positions, predicted_backbone_atoms in zip(coords, backbone_atoms_padded.permute([1,0,2])):
        to_remove_padding =  tertiary_positions.shape[0]
        actual_coords = tertiary_positions.transpose(0,1).contiguous().view(-1,3)
        predicted_coords = predicted_backbone_atoms[:to_remove_padding].contiguous().view(-1,3).detach()
        drmsd = calc_drmsd(predicted_coords, actual_coords, device)
        dRMSD_list.append(drmsd)
    drmsd_avg = torch.Tensor(dRMSD_list).mean().item()
    return seq_cross_ent_loss, seq_acc, angular_loss, drmsd_avg