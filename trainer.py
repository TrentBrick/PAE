# do I need to import everything here too??
#import torch 
#from util import *
#from models import *

def fitModel(encoder_net, decoder_net, encoder_optimizer, 
             decoder_optimizer, BATCH_SIZE, epochs, e, learning_rate, 
             mem_pin, device, save_name, load_name, readout, allow_teacher_force, 
             teaching_strategy, clip, want_preds_printed, encoder_scheduler, decoder_scheduler,
             training_file, validation_file, testing_file):
    
    
    set_experiment_id(save_name, learning_rate, BATCH_SIZE)

    '''sampler = BucketDataset(primary_train, BATCH_SIZE)
    dataloader = DataLoader(df_train, batch_size=1, 
                            batch_sampler=sampler, shuffle=False,
                            num_workers=8, collate_fn=one_hotter, 
                            drop_last=False, pin_memory=mem_pin)

    evaluate_sampler = BucketDataset(primary_cv, BATCH_SIZE)
    evaluate_dataloader = DataLoader(df_cv, batch_size=1, 
                        batch_sampler=evaluate_sampler, shuffle=False,
                        num_workers=8, collate_fn=one_hotter, 
                        drop_last=False, pin_memory=mem_pin)'''

    train_loader = contruct_dataloader_from_disk(train_file, BATCH_SIZE)
    validation_loader = contruct_dataloader_from_disk(val_file, BATCH_SIZE)
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
            
            seqs = x[0].to(device)
            coords = x[1].to(device)
            mask = x[2].to(device)

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            if allow_teacher_force:
                rand_for_teacher_forcing = np.random.rand(1)[0]
                teacher_force= True if rand_for_teacher_forcing < prob_of_teacher_forcing else False
            else:
                teacher_force= False
                
            seq_cross_ent_loss, seq_acc, angular_loss = train_forward(encoder_net, decoder_net, seqs, coords, mask, device, 
                                      teacher_forcing=teacher_force, readout=readout)#, print_preds=want_preds_printed)

            loss = seq_cross_ent_loss+angular_loss
            loss.backward()
            
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
            #print('mini batch count', mini_batch_iters)
            #print('number of batches per ep', num_batches_per_epoch)
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
            for x in validation_loader:
                num_eval_batches_per_epoch += 1
                
                seqs = x[0].to(device)
                coords = x[1].to(device)
                mask = x[2].to(device)

                seq_cross_ent_loss, seq_acc, angular_loss = train_forward(encoder_net, decoder_net, seqs, device, teacher_forcing=False, readout=readout, print_preds=want_preds_printed)
                eval_loss = seq_cross_ent_loss+angular_loss
                
                tot_eval_loss+= eval_loss.item()
                tot_eval_acc+= eval_acc.item()
            tot_eval_acc = tot_eval_acc/num_eval_batches_per_epoch
            tot_eval_loss = tot_eval_loss/num_eval_batches_per_epoch
            plot_losses_eval.append(tot_eval_loss)
            print('Eval Loss average per batch: %.4f Accuracy: %.4f' % (tot_eval_loss, tot_eval_acc) ) 
            #right now this is actually train accuracy just because I want to overfit!!! 
            if (accuracy/num_batches_per_epoch>best_eval_acc):
                print('new best eval_accuracy! At:', round(tot_eval_acc,4),' Saving model')
                best_eval_acc = accuracy/num_batches_per_epoch
                saveModel(encoder_net, decoder_net,encoder_optimizer, decoder_optimizer, save_name, loss.item(), tot_eval_acc, e)
        
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

def train_forward(encoder_net, decoder_net, seqs, coords, mask, device, readout=True, teacher_forcing=False, make_preds=False, print_preds=False ):
  
    latent, padded_dihedrals = encoder_net(seqs, coords)
    
    batch_size = seqs.shape[0]
    max_l = torch.max(seqs.shape[1]) # NEED TO GET THE LONGEST SEQUENCE HERE! 
    #hidden = decoder_net.initHidden(batch_size)
    
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
                #print('is prev out requires grad? ', prev_out.requires_grad)
                batch_outs[:,t,:]= prev_out.squeeze()
                prev_out = prev_out.max(dim=2, keepdim=True)[1].to(torch.float32)
                
        else:
            latent = latent.view(latent.shape[0],1,latent.shape[1]).expand(-1,max_l.item(),-1)
            ground_truth = seqs.max(dim=2, keepdim=True)[1][:,:-1,:]
            #prev_out = input[0].max(dim=2, keepdim=True)[1].to(torch.float32).to(device) # this was just seeing if the model would learn. 
            prev_out = torch.cat( (torch.zeros([ground_truth.shape[0], 1, 1]), ground_truth.to(torch.float32) ) ,1).to(device) #as both tensors should already be in cuda!
            #print('shape of prev out', prev_out.shape)
            #print('what does prev out look like? arranged right?', prev_out[0,:,:])
            batch_outs, hidden = decoder_net(latent, prev_out, hidden)
        # add zero padding to the end. Dont need to do as I am computing each batch now. 
        #batch_outs[b_ind,:,:].add_( torch.cat( ( seq_outs, torch.zeros([ (max_l-length) ,VOCAB_SIZE] , device=device, requires_grad=False) ) , 0))

    else: 
        #prev_out = torch.zeros([batch_size,max_l.item(), 1], device=device, requires_grad=False)
        # this was for the original keras model where I needed to repeat vector the latent space. 
        pred_seqs, pred_dihedrals, backbone_atoms_padded, batch_sizes_backbone = decoder_net( latent.view(batch_size,1,-1).expand(-1,max_l.item(),-1))

    #only use the backbone_atoms_padded if I want to calc. drmsd.

    if make_preds:
        return pred_seqs, backbone_atoms_padded
    
    if print_preds:
        # always print a random number. 
        rand_ind = int(torch.rand([1]).item()*batch_size)
        print('ground truth sequence', seqs.max(dim=2)[1][rand_ind,:])
        print('predicted values sequence', pred_seqs.max(dim=2)[1][rand_ind,:])
    
    # feed in the sequence length for each example and the truth
    return seq_and_angle_loss(pred_seqs, seqs, pred_dihedrals, padded_dihedrals, mask)