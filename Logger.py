import torch
import os


__PRINTED__ = 0
__LOG_PRINTED__ = 0
__top_models__ = []
__epoch__ = 0


def print_train_progress(epoch, item_reached, length_data, loss, phase, accuracy=None):
        
        '''
        Summary :
        Gives a beautified output of the training progress

        Args :
        epoch = give the current epoch in the run
        item_reached = the number of data samples already encountered
        length_data = the actual size of the data
        loss = loss in the current batch
        phase = phase of training "train" or "validate"
        accuracy = if in phase == "validate" provide accuracy for priniting

        Returns : 
        Nothing just prints a beautified output
        '''
        
      
        global __PRINTED__
        
        percent = (item_reached/length_data)*100
        
            
        if int(percent)%10 == 0 and __PRINTED__ != int(percent):
            __PRINTED__ = int(percent)
            print("EPOCH : ",epoch,"\t completed ({}/{}) :\t".format(item_reached,length_data),int(percent),"%")
            
            if phase == 'train':
                print("\tLOSS : {:.6f}".format(loss))
            
            if phase == 'validate':
                print("\tLOSS : {:.6f} \tACCURACY : {}".format(loss,accuracy) )
         
        return

    
    
def write_log(file, epoch, item_reached, length_data, loss, phase, accuracy=None):
        
        '''
        Summary :
        Puts the output of train progress as displayed by print_train_progress() in a log file.
        The file is named as  :  Complete[Current date and time].log

        Args :
        file = name of the file where the output will be updated
        epoch = give the current epoch in the run
        item_reached = the number of data samples already encountered
        length_data = the actual size of the data
        loss = loss in the current batch
        phase = phase of training "train" or "validate"
        accuracy = if in phase == "validate" provide accuracy for priniting

        Returns : 
        Nothing just prints a beautified output
        '''
    
        global __LOG_PRINTED__
        
        log = open(file,"a")
        
        percent = (item_reached/length_data)*100
        
        if int(percent)%10 == 0 and __LOG_PRINTED__ != int(percent):
            __LOG_PRINTED__ = int(percent)
            
            
            log.write("EPOCH : "+str(epoch)+"\t completed ({}/{}) \t".format(item_reached,length_data)+str(int(percent))+"%\n")
            
            if phase == 'train':
                log.write("\tLOSS : {:.5f}\n".format(loss))
            
            if phase == 'validate':
                log.write("\tLOSS : {:.5f} \tACCURACY : {}\n".format(loss,accuracy))
                
        log.close()   
        
        return

    
    
def create_checkpoint( accuracy, epoch, loss, optimizer_state, model_state, device, chk_path):
    '''
    Summary :
    Creates model & optimizer (complete) checkpoints in the directory specified by the chk_path. 
    Also doesn't store uneccesary models but only the top three models in the current run.
    names the models as : model_at_[accuracy]%_epoch_[EPOCH].st
    
    WARNING : 
    If the kernel is sutdown or restarted three more new models will be created during the run.
    
    Args :
    accuracy = accuracy of the current model (so that model with lower accuracy can be deleted)
    epoch = current epoch that was finished
    loss = loss in the current epoch
    optimizer_state = Optimizer.state_dict()
    model_state = model.state_dict()
    device = device on which the model is running 
            (this will make the porting of models to different devices later easier) 
    chk_path = path to save the checkpoint

    Returns : 
        Nothing.
    '''
    
    global __top_models__
    
    name = "model_at_{:.3f}%_epoch_{}.st".format(accuracy,epoch)
    
    Complete_model_state = {"name":name,
                            "device":device,
                            "accuracy" : accuracy,
                            "loss" : loss,
                            "epoch" : epoch,
                            "optimizer_state": optimizer_state,
                            "model_state": model_state}
    
    
    
    __top_models__.append( [accuracy, name, Complete_model_state] ) 
    __top_models__.sort()
    
    if not os.path.isfile(os.path.join(chk_path,name)) :
        torch.save(Complete_model_state,os.path.join(chk_path,name))
    
    if len(__top_models__) > 3 :
        remove = __top_models__[3][2]["name"]
        path = os.path.join(chk_path,remove) 
        os.remove(path)
        
    __top_models__ = __top_models__[:3]
    
    return

def continue_checkpoint(PATH, model, optimizer, device):
    
    '''
    Summary :
    Loads the complete checkpoint saved into a specific device as chosen by the user.
    
    Args :
    PATH = the path with the model name (in full) which needs to be loaded
    device = the device to which this needs to be loaded
    model = model
    optimizer = Optimizer
    
    Returns : 
    model, Optimizer, epoch at which it stopped, loss at that epoch
    '''
    
    
    device = torch.device(device)
    checkpoint = torch.load(PATH)
    
    prev_device = checkpoint['device']
    
    if prev_device == device and device == "cpu" :
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    
    elif prev_device == device and device == "cuda" :
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        model.to(device)
        optimizer.to(device)
        
    elif prev_device == "cuda" and device == "cpu":
        model.load_state_dict(checkpoint["model_state"], map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer_state"], map_location=device)
        
    elif prev_device == "cpu" and device == "cuda":
        model.load_state_dict(checkpoint["model_state"], map_location="cuda:0")
        optimizer.load_state_dict(checkpoint["optimizer_state"], map_location="cuda:0")
        model.to(device)
        model.to(device)
        
    
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    
    return model, optimizer, epoch, loss