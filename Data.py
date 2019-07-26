from PIL import Image
import numpy as np
import torch as torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from pprint import pprint
import subprocess

#Variables for users to edit


__batch_size__ = 16
folder_path = "//media/rishi/Ubuntu file 2/datasets/cell_images/"


#Variables that are for internal use (don't mess with this until you understand the code)

__length__ = {"train" : 0, "validate" : 0, "test" : 0}
class_names = { 0 : "Parasitized", 1 : "Uninfected" }


#definition of DataLoader (Pneumonia guys may need to edit this for classification)

class Data(Dataset):
    
    def __init__(self, folder_path, infected_list, uninfected_list, device="cpu", transform=None):
        
        super(Data, self).__init__()
        
        self.path = folder_path
        self.device = device
        
        self.classes = { 0 : infected_list, 1 : uninfected_list }
        
        self.marker = len(self.classes[0])
        
        self.lenth = len(self.classes[0]) + len(self.classes[1])
        
        if transform == None :
            self.transform = transforms.Compose([transforms.ToTensor()])
        else :
            self.transform = transform
        
        return
    
    def __getitem__(self, inx):
        
        
        if inx - self.marker < 0:
            index = inx
            cls = 0
        
        else :
            index = inx - self.marker
            cls = 1
        
        path = self.path + class_names[cls] + '/' + self.classes[cls][index]
        
        image = Image.open(path)    
        image = self.transform(image)
        label = torch.tensor(cls, dtype = torch.long, device = self.device)
        
        return image, label
    
    
    def __len__(self):
        
        return self.lenth

#Routines used in the template 
    
def getData(split = [0.8, 0.1], device="cpu", transform=None):
    
    '''
    Summary :
    The function is used to get data loader. This returns a dictionary with "train", "validate" and "test" dataloaders 
    (The above marken in quotes are precisely the keys for the dictionary).
    
    Args :
    split = [train size, validate size]
    device = device in which the data loader has to be loaded
    transform = torch.Compose() transforms
    
    Returns : 
    a dictionary {"train" : dataloader1, "validate" : dataloader2, "test" : dataloader3}
    '''
    
    global __length__
    
    
    
    #Load your data for prediction

    Infected_list = str(subprocess.check_output(["ls", folder_path+class_names[0]+'/']), 'utf-8').split("\n")
    Uninfected_list = str(subprocess.check_output(["ls", folder_path+class_names[1]+'/']),'utf-8').split("\n")
    Infected_list.remove("")
    Uninfected_list.remove("")
     
    data = Data(folder_path, Infected_list, Uninfected_list, device, transform)
    
    #The below code creates a train, valid & test split based on the splitting proportion given above, 
    #The length of the test is infered from the train and valid
    #The below lines need not be touched they are working fine
    
    __length__["train"] = int(split[0]*len(data))
    __length__["validate"] = int(split[1]*len(data))
    __length__["test"]  = len(data) - (__length__["train"] + __length__["validate"])
    
    trainData, validData, testData = torch.utils.data.random_split(data, lengths=[__length__["train"], __length__["validate"], __length__["test"]])
    
    trainDataLoader = DataLoader(trainData, batch_size=__batch_size__, shuffle=False, num_workers=0)
    validDataLoader = DataLoader(validData, batch_size=__batch_size__, shuffle=False, num_workers=0)
    testDataLoader = DataLoader(testData, batch_size=__batch_size__, shuffle=False, num_workers=0)

    dataLoader = {"train" : trainDataLoader, "validate" : validDataLoader, "test" : testDataLoader }
    
    return dataLoader


def getLength(phase="train"):
    
    '''
    Summary :
    This gets the total length of the specified dataloader of that phase ("train", "validate" or "test")
    
    Args :
    phase = "train", "validate" or "test"
    
    Returns : 
    the size of the data for that phase
    '''
    
    return __length__[phase]

def getBatchSize():
    
    '''
    Summary :
    Returns the size of the batch size chosen
    
    Args :
    None
    
    Returns : 
    Batch size that was specified
    '''
    
    return __batch_size__