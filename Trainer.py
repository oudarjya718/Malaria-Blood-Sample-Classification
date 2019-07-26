import torch
import torch.optim


#Avoid writing anything in the notebook as that breaks the idea of standard template
#Write your train routine here, if need be for a different routine

def train(model, optimizer, criterion, data):
    
    optimizer.zero_grad()
    
    img, lbl = data
    
    preds = model(img)
    
    loss = criterion(preds, lbl)
    loss.backward()
    
    optimizer.step()
    
    return loss.item()



#Write your validate routine here, if need be for a different routine

def validate(model, criterion, data):
    
    img, lbl = data
    
    with torch.no_grad():
        
        preds = model(img)
        loss = criterion(preds, lbl)
        
        top_p, top_class = preds.topk( 1, dim=1)
        accuracy = ( top_class == lbl.view(*top_class.shape) ).sum()
        
    
    return loss.item(), accuracy.item()




def test(model, dataloader):
    
    length = 0
    
    for data in dataloader :
        
        img, lbl = data
        length+=len(lbl)    
        
        with torch.no_grad():

            preds = model(img)
            top_p, top_class = torch.topk(preds, 1, dim=1)
            accuracy += ( top_class == lbl.view(*top_class.shape) ).sum()
    
    accuracy = 100 * accuracy/length
    
    return accuracy