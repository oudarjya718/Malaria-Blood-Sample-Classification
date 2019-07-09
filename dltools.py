import numpy as np

__PRINT__ = [0,0]
__FILE__ = None

def make_image_grid(sample, row, col):

    img = sample[0]
    label = sample[1]

    r = 0
    c = 0
    inx = 0

    shape = img[0].size()

    if shape[0]==1:
        dim = 1
    else :
        dim = 3

    cflag = 0
    rflag = 0

    for i in range(0,row):

        cflag = 0

        for j in  range(0,col):

            try :
                temp = img[inx].cpu().detach().numpy()
            except IndexError :
                temp = np.zeros(shape)

            if dim == 3:
                temp = np.transpose(temp, (1,2,0))
            elif dim == 1:
                temp = temp[0]

            if cflag == 0 :
                col_matrix = temp
                cflag = 1

            elif cflag == 1 :
                col_matrix = np.concatenate((col_matrix,temp), axis=1)

            inx+=1

        if rflag == 0 :
            row_matrix = col_matrix
            rflag = 1

        else :
            row_matrix = np.concatenate((row_matrix,col_matrix),axis=0)

    return row_matrix

def print_train_progress( epoch, item_reached, length_data, train_loss, train=0, threshold=10):
      
        global __PRINT__
              
        percent = (item_reached/length_data)*100
        
               
        if int(percent) - __PRINT__[train]  >= threshold :
            __PRINT__[train] = (int(percent)//threshold)*threshold
            
            if train == 0:
                print("EPOCH : ",epoch,"\t completed ({}/{})\t\t:{}% \ttrain loss : {}".format(item_reached,length_data,int(percent),train_loss))
            
            if train == 1:
                print("EPOCH : ",epoch,"\t completed ({}/{})\t\t:{}% \taccuracy : {}".format(item_reached,length_data,int(percent),train_loss))
            
        
        if __PRINT__[train] >= 100:
            __PRINT__[train] = 0
        
        return
    
def creat_log(epoch, train_loss, accuracy, filename):
        
        log = open(filename, 'a')
        
        string = "EPOCH : {}\n\ttrain loss : {}\t\ttest accuracy : {}\n".format(epoch, train_loss, accuracy)
        
        print( string )
        log.write( string )
        
        log.close()
        
        return
        
    