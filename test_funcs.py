from sklearn import metrics
import time

def test(y_test, y_pred):
    # Assuming test_dataloader gives out y_pred as our predictions and y_test as the ground truth data
    acc = metrics.accuracy_score(y_test, y_pred)
    cm = metrics.confusion_matrix(y_test, y_pred)
    #Confusion matrix: TN = No malaria detected correctly         FP = Malaria detected, incorrectly
    #                  FN = No malaria detected, incorrectly      TP = Malaria detected correctly
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    precision = TP / float(TP + FP)

    return acc, cm, FP, FN, TP, TN, precision

def timer(model, train_loader, num_epochs=2):
    total_time = 0
    for i in range(num_epochs):
        time_before = time.time()
        #training takes place
        time_after = time.time()
        time_taken = float((time_after - time_before)/60)
        print("Time taken for epoch is: {0}".format(time_taken))
        total_time += time_taken
    avg_time = total_time/num_epochs
    return avg_time
