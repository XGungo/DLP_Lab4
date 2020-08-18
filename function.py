import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import time

# show training and testing accuracy trend
def show_result(accuracy, model):            
    label = ["Train(with pretraining)", "Train(w/o pretraining)", "Test(with pretraining)", "Test(w/o pretraining)"]
    
    plt.figure(figsize=(10, 6))
    
    plt.ylabel("Accuracy(%)")
    plt.xlabel("Epochs")
    plt.title("Result comparison("+model+")", fontsize=18)
    plt.yticks(np.arange(0.7, 0.9, step=0.02))
    
    for idx, acc in enumerate(accuracy):
        plt.plot(acc, label=label[idx], marker='o')

    plt.legend()
    plt.show()



# predict and calculate accuracy
def get_predict(model, loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    predictions = None
    total = 0
    correct = 0
    
    with torch.no_grad():
        for data, label in loader:
            data = data.to(device)
            label = label.to(device)
            
            # forward pass
            output = model(data)
            output = nn.Softmax(dim=1)(output)
            
            # get output label
            _, pred = torch.max(output, 1)

            total += label.shape[0]
            correct += torch.sum(pred == label).item()
            
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
            
            # if plot=True, keep labels for confusion matrix

                
    acc = correct / total
            
    return predictions, acc


# train model, return train accuracy and test accuracy
def train(model, trainset, testset, batch_size=4, epochs=10, lr=1e-3, model_name=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    
    # creare dataloaders
    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, batch_size=batch_size)
    
    # set optimizer, loss function
    criterion = nn.CrossEntropyLoss(ignore_index= model.)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
   
    # record acc
    train_acc_list = []
    test_acc_list = []
    max_acc = 0

    for epoch in range(epochs):
        since = time.time()
        
        for data, label in trainloader:
            data = data.to(device)
            label = label.to(device)
            
            # set parameter gradient into zero
            optimizer.zero_grad()

            # forward pass
            output = model(data)

            # backward
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

        # print accuracy each epoch
        _, train_acc = get_predict(model, trainloader)
        train_acc_list.append(train_acc)           
        _, test_acc = get_predict(model, testloader)
        test_acc_list.append(test_acc)
        print ("Epoch: %2s, train accuracy: %8s, test accuracy: %8s - time: %4s\n"%
               (epoch, train_acc, test_acc, (time.time() - since)/60.0))
        # save the parameters when accuracy is higher
        if test_acc > max_acc:
            torch.save(model, "./models/"+model_name)
            max_acc = test_acc
    
    print ("The highest accuracy of %s model is %s"%(model_name, max_acc))
    
    return train_acc_list, test_acc_list