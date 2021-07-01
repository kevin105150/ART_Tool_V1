"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random

import art.attacks.evasion as art_att
from art.estimators.classification import PyTorchClassifier
from art.utils import load_mnist
import model as modl
import os

# Step 1: Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Step 1a: Swap axes to PyTorch's NCHW format
x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)

# Load Model and Optimizer function
def Load_Model(filename, model, optimizer):
    PATH = os.path.dirname(os.path.abspath(__file__)) + "/"
    PATH_model = PATH + filename + ".model"
    PATH_optim = PATH + filename + ".optimizer"
    isload = False
    if os.path.isfile(PATH_model) and os.path.isfile(PATH_optim):
        model.load_state_dict(torch.load(PATH_model))
        optimizer.load_state_dict(torch.load(PATH_optim))
        model.eval()
        print('Load model:', PATH_model)
        isload = True
    return model, optimizer, isload

# Step 2: Create the model
model_select = int(input("Please select model (1:LeNet5, 2:AlexNet, 3:CNN):"))

if model_select == 1:
    model = modl.LeNet5()
    text = 'LeNet5_'
    print('Use LeNet5')
elif model_select == 2:
    model = modl.AlexNet()
    text = 'AlexNet_'
    print('Use AlexNet')
elif model_select == 3:
    model = modl.CNN()
    text = 'CNN_'
    print('Use CNN')

# Select attack function
attack_select = int(input("Please select Attack (1:FGSM, 2:BIM, 3:PGD, 4:C&W L2, 5:C&W Linf):"))
if attack_select == 1: 
    attack_func = [True, False, False, False, False]
    print('Use FGSM')
    att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
elif attack_select == 2: 
    attack_func = [False, True, False, False, False]
    print('Use BIM')
    att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
elif attack_select == 3: 
    attack_func = [False, False, True, False, False]
    print('Use PGD')
    att_range = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
elif attack_select == 4: 
    attack_func = [False, False, False, True, False]
    print('Use C&W L2')
    att_range = [0, 2, 4, 6, 8, 10, 12, 14, 16]
elif attack_select == 5: 
    attack_func = [False, False, False, False, True]
    print('Use C&W Linf')
    att_range = [0, 2, 4, 6, 8, 10, 12, 14, 16]
else:
    attack_func = [False, False, False, False, False]  #attack_func = [FGSM, BIM, PGD, C&W L2, C&W Linf]
    att_range = []

# Basic parameter
epoch_num = 10
batch_size = 8
max_iter = 20

# Step 2a: Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=1e-3, momentum=0.9)

# load model, optimizer
model, optimizer, isload = Load_Model(text, model, optimizer)

# Step 3: Create the ART classifier
classifier = PyTorchClassifier(
    model=model,
    clip_values=(min_pixel_value, max_pixel_value),
    loss=criterion,
    optimizer=optimizer,
    input_shape=(1, 28, 28),
    nb_classes=10,
    device_type='gpu'
)

print(classifier.device)

# Step 4: Train the ART classifier
print('Start training ...')
if isload == False:
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=epoch_num)
    classifier.save(filename=text, path=os.path.dirname(os.path.abspath(__file__)))
print('Finish training ...')

# Step 5: Evaluate the ART classifier on benign test examples
predictions = classifier.predict(x_test)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
print("Accuracy on benign test examples: {}%".format(accuracy * 100))
np.savetxt(text + 'origin_Accuracy.csv', [accuracy*100], delimiter=",")

# Generate adversarial test examples
def attack(eps, max_iter, attack_func):
    
    out_accuracy = []
    
    # FGSM
    if attack_func[0] == True:
        print('Running FGSM, eps = ', eps)
        attack_FGSM = art_att.FastGradientMethod(estimator=classifier, eps=eps)
        x_test_FGSM = attack_FGSM.generate(x=x_test)
        
        predictions = classifier.predict(x_test_FGSM)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on FGSM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # BIM
    if attack_func[1] == True:
        print('Running BIM, eps = ', eps)
        attack_BIM = art_att.BasicIterativeMethod(estimator=classifier, eps=eps, max_iter=max_iter)
        x_test_BIM = attack_BIM.generate(x=x_test)
        
        predictions = classifier.predict(x_test_BIM)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on BIM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # PGD
    if attack_func[2] == True:
        print('Running PGD, eps = ', eps)
        attack_PGD = art_att.ProjectedGradientDescentPyTorch(estimator=classifier, eps=eps, max_iter=max_iter)
        x_test_PGD = attack_PGD.generate(x=x_test)
        
        predictions = classifier.predict(x_test_PGD)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on PGD adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # C&W L2
    if attack_func[3] == True:
        print('Running C&W L2, conf = ', eps)
        attack_CW_L2 = art_att.CarliniL2Method(classifier=classifier, confidence=eps, max_iter=max_iter)
        x_test_CW_L2 = attack_CW_L2.generate(x=x_test)
        
        predictions = classifier.predict(x_test_CW_L2)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on C&W L2 adversarial test examples: {}%".format(accuracy * 100))
        print('eps=',eps)
        out_accuracy.append(accuracy * 100)
    
    # C&W Linf
    if attack_func[4] == True:
        print('Running C&W Linf, conf = ', eps)
        attack_CW_Linf = art_att.CarliniLInfMethod(classifier=classifier, confidence=eps, max_iter=max_iter)
        x_test_CW_Linf = attack_CW_Linf.generate(x=x_test)
        
        predictions = classifier.predict(x_test_CW_Linf)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
        print("Accuracy on C&W Linf adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    return out_accuracy

# Step 6: Generate adversarial test examples
out_accuracy = []

for eps in att_range:
    out_list = attack(eps, max_iter, attack_func)
    out_accuracy.append(out_list)
    
# Save Data
att_text = ''
if attack_func[0] == True: att_text = att_text + '_FGSM'
if attack_func[1] == True: att_text = att_text + '_BIM'
if attack_func[2] == True: att_text = att_text + '_PGD'
if attack_func[3] == True: att_text = att_text + '_C&W_L2'
if attack_func[4] == True: att_text = att_text + '_C&W_Linf'
np.savetxt(text + att_text + '_Accuracy.csv', out_accuracy, delimiter=",")