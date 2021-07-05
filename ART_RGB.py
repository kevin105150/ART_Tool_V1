# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 15:38:15 2021

@author: csyu
"""

"""
The script demonstrates a simple example of using ART with PyTorch. The example train a small model on the MNIST dataset
and creates adversarial examples using the Fast Gradient Sign Method. Here we use the ART classifier to train the model,
it would also be possible to provide a pretrained model to the ART classifier.
The parameters are chosen for reduced computational requirements of the script and not optimised for accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
import art.attacks.evasion as art_att
from art.estimators.classification import PyTorchClassifier
from art.data_generators import PyTorchDataGenerator
from art.utils import load_mnist

# import model
import ResNet
import googlenet
import VGG

import os

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_mnist()

# Swap axes to PyTorch's NCHW format
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

# Create the model
# Pytorch 1.6 and CUDA 10.2
print("Please select model (1:GoogLeNet, 2:VGG19, 3:ResNeXt101):")
model_select = int(input())

if model_select == 1:
    model = googlenet.GoogLeNet()
    text = 'GoogLeNet_'
    print('Use GoogLeNet')
elif model_select == 2:
    model = VGG.vgg19()
    text = 'VGG19_'
    print('Use VGG19')
elif model_select == 3:
    model = ResNet.resnext101_32x8d()
    text = 'ResNeXt101_'
    print('Use ResNeXt101')

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
batch_size = 32
max_iter = 20

torch.backends.cudnn.benchmark = False


# Data transform
data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

class MNISTDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x = x
        self.y = y
        self.transform = transform
     
    def __len__(self):
        return int(len(self.y))
    
    def __getitem__(self, index):
        image = Image.fromarray(self.x[index][0]*255).convert('RGB')
        image = self.transform(image)
        return image, self.y[index]

# Train Data loader
train_set = MNISTDataset(x_train, y_train, data_transform)
data_loader = DataLoader(dataset=train_set, batch_size=batch_size)
data_Generator = PyTorchDataGenerator(iterator=data_loader, size=int(len(y_train)), batch_size=batch_size)

# Test Data transform
test_num = int(x_test.shape[0])
x_test_3d = np.zeros((test_num, 3, 256, 256))
y_test_3d = np.zeros((test_num, 10))

for i in tqdm(range(0,test_num)):
    image = Image.fromarray(x_test[i][0]*255).convert('RGB')
    image = data_transform(image)
    x_test_3d[i] = image
    y_test_3d[i] = y_test[i]

x_test_3d = np.array(x_test_3d)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=1e-4, momentum=0.9)

# load model, optimizer
model, optimizer, isload = Load_Model(text, model, optimizer)

# Create the ART classifier
classifier = PyTorchClassifier(
    model = model,
    clip_values = (min_pixel_value, max_pixel_value),
    loss = criterion,
    optimizer = optimizer,
    input_shape = (3, 256, 256),
    nb_classes = 10,
    device_type = 'gpu',
)

print(classifier.device)

# Train the ART classifier
print('Start training ...')
if isload == False:
    classifier.fit_generator(data_Generator, nb_epochs=epoch_num)
    classifier.save(filename=text, path=os.path.dirname(os.path.abspath(__file__)))
print('Finish training ...')

# Evaluate the ART classifier on benign test examples
predictions = []

for x_test_one in tqdm(x_test_3d):
    predictions_test = classifier.predict([x_test_one])
    predictions.append(predictions_test[0])

predictions = np.array(predictions)
accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
print("\nAccuracy on benign test examples: {}%".format(accuracy * 100))
np.savetxt(text + 'origin_Accuracy.csv', [accuracy*100], delimiter=",")

# Generate adversarial test examples
def attack(eps, max_iter, attack_func):
    
    out_accuracy = []
    
    # FGSM
    if attack_func[0] == True:
        print('Running FGSM, eps = ', eps)
        attack_FGSM = art_att.FastGradientMethod(estimator=classifier, eps=eps, batch_size=batch_size)
        
        predictions = []
        for x_test_one in tqdm(x_test_3d):
            x_test_FGSM = attack_FGSM.generate(x=[x_test_one])
            predictions_test = classifier.predict(x_test_FGSM)
            predictions.append(predictions_test[0])
        
        predictions = np.array(predictions)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
        print("\nAccuracy on FGSM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # BIM
    if attack_func[1] == True:
        attack_BIM = art_att.BasicIterativeMethod(estimator=classifier, eps=eps, max_iter=max_iter, batch_size=batch_size)
        
        predictions = []
        for x_test_one in tqdm(x_test_3d):
            x_test_BIM = attack_BIM.generate(x=[x_test_one])
            predictions_test = classifier.predict(x_test_BIM)
            predictions.append(predictions_test[0])
            print('Running BIM, eps = ', eps)
        
        predictions = np.array(predictions)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
        print("\nAccuracy on BIM adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # PGD
    if attack_func[2] == True:
        attack_PGD = art_att.ProjectedGradientDescentPyTorch(estimator=classifier, eps=eps, max_iter=max_iter, batch_size=batch_size)
        
        predictions = []
        for x_test_one in tqdm(x_test_3d):
            x_test_PGD = attack_PGD.generate(x=[x_test_one])
            predictions_test = classifier.predict(x_test_PGD)
            predictions.append(predictions_test[0])
            print('Running PGD, eps = ', eps)
        
        predictions = np.array(predictions)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
        print("\nAccuracy on PGD adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # C&W L2
    if attack_func[3] == True:
        attack_CW_L2 = art_att.CarliniL2Method(classifier=classifier, confidence=eps, max_iter=max_iter, batch_size=batch_size)
        
        predictions = []
        for i in tqdm(range(0,x_test_3d.shape[0])):
            x_test_CW_L2 = attack_CW_L2.generate(x=[x_test_3d[i]])
            predictions_test = classifier.predict(x_test_CW_L2)
            predictions.append(predictions_test[0])
            print('Running C&W L2, conf = ', eps)
        
        predictions = np.array(predictions)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
        print("\nAccuracy on C&W L2 adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
    
    # C&W Linf
    if attack_func[4] == True:
        attack_CW_Linf = art_att.CarliniLInfMethod(classifier=classifier, confidence=eps, max_iter=max_iter, batch_size=batch_size)
        
        predictions = []
        for i in tqdm(range(0,x_test_3d.shape[0])):
            x_test_CW_Linf = attack_CW_Linf.generate(x=[x_test_3d[i]])
            predictions_test = classifier.predict(x_test_CW_Linf)
            predictions.append(predictions_test[0])
            print('Running C&W Linf, conf = ', eps)
        
        predictions = np.array(predictions)
        accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test_3d, axis=1)) / len(y_test_3d)
        print("\nAccuracy on C&W Linf adversarial test examples: {}%".format(accuracy * 100))
        out_accuracy.append(accuracy * 100)
        
    return out_accuracy

# Main
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
