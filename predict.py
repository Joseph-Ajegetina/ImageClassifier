
import argparse
import torch
import json
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.models import VGG13_Weights, ResNet18_Weights

## A function that loads a checkpoint and rebuilds the model

def load_checkpoint(filepath):
	model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

	
	checkpoint = torch.load(filepath)
	lr = checkpoint['learning_rate']
	model.fc = checkpoint['classifier']
	model.load_state_dict(checkpoint['model_state_dict'])
	model.class_to_idx = checkpoint['class_to_idx']
	optimizer = optim.Adam(model.fc.parameters(), lr=lr)
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

	input_size = checkpoint['input_size']
	output_size = checkpoint['output_size']
	epoch = checkpoint['epoch']


	return model, optimizer, input_size, output_size, epoch


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # loading the image from the path
    loaded_image = Image.open(image)
    transform_image = transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    transformed_image_tensor = transform_image(loaded_image)
    return transformed_image_tensor


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    image_processed = process_image(image_path).to(device)

    # Getting the image for 1 format
    image_processed.unsqueeze_(0)
 
    
    log_ps = model.forward(image_processed)
    ps = torch.exp(log_ps)
    top_ps,top_class = ps.topk(5)

    # Since we have labels and probs of prediction we need a table to look up the idx from the class
    idx_to_class = {}
    
    # creating a class -> idx lookup 
    for key, value in model.class_to_idx.items():
        idx_to_class[value] = key

    top_labels_numpy = top_class[0].cpu().numpy()

    top_labels = []
    for label in top_labels_numpy:
        top_labels.append(int(idx_to_class[label]))

    # since we have the idx of the top labels we can look up name from cat_to_name
    top_flowers = [cat_to_name[str(label)] for label in top_labels]
    
    return top_ps, top_labels, top_flowers

# Initialization using question configurations
parser = argparse.ArgumentParser(description="This program predicts flowers' names from their images",
								 usage='''
        needs a saved checkpoint
        python predict.py ( use default image 'flowers/test/1/image_06743.jpg' and root directory for checkpoint)
        python predict.py /path/to/image checkpoint (predict the image in /path/to/image using checkpoint)
        python predict.py --top_k 3 (return top K most likely classes)
        python predict.py --category_names cat_to_name.json (use a mapping of categories to real names)
        python predict.py --gpu (use GPU for inference)''',
								 prog='predict')

## Get path of image
parser.add_argument('path_to_image', action="store", nargs='?', default='flowers/test/4/image_05637.jpg', help="path/to/image")

parser.add_argument('path_to_checkpoint', action="store", nargs='?', default='checkpoint.pth', help="path/to/checkpoint")
## set top_k
parser.add_argument('--top_k', action="store", default=1, type=int, help="enter number of guesses", dest="top_k")
## Choose json file:
parser.add_argument('--category_names', action="store", default="cat_to_name.json", help="get json file", dest="category_names")
## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Select GPU", dest="gpu")

## Get the arguments
args = parser.parse_args()

path_to_image =  args.path_to_image
path_to_checkpoint = args.path_to_checkpoint
top_k =  args.top_k
category_names =  args.category_names
# Use GPU if it's selected by user and it is available
if args.gpu and torch.cuda.is_available(): 
	gpu = args.gpu
# if GPU is selected but not available use CPU and warn user
elif args.gpu:
	gpu = False
	print('GPU is not available, will use CPU...')

else:
	gpu = args.gpu

# Use GPU if it's selected by user and it is available
device = torch.device("cuda" if gpu else "cpu")
print('Will use {} for prediction...'.format(device))

print("\nPath of image: {} \nPath of checkpoint: {} \nTopk: {} \nCategory names: {} ".format(path_to_image, path_to_checkpoint, top_k, category_names))
print('GPU: ', gpu)
print()

## Label mapping
with open(category_names, 'r') as f:
    cat_to_name = json.load(f)

## Loading model
print('Loading model........................ ')
print()

model, my_optimizer, input_size, output_size, epoch  = load_checkpoint(path_to_checkpoint)

if(gpu):
    model.cuda()
else:
    model.cpu()

        
model.eval()

# Make prediction
top_probs, top_labels, top_flowers = predict(path_to_image, model) 
top_probs = top_probs[0].detach().cpu().numpy()

flower_num = path_to_image.split('/')[2]
title_ = cat_to_name[flower_num]

print(f'actual label {title_}\n')
for counter in range(top_k):
     print(f'{top_flowers[counter]}\t{top_probs[counter]:.3f} %')
