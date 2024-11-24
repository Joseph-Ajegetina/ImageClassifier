
import argparse
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
from collections import OrderedDict
from PIL import ImageFile
from torchvision.models import VGG13_Weights, ResNet18_Weights
ImageFile.LOAD_TRUNCATED_IMAGES = True



# Initialization
parser = argparse.ArgumentParser(description='Model for training flowers dataset using pytorch',
								 usage='''
        python train.py (data set shall be initially extracted to the 'flowers' directory)
        python train.py data_dir (data set shall be initially extracted to the 'data_dir' directory)
        python train.py data_dir --save_dir save_directory (set directory to save checkpoints)
        python train.py data_dir --arch "vgg13" (choose between vgg13 and resnet18)
        python train.py data_dir --learning_rate 0.01 --hidden_units [1024, 512, 256] --epochs 20 (set hyperparameters)
        ''',
prog='train')


## Adding argument for data directory with default of flowers and help messages
parser.add_argument('data_directory', action="store", nargs='?', default="flowers", help="The dataset directory for the model")

## Adding argument for save directory 
parser.add_argument('--save_dir', action="store", default="", dest="save_directory", help="Checkpoint directory for train model state")

## Adding architecture argument with choices
parser.add_argument('--arch', action="store", default="resnet18", choices=['vgg13', 'resnet18'],
					 help="you can only choose vgg13 or resnet", dest="architecture")

## Argument for setting hyperparameters
parser.add_argument('--learning_rate', action="store", default="0.003", type=float, help="This sets the learning rate",
					 dest="learning_rate")
parser.add_argument('--hidden_units', action="store", nargs=3, default=[1024, 512, 256], type=int, help="Enter the hidden units parameters",
					 dest="hidden_units")
parser.add_argument('--epochs', action="store", default=3, type=int, help="This sets the epochs of training", dest="epochs")

## Set GPU
parser.add_argument('--gpu', action="store_true", default=False, help="Use GPU", dest="gpu")

## Parsing arguments to get their value
args = parser.parse_args()

data_dir =  args.data_directory
save_dir =  args.save_directory
arch=  args.architecture
lr = args.learning_rate
hidden_units = args.hidden_units
epochs = args.epochs
                                                                        
# Condition for selecting GPU if it's available
                                                                        
if args.gpu and torch.cuda.is_available(): 
	gpu = args.gpu
                                                                        
# Warn user of inability to gpu if unavailable
elif args.gpu:
	gpu = False
	print('CPU will be used as GPU is not available on this machine.')

else:
	gpu = args.gpu

# Printing the configuration details from args
print(f'''Data directory: {data_dir} \nSave directory: {save_dir} \nArchitecture: {arch}.
\nLearning rate: {lr} 
 \nHidden Units: {hidden_units}
 \nEpochs: {epochs} 
 \nUsing GPU: {gpu}''')



## set data directory locations

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

## Define transforms for the training, validation, and testing sets
                                                                        
data_transforms = {'train':transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                   'valid':transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                                    
                   'test':transforms.Compose([transforms.Resize(255),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])]),
                  }

dataset_categories = ['train', 'valid', 'test']
                                                                        
image_datasets = {category:datasets.ImageFolder(os.path.join(data_dir, category), transform=data_transforms[category]) 
                  for category in dataset_categories}

dataloaders = {category:torch.utils.data.DataLoader(image_datasets[category], batch_size=64, shuffle=True)
                  for category in dataset_categories}
## Build and train the network

# download architecture
if arch == 'vgg13':
	print('Downloading VGG-13 pretrained model ...')
	model = models.vgg13(weights=VGG13_Weights.DEFAULT)                                                                     

if arch == 'resnet18':
	print('Downloading RESNET18 ...')
	model = models.resnet18(weights=ResNet18_Weights.DEFAULT)


# detting device based on user args received
device = torch.device("cuda" if gpu else "cpu")


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

if arch.upper()  == 'VGG13':                                                                     
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units[0]),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units[0], hidden_units[1]),
                                     nn.ReLU(),                               
                                     nn.Linear(hidden_units[1], 102),
                                     nn.LogSoftmax(dim=1))
                                                                       
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
                                                                        
                                                                        
if arch.upper() == 'RESNET18':
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(512, 256)),
                          ('relu1', nn.ReLU()),
                          ('dpout1', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    model.fc = classifier
    optimizer = optim.Adam(model.fc.parameters(), lr=lr)


model.to(device)
if torch.cuda.is_available() and device == 'gpu':
   model.cuda()

                                                                        

criterion = nn.NLLLoss()

                                                                        
print('\n\n\nTraining the model................... ')
print('Do not turn off your computer........ ')
print('..................................... ')

steps = 0
running_loss = 0
print_every = 10
for epoch in range(epochs):
    trainloader = dataloaders['train']
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                validloader = dataloaders['valid']
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                  f"Validation accuracy: {accuracy/len(validloader):.3f}")
            running_loss = 0
            model.train()
print()
print('\nValidating the model......................................................... ')
print('Do not turn off your computer..................................................\n................................................................................. ')

## Do validation on the test set

# number of total true classifications
total = 0
# number of total images tested
total_length = 0
# total accuracy for test dataset (calculated so far)
total_accuracy = 0
# batch number
batch = 0
testloader = dataloaders['test']
for inputs, labels in testloader:
    batch += 1    
    # Move input and label tensors to the default device
    inputs, labels = inputs.to(device), labels.to(device)
   
    
    accuracy = 0
    model.eval()
    with torch.no_grad():

        logps = model.forward(inputs)
                 
        # Calculate accuracy
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        matches = top_class == labels.view(*top_class.shape)
        
        # this is batch accuracy
        accuracy += torch.mean(matches.type(torch.FloatTensor)).item()
                                                                        
        # number of true classifications so far
        total += torch.sum(equals)
        # number of classification attempts so far
        total_length += len(equals)
        
        total_accuracy = total.item()/total_length
    print(f"Batch {batch}.. "
          f"Accuracy: {accuracy*100:.3f}%.. "
          f"Total Accuracy: {total_accuracy*100:.3f}%")

    model.train()

                                                                        
## Save the checkpoint 
print('\nSaving the model..................... ')

if save_dir:
	if not os.path.exists(save_dir):
		os.mkdir(arg_save_dir)
		print("Directory " , save_dir ,  " has been created for saving checkpoints")
	else:
		print("Directory " , save_dir ,  " allready exists for saving checkpoints")
	save_dir = save_dir + '/checkpoint.pth'
else:
	save_dir = 'checkpoint.pth'

print()

model.class_to_idx = image_datasets['train'].class_to_idx
                                                                        
checkpoint = {'input_size': 25088,
              'output_size': 102,
              'epoch': epochs,
              'classifier': model.classifier if arch=='vgg13' else model.fc,
              'optimizer_state_dict': optimizer.state_dict(),
              'class_to_idx': model.class_to_idx,
              'learning_rate': lr,
              'model_state_dict': model.state_dict()}

torch.save(checkpoint, save_dir)

print('Model saved')
print()

## this part is for validating saved checkpoint

def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    epoch = checkpoint['epoch']
    
    return model, optimizer, input_size, output_size, epoch 

print('\n\nValidating checkpoint and loading................ ')

my_model, my_optimizer, input_size, output_size, epoch  = load_checkpoint(save_dir)


print('Loaded saved model')
print(my_model)