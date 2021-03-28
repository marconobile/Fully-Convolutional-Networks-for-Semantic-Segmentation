import warnings
warnings.filterwarnings("ignore")
import torch
from random import shuffle
import os
import random
import time
import copy
from torchvision import datasets, models, transforms, utils
from random import shuffle
import numpy as np
from custom_utils import *
from train import *
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from skimage.transform import resize
import seaborn as sns
from custom_models import *
from targets_dataset import *
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    ### seed for replicability: ###
    manualSeed = 23
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if are using GPU:
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    root ='./train_templates/'
    cwd = os.getcwd()

    # Data augmentation is a good practice for the train set
    # Here, we randomly crop the image to 224x224 and
    # randomly flip it horizontally.

    data_transforms = transforms.Compose([transforms.RandomResizedCrop(224),
                                          transforms.RandomGrayscale(p=0.1),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
                                          transforms.RandomAffine(degrees=45, translate=(0,1), scale=(.1,1)),
                                          transforms.RandomPerspective(distortion_scale=0.5, p=0.5, interpolation=3),
                                          transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0),
                                                                      ratio=(0.75, 1.3333333333333333),
                                                                      interpolation=2),

                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                               std= [0.229, 0.224, 0.225])])

    template_data = datasets.ImageFolder(root = root, transform = data_transforms)

    # cast data to list and shuffle
    data = []
    for el in template_data:
       data.append(el)
    shuffle(data)

    # split data in train and val set
    train_dataset = data[0:int(0.85*len(data))]
    val_dataset = data[int(0.85 * len(data)):]

    print(f'Len of original dataset {len(data)}, len of training data {len(train_dataset)}, len of val data {len(val_dataset)}')

    # create the respective data loaders for the train function
    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle= True)
    val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size = 32, shuffle= True)

    # Load the model from torchvision.models, pretrained = true s.t. we get the weights
    vgg16_updated = models.vgg16(pretrained=True)
    # print('VGG16 imported with pretrained weights!',vgg16_updated)
    # print('The loaded network is in training mode: ', vgg16_updated.training) # 25088: input fully connected

    new_sequential = nn.Sequential(
        nn.Linear(in_features=25088, out_features=2048, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=2048, out_features=512, bias=True),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.5, inplace=False),
        nn.Linear(in_features=512, out_features=44, bias=True))

    vgg16_updated.classifier = new_sequential
    # print('Network modified', vgg16_updated)

    #########################################   FINE TUNING OF THE VGG  #########################################
    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Observe that ALL parameters are being optimized
    optimizer_ft = optim.Adam(vgg16_updated.parameters(), lr=1e-5, weight_decay=5e-4)

    num_epochs = 26

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=num_epochs//2, gamma=0.1)

    vgg16_updated.to(device)

    ########################
    ## DECOMMENT TO TRAIN ##
    ########################
    ## the following function returns a model
    # model_updated = train_model(vgg16_updated, criterion = criterion, optimizer = optimizer_ft, scheduler = None,
    #                 num_epochs = num_epochs,
    #                 device = device,
    #                 train_data_loader = train_data_loader,
    #                 val_data_loader = val_data_loader,
    #                 length_train = len(train_dataset),length_val = len(val_dataset))
    ## once we are done with training, we save the model
    # torch.save(model_updated.state_dict(), cwd + '/weights.pt')

#########################################   MOVING TO FCN  #########################################


# LOAD THE FINE TUNED MODEL:
# 1) Let's grab again the pre-trained VGG:
vgg16_NEW = VGG_custom(vgg16_updated.features)

# 2) Let's add again our newly constructed Fully connected classifier
vgg16_NEW.classifier = new_sequential

# 3) load the fine tuned weights
vgg16_NEW.to(device)

if torch.cuda.is_available():
    vgg16_NEW.load_state_dict(torch.load(cwd + '/weights.pt', map_location=torch.device("cuda:0")))
else:
    vgg16_NEW.load_state_dict(torch.load(cwd + '/weights.pt', map_location=torch.device('cpu')))

# so now we have our finetuned VGG
# so now we need to define the FullyConv layers
fully_convolution = FCN8s(n_class=44)

# LOCK all params of vgg finetuned
for param in vgg16_NEW.parameters():
    param.requires_grad = False

vgg16_NEW.classifier = fully_convolution

FCN_final = copy.deepcopy(vgg16_NEW)
del vgg16_NEW
del vgg16_updated
torch.cuda.empty_cache()

FCN_final.to(device)
print(FCN_final)

#########################################   TRAINING  FCN  #########################################
# 1) Let's load a single picture from the train_set
path = './FullIJCNN2013/'
print('Loading the dataset:')
dataset_targets = Target_dataset(get_picture_dict(path), get_labels_dict())
targets_data_loader = torch.utils.data.DataLoader(dataset_targets, batch_size = 20, shuffle = True)

print('Starting with training of the FCN block:')
optimizer_ft_2 = optim.Adam(FCN_final.parameters(), lr = 3e-4, weight_decay = 5e-4)
max_num_epochs = 24

########################
## DECOMMENT TO TRAIN ##
########################

# writer_2 = SummaryWriter()
# for i in range(max_num_epochs):
#     print('Starting epoch num', i)
#     loss = train_FCN(model = FCN_final, optimizer = optimizer_ft_2, num_epochs = i, device = device, data_loader = targets_data_loader)
#     writer_2.add_scalar('loss FCN-8S', loss, i)
#
#     print('Loss on epoch {}/{}: {}'.format(i, max_num_epochs, loss))
# print('Saving final model...')
# torch.save(FCN_final.state_dict(), './weights_FCN.pt')
# writer_2.close()


#########################################   VISUALIZING AND VALIDATING RESULTS OF FCN  #########################################
# let's load the model
FCN_final.to(device)
if torch.cuda.is_available():
    FCN_final.load_state_dict(torch.load('./weights_FCN_final.pt', map_location=torch.device("cuda:0")))
else:
    FCN_final.load_state_dict(torch.load('./weights_FCN_final.pt', map_location=torch.device('cpu')))
FCN_final.eval()

####  GET PREDICTION FOR A SINGLE PICTURE:

path = '/Users/marconobile/Desktop/CVPR_PRJ/FullIJCNN2013/00097.ppm'
prediction = get_prediction_for_single_pic(path, FCN_final, pic_size = 800)

## decomment for PLOT HEAMAPS
prediction = torch.squeeze(prediction, dim = 0)
prediction = torch.nn.functional.softmax(prediction, dim =0).detach().numpy()
best = prediction.argmax(axis=0)

image = Image.open(path)
image = image.resize((best.shape[1], best.shape[0]))

fig, ax = plt.subplots(figsize=(8,8))
ax.imshow(image)
sns.heatmap(best, alpha=0.5, ax=ax, cbar=False,
           xticklabels=[], yticklabels=[], cmap='prism')
plt.show()

## decomment for PLOT TEMPLATE PREDICTIONS
#
# prediction = torch.squeeze(prediction, dim = 0)
# prediction = torch.nn.functional.softmax(prediction, dim =0)
# for chan in range(prediction.shape[0]):
#     plt.title('TEMPLATE PREDICTIONS_'+str(chan))
#     imgplot = plt.imshow(prediction[chan].detach().numpy())
#     plt.show()

## decomment for PLOT LABELS
# ground_t = dataset_targets[43][1] #[1,44,64,108]  # 12 36 35
# ground_t = torch.squeeze(ground_t,dim = 0)
# for chan in range(ground_t.shape[0]):
#     plt.title('Ground trhuth'+str(chan))
#     imgplot = plt.imshow(ground_t[chan].numpy())
#     plt.show()

## decomment for PLOT FEATURE MAPS
# prediction_1 = torch.squeeze(prediction,dim = 0) # 44, 64, 108
# prediction_1 = torch.sum(prediction_1, dim = 0)
# plt.title('coarse feature map')
# imgplot_1 = plt.imshow(prediction_1.detach().numpy())
# plt.show()

#####   Get pixel accuracy, with and without classes:
# path = '/Users/marconobile/Desktop/CVPR_PRJ/FullIJCNN2013/'
# labels_dict =  get_labels_dict()
# value = get_pixel_acc(path, FCN_final, labels_dict, pic_size = 512) # values aviable: 800, 512
# print(value)
#
# value1 = get_total_pixel_acc(path, FCN_final, labels_dict, pic_size = 800)
# print(value1)
quit()


##############################################################
