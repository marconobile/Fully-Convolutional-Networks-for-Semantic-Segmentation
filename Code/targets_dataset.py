import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision import transforms
from skimage.transform import resize
from skimage import img_as_bool


class Target_dataset(torch.utils.data.Dataset):

    def __init__(self, picture_dict, labels_dict):  # LIST OF PICTURES TAKE FROM THE FOLDER
        self.picture_dict = picture_dict
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.labels_dict.keys())

    def __getitem__(self, idx):

        name = list(self.labels_dict.keys())[idx]
        h, w = (self.picture_dict[name].shape[1], self.picture_dict[name].shape[2])
        h = h//8
        w = w // 8

        empty_labels = torch.zeros(([43, 800, 1360]))
        label_bkg = torch.ones(([1, 800, 1360]))
        for patch in self.labels_dict[name]:
            if int(patch[4]) != 43:
                empty_labels[int(patch[4]), int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 1
                label_bkg[0, int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 0

        target = torch.cat((empty_labels, label_bkg), dim=0)

        for i in range(target.shape[0]):
            if i == 0:
                resized = img_as_bool(resize(target[i], (h, w)))
                resized_1 = torch.unsqueeze(torch.tensor(resized), dim=0)
            else:
                resized = img_as_bool(resize(target[i], (h, w)))
                asd = torch.unsqueeze(torch.tensor(resized), dim=0)
                resized_1 = torch.cat((resized_1, asd), dim=0)

        target = resized_1
        return self.picture_dict[name], target

def get_labels_dict():
    obs_data = {}
    with open("./gt.txt", "r") as file:
        data = file.readlines()
        for line in data:
            LINE = line.split(';')
            if LINE[0] not in obs_data.keys():
                obs_data[LINE[0]] = [[LINE[1],LINE[2],LINE[3],LINE[4],LINE[5].rstrip('\n')]]
            else:
                obs_data[LINE[0]].append([LINE[1],LINE[2],LINE[3],LINE[4],LINE[5].rstrip('\n')])
    return obs_data


def get_picture_dict(path):
    to_pil = transforms.ToPILImage(mode='RGB')
    resize_pic = transforms.Resize(512, interpolation=2)
    to_tens = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    x_dict = {}
    for _, _, all_pics in os.walk(path, topdown=True):
        for pic in all_pics:
            if pic[0] != '.':
                image = plt.imread(path + pic)
                image = to_pil(image)
                image = resize_pic(image)
                image = to_tens(image)
                x_dict[pic] = normalize(image)

    return x_dict

def get_prediction_for_single_pic(path, model, pic_size = None):
    to_pil = transforms.ToPILImage(mode='RGB')
    if pic_size:
        resize_pic = transforms.Resize(pic_size, interpolation=2)
    to_tens = transforms.ToTensor()
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = plt.imread(path)
    image = to_pil(image)
    if pic_size:
        image = resize_pic(image)
    image = to_tens(image)
    image = normalize(image) # (C x H x W) 3D
    # add the batch dimension
    prediction = model(torch.unsqueeze(image, dim=0))

    return prediction

def get_pixel_acc(path, model, labels_dict, pic_size):  # model, picsie
    IOU = 0
    N = 1
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            current_file = str(os.path.join(root, name))
            if (current_file[-1] == 'm') and (current_file[current_file.rfind('/') + 1:] in list(labels_dict.keys())):
                # get prediction for each picture in the dataset
                prediction = get_prediction_for_single_pic(current_file, model, pic_size = pic_size)
                prediction = torch.squeeze(prediction, dim=0)  # [44, h, w]
                prediction_ = torch.nn.functional.softmax(prediction, dim=0)

                # get target in a dynamic manner (adaptive wrt input shape)
                name = current_file[current_file.rfind('/') + 1:]
                empty_labels = torch.zeros(([43, 800, 1360]))
                label_bkg = torch.ones(([1, 800, 1360]))
                for patch in labels_dict[name]:
                    if int(patch[4]) != 43:
                        empty_labels[int(patch[4]), int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 1
                        label_bkg[0, int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 0
                target = torch.cat((empty_labels, label_bkg), dim=0)
                for i in range(target.shape[0]):
                    if i == 0:
                        resized = img_as_bool(resize(target[i], (prediction_.shape[1], prediction_.shape[2])))
                        resized_1 = torch.unsqueeze(torch.tensor(resized), dim=0)
                    else:
                        resized = img_as_bool(resize(target[i], (prediction_.shape[1], prediction_.shape[2])))
                        asd = torch.unsqueeze(torch.tensor(resized), dim=0)
                        resized_1 = torch.cat((resized_1, asd), dim=0)

                target = resized_1

                # loop thru each box to compute the score
                traffic_signs_in_pic = []
                for box_coords in labels_dict[name]:
                    traffic_signs_in_pic.append(box_coords[-1])
                    if pic_size == 800:
                        new_coords1 = [int(el) // 8 for el in box_coords[:-1]]
                        # here we tweaked the box to ensure considering every GT pixel
                        new_coords1[1] = new_coords1[1] - 2
                        new_coords1[3] = new_coords1[3] + 2
                        new_coords1[0] = new_coords1[0] - 2
                        new_coords1[2] = new_coords1[2] + 2

                    elif pic_size == 512:
                        new_coords1 = [int(el) for el in box_coords[:-1]]
                        new_coords1[1] = ((new_coords1[1] * 870) // 1360)//8
                        new_coords1[3] = ((new_coords1[3] * 870) // 1360)//8
                        new_coords1[0] = ((new_coords1[0] * 512) // 800)//8
                        new_coords1[2] = ((new_coords1[2] * 512) // 800)//8

                    area_target = torch.sum(target[int(box_coords[-1]), new_coords1[1]:new_coords1[3],
                                            new_coords1[0]:new_coords1[2]]).item()

                    our_value = torch.sum(torch.round(prediction_[int(box_coords[-1]), new_coords1[1]:new_coords1[3],
                                                      new_coords1[0]:new_coords1[2]])).item()

                    try:
                        IOU += (our_value / area_target)
                        print('Processing pic: ', name,' ',N,'/1213 : IOU = ',our_value,'/',area_target,' = ', IOU/N)
                        N += 1
                    except:
                        print("Problems with picture: ", name)

    return IOU / N


def get_total_pixel_acc(path, model, labels_dict, pic_size):  # model, picsie
    IOU = 0
    N = 1
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            current_file = str(os.path.join(root, name))
            if (current_file[-1] == 'm') and (current_file[current_file.rfind('/') + 1:] in list(labels_dict.keys())):
                prediction = get_prediction_for_single_pic(current_file, model, pic_size=pic_size)
                prediction = torch.squeeze(prediction, dim=0)  # [44, h, w]
                prediction_ = torch.nn.functional.softmax(prediction, dim=0)

                name = current_file[current_file.rfind('/') + 1:]
                empty_labels = torch.zeros(([43, 800, 1360]))
                label_bkg = torch.ones(([1, 800, 1360]))
                for patch in labels_dict[name]:
                    if int(patch[4]) != 43:
                        empty_labels[int(patch[4]), int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 1
                        label_bkg[0, int(patch[1]): int(patch[3]), int(patch[0]):int(patch[2])] = 0
                target = torch.cat((empty_labels, label_bkg), dim=0)
                for i in range(target.shape[0]):
                    if i == 0:
                        resized = img_as_bool(resize(target[i], (prediction_.shape[1], prediction_.shape[2])))
                        resized_1 = torch.unsqueeze(torch.tensor(resized), dim=0)
                    else:
                        resized = img_as_bool(resize(target[i], (prediction_.shape[1], prediction_.shape[2])))
                        asd = torch.unsqueeze(torch.tensor(resized), dim=0)
                        resized_1 = torch.cat((resized_1, asd), dim=0)

                target = resized_1

                target = torch.sum(target[:-1], dim = 0)
                prediction_ = torch.sum(prediction_[:-1], dim = 0)

                traffic_signs_in_pic = []
                for box_coords in labels_dict[name]:
                    traffic_signs_in_pic.append(box_coords[-1])
                    if pic_size == 800:
                        new_coords1 = [int(el) // 8 for el in box_coords[:-1]]
                        new_coords1[1] = new_coords1[1] - 2
                        new_coords1[3] = new_coords1[3] + 2
                        new_coords1[0] = new_coords1[0] - 2
                        new_coords1[2] = new_coords1[2] + 2
                    elif pic_size == 512:

                        new_coords1 = [int(el) for el in box_coords[:-1]]
                        new_coords1[1] = ((new_coords1[1] * 870) // 1360) // 8
                        new_coords1[3] = ((new_coords1[3] * 870) // 1360) // 8
                        new_coords1[0] = ((new_coords1[0] * 512) // 800) // 8
                        new_coords1[2] = ((new_coords1[2] * 512) // 800) // 8


                    area_target = torch.sum(target[new_coords1[1]-2:new_coords1[3]+2,
                                            new_coords1[0]-2:new_coords1[2]+2]).item()

                    our_value = torch.sum(torch.round(prediction_[new_coords1[1]:new_coords1[3],
                                                      new_coords1[0]:new_coords1[2]])).item()

                    try:
                        IOU += (our_value / area_target)
                        print('Processing pic: ', name, ' ', N, '/1213 : IOU = ', our_value, '/', area_target, ' = ',
                              IOU / N)
                        N += 1
                    except:
                        print("Problems with picture: ", name)

    return IOU / N

