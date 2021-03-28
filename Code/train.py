import torch
import time
import copy
import warnings
warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter


def train_model(model, criterion, optimizer, scheduler, num_epochs, device, train_data_loader, val_data_loader, length_train,length_val):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over train data
            if phase == 'train':
                data = train_data_loader
            else:
                data = val_data_loader

            for inputs, labels in data:

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'): # updating weights
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler:
                scheduler.step()

            if phase == 'train':
                epoch_loss = running_loss / length_train
                epoch_acc = running_corrects.double() / length_train
            else:
                epoch_loss = running_loss / length_val
                epoch_acc = running_corrects.double() / length_val

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def train_FCN(model, optimizer, num_epochs, device, data_loader):

    total_loss = 0
    batch_num = 0

    writer_2 = SummaryWriter()

    for inputs, labels in data_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()
        model.zero_grad()

        outputs = model(inputs)


        values, indices = torch.max(labels, 1)


        loss = torch.nn.functional.cross_entropy(outputs, indices)

        loss.backward()

        optimizer.step()

        print('Loss on batch {}/37 = {}'.format(batch_num, loss.data))
        writer_2.add_scalar('loss FCN-8S', loss, batch_num)

        batch_num+=1
        total_loss +=loss

    return total_loss/(batch_num+1)

