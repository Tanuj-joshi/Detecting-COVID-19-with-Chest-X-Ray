import os
import shutil
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time
from random import random as rnd, seed
from shutil import copyfile
torch.manual_seed(0)
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


# Arrange Images into different subdirectories
# Train-Test Split
def train_test_split(src_dir, dest_dir, val_ratio):
    subdirs = ['train/', 'val/']
    labeldirs = ['normal/', 'viral/', 'covid/']
    for subdir in subdirs:
        for labldir in labeldirs:
            newdir = dest_dir + subdir + labldir
            os.makedirs(newdir, exist_ok=True)
    # seed random number generator
    seed(1)
    # copy dataset images into subdirectories
    images = [x for x in os.listdir(src_dir) if x.lower().endswith('png')]
    for file in images:
        src = src_dir + file
        dst_dir = 'train/'
        if rnd() < val_ratio:
            dst_dir = 'val/'
        if file.startswith('COVID'):
            dst = dest_dir + dst_dir + 'covid/' + file
            copyfile(src, dst)
        elif file.startswith('NORMAL'):
            dst = dest_dir + dst_dir + 'normal/' + file
            copyfile(src, dst)
        elif file.startswith('Viral'):
            dst = dest_dir + dst_dir + 'viral/' + file
            copyfile(src, dst)
    path1 = dest_dir + "train/covid"
    path2 = dest_dir + "train/normal"
    path3 = dest_dir + "train/viral"
    path4 = dest_dir + "val/covid"
    path5 = dest_dir + "val/normal"
    path6 = dest_dir + "val/viral"
    print('Then number of covid images in training data is' ,len(os.listdir(path1)))
    print('Then number of normal images in training data is' ,len(os.listdir(path2)))
    print('Then number of viral images in training data is' ,len(os.listdir(path3)))
    print('Then number of covid images in validation data is' ,len(os.listdir(path4)))
    print('Then number of normal images in validation data is' ,len(os.listdir(path5)))
    print('Then number of viral images in validation data is' ,len(os.listdir(path6)))


class ChestXRayDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('png')]
            print(f'Found {len(images)} {class_name} examples')
            return images
        
        self.images = {}
        self.class_names = ['normal', 'viral', 'covid']
        
        for class_name in self.class_names:
            self.images[class_name] = get_images(class_name)
            
        self.image_dirs = image_dirs
        self.transform = transform
        
    def __len__(self):
        return sum([len(self.images[class_name]) for class_name in self.class_names])
    
    def __getitem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)
    

# Display Images with their labels
def show_images(images, labels, preds, class_names):
    plt.figure(figsize = (8, 4))
    for i, image in enumerate(images):
        plt.subplot(1, 6, i + 1, xticks=[], yticks=[])
        image = image.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0., 1.)
        plt.imshow(image)
        col = 'green' if preds[i] == labels[i] else 'red'
        plt.xlabel(f'{class_names[int(labels[i].numpy())]}')
        plt.ylabel(f'{class_names[int(preds[i].numpy())]}', color = col)
    plt.tight_layout()
    plt.show()


def create_dataset(image_dir, train):
    ''' 
    Image Transformation.
        Convert all the images to the size that, Resnet-18 model (pre-trained) expects.
        Normalize the input images to a specific range (mean and std), which is used 
        when this Resnet-18 model was trained on IMAGENET dataset.
    '''
    if train==True:
        train_transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize(size=(224, 224)),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225]) 
                                                    ])
        train_dirs = {
                    'normal': image_dir +'train/normal',
                    'viral': image_dir +'train/viral',
                    'covid': image_dir +'train/covid'
                    }
        
        val_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]) 
                                            ])
        val_dirs = {
                'normal': image_dir +'val/normal',
                'viral': image_dir +'val/viral',
                'covid': image_dir +'val/covid'
                } 
        train_dataset = ChestXRayDataset(train_dirs, train_transform)
        val_dataset = ChestXRayDataset(val_dirs, val_transform)
        return train_dataset, val_dataset
    else:
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=(224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]) 
                                            ])
        test_dirs = {
                'normal': image_dir +'normal',
                'viral': image_dir +'viral',
                'covid': image_dir +'covid'
                } 
        test_dataset = ChestXRayDataset(test_dirs, test_transform)
        return test_dataset


# Predict model results in Images and display results
def show_preds(model, batches_test, class_names, device):
    model.eval()
    images, labels = next(iter(batches_test))
    print(labels)
    #images = images.unsqueeze(0)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    show_images(images.cpu(), labels.cpu(), preds.cpu(), class_names)


# Predict trained model results in new Test Images and display results
def predict(model, batches_test, class_names, loss_fn, device):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0
    with torch.no_grad():
        for step, (images,labels) in enumerate(batches_test):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
            #show_images(images.cpu(), labels.cpu(), preds.cpu(), class_names)
    # Compute metrics
    total_loss /= (step + 1)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)

    return precision,recall,f1,accuracy, total_loss


# Plot a line graph showing trend of losses (train, val) over time 
def plot_curve(train, val, param):
    steps = len(val)
    print("Steps: ", steps)
    plt.plot(np.arange(1, steps+1,1),train[:steps], label='train '+param)
    plt.plot(np.arange(1,steps+1,1), val[:steps], label='validation '+param)
    plt.xticks(range(1,steps+1,2))
    plt.xlim(1,steps+1)
    plt.xlabel('steps (batches)')
    plt.ylabel(param)
    #plt.title(param)
    plt.legend(loc='upper right')
    filename = param+'_profile.png'
    plt.savefig(filename)
    #plt.show()


# Training and Validating Model using Train and Validation dataset
def train(epochs, batches_train, batches_test, model, optimizer, loss_fn, test_dataset, train_dataset, trained_model, class_names, device):

    print('Starting training..')
    start_time = time.time()
    running_train_loss = []
    running_train_acc = []
    running_train_precision = []
    running_train_recall = []
    running_val_loss = []
    running_val_acc = []
    running_val_precision = []
    running_val_recall = []

    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)
        train_loss = 0.
        model.train()
        train_acc = 0.
        all_preds = []
        all_labels = []
        for train_step, (images, labels) in enumerate(batches_train):
            print('Train_step: ', train_step)
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
        train_precision = precision_score(all_labels, all_preds, average='weighted')
        train_recall = recall_score(all_labels, all_preds, average='weighted')
        train_f1 = f1_score(all_labels, all_preds, average='weighted')
        train_acc = accuracy_score(all_labels, all_preds)
        train_loss /= (train_step + 1)
  
        running_train_loss.append(train_loss)
        running_train_acc.append(train_acc)
        running_train_precision.append(train_precision)
        running_train_recall.append(train_recall)
        print(f'Train loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f} Training time: {(current_time-start_time)//60:.0f} minutes')
        # Metrics calculation
        val_precision,val_recall,val_f1,val_acc,val_loss = predict(model, batches_test, class_names, loss_fn, device)
        
        running_val_loss.append(val_loss)
        running_val_acc.append(val_acc)
        running_val_precision.append(val_precision)
        running_val_recall.append(val_recall)
        current_time = time.time()
        print(f'Val loss: {val_loss:.4f}, Acc: {val_acc:.4f} Training time: {(current_time-start_time)//60:.0f} minutes')

        model.train()
        if val_acc > 0.98:
            print('Performance condition satisfied, Stopping..')
            save_file = trained_model +'epoch'+str(e+1)+'_best_classifier.pt'
            torch.save(model.state_dict(), save_file)
            return
        plot_curve(running_train_loss, running_val_loss, param='loss')
        plot_curve(running_train_acc, running_val_acc, param='acc')
        plot_curve(running_train_precision, running_val_precision, param='precision')
        plot_curve(running_train_recall, running_val_recall, param='recall')
        
        print("Training Complete...")
        save_file = trained_model +'epoch'+str(e+1)+'_classifier.pt'
        torch.save(model.state_dict(), save_file)

