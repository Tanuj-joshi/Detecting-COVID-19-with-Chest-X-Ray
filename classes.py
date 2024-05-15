
import os
import shutil
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import time

torch.manual_seed(0)

# Arrange Images into different subdirectories
def train_test_split():
    # Directories for Train and Test Image Data
    class_names = ['normal', 'viral', 'covid']
    root_dir = 'COVID-19_Radiography_Database'
    source_dirs = ['NORMAL', 'Viral Pneumonia', 'COVID-19']

    if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
        os.mkdir(os.path.join(root_dir, 'test'))

        for i, d in enumerate(source_dirs):
            os.rename(os.path.join(root_dir, d), os.path.join(root_dir, class_names[i]))   

        for c in class_names:
            os.mkdir(os.path.join(root_dir, 'test', c))
    
        for c in class_names:
            images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
            selected_images = random.sample(images, 30)
            for image in selected_images:
                source_path = os.path.join(root_dir, c, image)
                target_path = os.path.join(root_dir, 'test', c, image)
                shutil.move(source_path, target_path)


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


# Predict model results in Images and display results
def show_preds(model, batches_test, class_names, device):
    model.eval()
    images, labels = next(iter(batches_test))
    #images = images.unsqueeze(0)
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)
    show_images(images.cpu(), labels.cpu(), preds.cpu(), class_names)


# Predict trained model results in new Test Images and display results
def predict(model, batches_test, class_names, device):
    model.eval()
    for batch in batches_test:
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        show_images(images.cpu(), labels.cpu(), preds.cpu(), class_names)


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
    plt.show()


# Training and Validating Model using Train and Validation dataset
def train(epochs, batches_train, batches_test, model, optimizer, loss_fn, test_dataset, train_dataset, device):

    print('Starting training..')
    start_time = time.time()
    running_train_loss = []
    running_val_loss = []
    running_train_acc = []
    running_val_acc = []

    for e in range(0, epochs):
        print('='*20)
        print(f'Starting epoch {e + 1}/{epochs}')
        print('='*20)

        train_loss = 0.
        val_loss = 0.

        model.train()
        train_acc = 0.
    
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
            train_acc += sum((preds.cpu() == labels.cpu()).numpy())
        train_loss /= (train_step + 1)
        #print(f'Training loss: {train_loss:.4f}') 
        running_train_loss.append(train_loss)
        #print('running_train_loss: ', running_train_loss)
        train_acc = train_acc / len(train_dataset)
        current_time = time.time()
        print(f'Train loss: {train_loss:.4f}, Train_Acc: {train_acc:.4f} Training time: {(current_time-start_time)//60:.0f} minutes')
        running_train_acc.append(train_acc)
        #print('running_train_acc: ', running_train_acc)
        #print('Evaluating at step', train_step)
        val_acc = 0.
        model.eval()
        for val_step, (images, labels) in enumerate(batches_test):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_acc += sum((preds.cpu() == labels.cpu()).numpy())
        val_loss /= (val_step + 1)
        #print('val_loss: ', val_loss)
        running_val_loss.append(val_loss)
        #print('running_val_loss: ', running_val_loss)
        val_acc = val_acc / len(test_dataset)
        current_time = time.time()
        print(f'Val loss: {val_loss:.4f}, Acc: {val_acc:.4f} Training time: {(current_time-start_time)//60:.0f} minutes')
        running_val_acc.append(val_acc)
        #print('running_val_acc: ', running_val_acc)
        model.train()
        if val_acc > 0.98:
            print('Performance condition satisfied, Stopping..')
            save_file = './models/best_classifier.pt'
            torch.save(model.state_dict(), save_file)
            return
        
        plot_curve(running_train_loss, running_val_loss, param='loss')
        plot_curve(running_train_acc, running_val_acc, param='acc')
        
    print("Training Complete...")
    save_file = './models/classifier_02.pt'
    torch.save(model.state_dict(), save_file)

