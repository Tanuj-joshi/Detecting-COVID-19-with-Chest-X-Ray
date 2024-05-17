import os
import argparse
import torch
import torchvision
from torchsummary import summary
from classes import ChestXRayDataset, train, show_preds, predict


def main(action, batchsize, epoch, device, trained_model):

    class_names = ['normal', 'viral', 'covid']
    # Split the Image Dataset into 
    #train_test_split()

    ''' 
    Image Transformation.
        Convert all the images to the size that, Resnet-18 model (pre-trained) expects.
        Normalize the input images to a specific range (mean and std), which is used 
        when this Resnet-18 model was trained on IMAGENET dataset.
    '''
    train_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                                                ])
    test_transform = torchvision.transforms.Compose([
                torchvision.transforms.Resize(size=(224, 224)),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                std=[0.229, 0.224, 0.225]) 
                                                ])
    train_dirs = {
                'normal': 'COVID-19_Radiography_Database/normal',
                'viral': 'COVID-19_Radiography_Database/viral',
                'covid': 'COVID-19_Radiography_Database/covid'
                }
    test_dirs = {
            'normal': 'COVID-19_Radiography_Database/test/normal',
            'viral': 'COVID-19_Radiography_Database/test/viral',
            'covid': 'COVID-19_Radiography_Database/test/covid'
            }
    
    # Creating Customn Dataset
    train_dataset = ChestXRayDataset(train_dirs, train_transform)
    test_dataset = ChestXRayDataset(test_dirs, test_transform)

    # Prepare Dataloader
    dl_train = torch.utils.data.DataLoader(train_dataset, batch_size = batchsize, shuffle = True)
    dl_test = torch.utils.data.DataLoader(test_dataset, batch_size = batchsize, shuffle = True)

    # If training is carried out
    if action == 'train':

        # Creating the Model
        model = torchvision.models.resnet18(pretrained = True)
        # Adjust classification layer to problem at hand with 3 output classes
        model.fc = torch.nn.Linear(in_features = 512, out_features = 3)
        model = model.to(device)
        summary(model, (3, 224, 224))
        # Define loss function and Optimizer. Select all model parameters to be fined-tuned
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        print('Num of Training batches', len(dl_train))
        print('Num of Validation batches', len(dl_test))

        show_preds(model, dl_test, class_names, device) 
        # Evaluate accuracy of non-trained model
        print('calculating model accuracy before training (should be 1/3 for 3 classes)....')
        accuracy = 0
        model.eval()      
        for val_step, (images, labels) in enumerate(dl_test):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, dim=1)
            accuracy += sum((preds == labels).cpu().numpy())          
        accuracy = accuracy / len(test_dataset)
        print(f'Accuracy of un-trained model: {accuracy:.4f}')
        # Start Training
        train(epoch, dl_train, dl_test, model, optimizer, loss_fn, test_dataset, train_dataset, device)
        
    # If prediction/testing is carried out
    elif action == 'predict':
        # load the saved model
        if os.path.exists(trained_model):
            if device=="cuda":
                model.load_state_dict(torch.load(trained_model))
            else:
                model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
            print('Num of Test batches', len(dl_test))
            # Prediction on Test data
            predict(model, dl_test, class_names, device)
        else:
            print('No trained Model is yet avaliable')



if __name__ == "__main__":

    # Check if GPU is available
    if torch.cuda.is_available():
        print('Default GPU Device: {}'.format(torch.cuda.get_device_name(0)))
    else: 
        print('No GPU available, training on CPU; consider making n_epochs very small.')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=str, default='train', help='whether to train or predict')
    parser.add_argument('--epoch', type=int, help='total training epochs')
    parser.add_argument('--batch_size', type=int, help='total training epochs') 
    parser.add_argument('--model_path', type=str, help='trained model')
    args = parser.parse_args()
        
    if args.epoch and args.batch_size != None:
        main(args.run, args.batch_size, args.epoch, device, None)
    else:
        main(args.run, args.batch_size, None, device, args.model_path)

