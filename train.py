import os
import argparse
import torch
import torchvision
from torchsummary import summary
from classes import train, show_preds, predict, create_dataset, train_test_split
from os import makedirs


def main(action, image_dir, epoch, batchsize, device, trained_model):

    class_names = ['normal', 'viral', 'covid']
    
    # Split the Image Dataset into 
    train_test_split('Covid_Database/', image_dir, 0.2)

    # If training is carried out
    if action == 'train':
        # path where we want to save the trained model
        makedirs(trained_model, exist_ok=True)
        # Creating Customn Dataset
        train_dataset, val_dataset = create_dataset(image_dir, True)
        # Prepare Dataloader
        dl_train = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
        dl_val = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=True)
        # Creating the Model
        model = torchvision.models.resnet18(pretrained=True)
        # Adjust classification layer to problem at hand with 3 output classes
        model.fc = torch.nn.Linear(in_features=512, out_features=3)
        model = model.to(device)
        summary(model, (3, 224, 224))
        # Define loss function and Optimizer. Select all model parameters to be fined-tuned
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

        print('Num of Training batches', len(dl_train))
        print('Num of Validation batches', len(dl_val))

        show_preds(model, dl_val, class_names, device) 
        # Evaluate accuracy of non-trained model
        print('calculating model accuracy before training (should be 1/3 for 3 classes)....')
        precision,recall,f1,accuracy,_ = predict(model, dl_val, class_names, loss_fn, device)

        print(f'Precision of un-trained model: {precision:.4f}')
        print(f'Recall of un-trained model: {recall:.4f}')
        print(f'F1 Score of un-trained model: {f1:.4f}')
        print(f'Accuracy of un-trained model: {accuracy:.4f}')
        # Start Training
        train(epoch, dl_train, dl_val, model, optimizer, loss_fn, val_dataset, train_dataset, trained_model, class_names, device)
    
    # If prediction/testing is carried out
    elif action == 'predict':
        # Creating Customn Dataset
        test_dataset = create_dataset(image_dir, False)
        # Prepare Dataloader
        dl_test = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)
        # load the saved model
        model = torchvision.models.resnet18(pretrained=True)
        # Adjust classification layer to problem at hand with 3 output classes
        model.fc = torch.nn.Linear(in_features=512, out_features=3)
        model = model.to(device)
        if os.path.exists(trained_model):
            if device=="cuda":
                model.load_state_dict(torch.load(trained_model))
            else:
                model.load_state_dict(torch.load(trained_model, map_location=torch.device('cpu')))
            print('Num of Test batches', len(dl_test))
            # Prediction on Test data
            precision,recall,f1,accuracy,_= predict(model, dl_test, class_names, device)
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'F1 Score: {f1:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
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
    parser.add_argument('--image_dir', type=str, default='', help='directory where training and validation images are located')
    parser.add_argument('--epoch', type=int, help='total training epochs')
    parser.add_argument('--batch_size', type=int, help='batch size') 
    parser.add_argument('--model_path', type=str, help='trained model')
    args = parser.parse_args()
        
    if args.epoch and args.batch_size != None:
        main(args.run, args.image_dir, args.epoch, args.batch_size, device, args.model_path)
    else:
        main(args.run, args.image_dir, None, args.batch_size, device, args.model_path)
