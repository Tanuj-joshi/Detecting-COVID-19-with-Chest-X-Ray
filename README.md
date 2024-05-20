# Detecting-COVID-19-with-Chest-X-Ray using PyTorch

In this Project, we created an Image Classifier using a Resnet-18 model and trained it on a Covid-19 Radiography Dataset which is available on Kaggle 

This dataset has images of chest X-Ray scan which are categorized in 3 classes: 
1) Normal
2) Viral Pneumonia
3) Covid-19

Goal is to predict chest X-ray scans that belonged to one of these three classes with a reasonably high accuracy.

## Table of Contents

- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Dataset

The dataset used for this project is the [Kaggle Chest-X-Ray-Radiography](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database). It consists of 3000 labeled images of lungs scan with 1)Normal, 2)Viral Pneumonia and 3)Covid-19. The dataset is divided into training and validation sets to evaluate the performance of the model.

## Requirements

To run this project, you will need the following dependencies:

- Python 3.7 or higher
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- scikit-learn
- Jupyter Notebook (optional, for interactive experimentation)

## Installation

1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/Tanuj-joshi/Detecting-COVID-19-with-Chest-X-Ray.git 
   ```

2. Create a virtual environment and activate it::

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Download the dataset from Kaggle and place it in the Covid_Database/ directory.

## Usage

### Training the Model

1. Ensure that the dataset is properly placed in the Covid_Database/ directory.

2. Run the training script:

   ```bash
   python train.py --run train --image_dir Dataset/ --batch_size 32 --epoch 100 --model_path model/
   ```
   This script will first split the data into Train-Validation set and save them into a new directory 'Dataset'. Then it trains the Resnet-18 (pretrained) model on the training dataset and save the trained model to the models/ directory.

### Evaluation

 To evaluate the performance of the trained model on the test set, run:

   ```bash
   python train.py --run predict --image_dir Dataset/val/ --batch_size 4 --model_path models/epoch2_best_classifier.pt
   ```
This script will load the trained model and output the Accuracy, Loss, Precision, recall and F1 score on the test set.

## Contributing

Contributions are welcome! If you have any ideas for improvements or new features, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- The dataset used in this project is from Kaggle.
- Special thanks to the PyTorch teams for their excellent deep learning frameworks.




