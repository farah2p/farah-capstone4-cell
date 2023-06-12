# Cell Nuclei Semantic Segmentation
This project aims to develop an algorithm for automatic detection and semantic segmentation of cell nuclei in images. The algorithm will assist in expediting research on various diseases, including cancer, heart disease and rare disorders, ultimately accelerating the development of cures and benefiting patients with different health conditions.

## Dataset
The dataset used for this project is sourced from the Data Science Bowl 2018 competition on Kaggle. You can find the dataset here https://www.kaggle.com/competitions/data-science-bowl-2018/overview. Please download the file and extract it to the appropriate directory.

## Usage
### 1. Navigate to the project directory:
```shell
cd your-repository
```
NOTES: make sure you replace 'your-repositary' with the correct name
### 2. Run the main script:
```shell
python farahAI03_capstone4_cell.py
```
## Data Preparation
The dataset is expected to be located in the following directory structure:

└── data-science-bowl-2018

    ├── train 
    │   ├── inputs
    │   │   ├── image1.png
    │   │   ├── image2.png
    │   │   └── ...
    │   └── masks
    │       ├── mask1.png
    │       ├── mask2.png
    │       └── ...
    └── test
        ├── inputs
        │   ├── image1.png
        │   ├── image2.png
        │   └── ...
        └── masks
            ├── mask1.png
            ├── mask2.png
            └── ...
Ensure that you have the necessary permissions to access the dataset files.

## Model Architecture
The model used in this project is a U-Net, which is a popular architecture for image segmentation tasks. It consists of an encoder (downsampling path) and a decoder (upsampling path) to capture high-level and low-level features of the input images.

The U-Net architecture has the following components:
- Pretrained model (MobileNetV2) as the feature extractor.
- Skip connections between the encoder and decoder to preserve spatial information.
- Upsampling layers to increase the resolution of the feature maps.
- The model architecture can be visualized using the provided diagram.

## Training
To train the model, follow these steps:
- Set the file paths for the train and test datasets in the code.
- Adjust the hyperparameters (e.g., batch size, learning rate) if needed.
- Run the training script.
- The training process will involve the following steps:

1. Load and preprocess the input and mask images.
2. Create data augmentation layers to improve the model's generalization.
3. Build the U-Net model using the specified architecture.
4. Compile the model with the appropriate loss function and optimizer.
5. Train the model using the training dataset.
6. Monitor the training progress using TensorBoard.
7. Evaluate the model's performance on the testing dataset.

## Evaluation
After training the model, it can be evaluated using the testing dataset. The evaluation process includes the following steps:
- Load the trained model weights.
- Preprocess the input and mask images from the testing dataset.
- Run the evaluation script.
- Compute the metrics (e.g., accuracy, loss) to assess the model's performance.

## Installation
### 1. Clone the repository to your local machine using the following command:
```shell
git clone https://github.com/farah2p/farah-capstone4-cell.git
```
### 2. Before running the code, ensure that you have the following dependencies installed:
- TensorFlow
- Pandas 1.5.3
- Matplotlib
- Tensorboard 2.12.3

Install the required dependencies by running the following command:
```shell
pip install tensorflow==2.12.0
pip install numpy==1.24.2
pip install matplotlib==3.7.1
pip install pandas==1.5.3
pip install tensorboard===2.12.3
```
### 3. Download the dataset from the provided Kaggle link and place it in the project directory.
### 4. Open the Jupyter Notebook or Python script containing the code.
### 5. Run the code cells or execute the script to perform data preprocessing, model training, and evaluation.
### 6. Use Tensorboard to visualize the model graph and training progress by running the following command in the project directory:
```shell
tensorboard --logdir tensorboard_logs/capstone4
```
Access Tensorboard in your web browser using the provided URL.
### 7. The trained model will be saved in the "saved_models" folder in .h5 format as model.h5

## Project Requirements
- Python
- TensorFlow library

## Results
After developing the U-Net-based model for Cell Nuclei Segmentation, the following results were achieved:

## Credits
The dataset used in this project is sourced from Kaggle:
https://www.kaggle.com/competitions/data-science-bowl-2018/overview

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
