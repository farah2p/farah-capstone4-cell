# Cell Nuclei Semantic Segmentation
This project aims to develop an algorithm for semantic segmentation that can automatically detect nuclei in images. The algorithm has the potential to expedite research in various diseases, including cancer, heart disease, and rare disorders, by enabling researchers to analyze the DNA contained within cell nuclei. This genetic information plays a crucial role in determining the function of each cell and understanding the underlying biological processes.

## Dataset
The dataset used for this project is sourced from the Data Science Bowl 2018 competition on Kaggle. You can find the dataset here https://www.kaggle.com/competitions/data-science-bowl-2018/overview. Please download the file and extract it to the appropriate directory.

## File Structure
### farahAI03-capstone4-cell.ipynb: The main Python script containing the code for the project.
### saved_models: A directory to store the trained model files:
- model.h5: Complete trained model.
- model_architecture.pkl: Serialized model architecture in a binary file, saved using pickle.
- model_weights.pkl: Trained weights.
This structure ensures ease of access, reuse, and deployment.

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

The segmentation model uses the U-Net architecture with:
- Pretrained MobileNetV2 as the encoder.
- Skip connections to preserve spatial information.
- Decoder with upsampling layers for resolution enhancement.

## Installation & Setup
1) Clone the repository:
```shell
git clone https://github.com/farah2p/farah-capstone4-cell.git
```
2) Install the dependencies
```shell
pip install tensorflow==2.12.0 numpy==1.24.2 matplotlib==3.7.1 pandas==1.5.3 tensorboard==2.12.3
```
3) Download and place the dataset in the project directory.
4) Start TensorBoard for visualization:
```shell
tensorboard --logdir tensorboard-log/capstone-4
```

## Training
1) Data Prepearation:
- Load and preprocess input and mask images.
- Apply data augmentation to improve generalization.
2) Model Training:
- Compile the U-Net model with a suitable loss function and optimizer.
- Train using the training set for 10 epochs.
3) Evaluation:
- Evaluate performance on the testing dataset.
- Monitor metrics such as loss and accuracy.
Training Metrics (10 epochs):
- Epoch 1/10: Loss: 0.0982, Accuracy: 0.9597
- Epoch 5/10: Loss: 0.0828, Accuracy: 0.9661
- Epoch 10/10: Loss: 0.0748, Accuracy: 0.9694  

## Evaluation
- Test Loss: 0.0899
- Test Accuracy: 96.32%

## Results
The trained U-Net model achieves high accuracy (96.32%) with low loss on the test dataset. These results demonstrate its potential to automate the detection of cell nuclei, expediting disease research and treatment development.



The computed metrics provide a quantitative measure of the model's performance. The lower the loss value, the better the model's predictions align with the ground truth annotations. Additionally, the accuracy metric indicates the percentage of correctly classified pixels in the test dataset.

These results demonstrate that the developed semantic segmentation model achieves a high level of accuracy, accurately identifying and segmenting cell nuclei in the test images.

### Conclusion
The developed semantic segmentation model shows promising results in detecting and segmenting cell nuclei in images. With its high accuracy and progressively decreasing loss values over the epochs, the model has the potential to streamline research processes in various diseases. By automating the identification of cell nuclei, researchers can expedite their analysis of cellular responses to different treatments, leading to faster insights into the underlying biological processes and ultimately expediting the development of cures and new drugs for the benefit of patients worldwide.

The robust performance of the model on the test dataset reinforces its potential as a valuable tool in expediting research on various diseases, including cancer, heart disease, and rare disorders. By automating the detection of cell nuclei, researchers can analyze the DNA contained within these nuclei, gaining valuable insights into cellular responses to different treatments and facilitating the development of new drugs and cures.

Overall, the evaluation results suggest that the developed model holds promise for advancing research in the biomedical field and has the potential to significantly impact the speed at which cures are developed, benefitting individuals suffering from a wide range of health conditions.

### Below are some sample visualizations generated by the project:

- Tensorboard Epoch Accuracy:

![Tensorboard Accuracy](farah-tensorboard-epoch-accuracy.png)

- Tensorboard Epoch Loss:

![Tensorboard Accuracy](farah-tensorboard-epoch-loss.png)

- Model Architecture:

![Model Summary](farah-model-summary.png)

- Model Evaluation:

![Model Evaluation](farah-model-evaluation.png)

- To showcase the model's performance on the test dataset, the `show_predictions` function was used to visualize the segmentation results of three sample images. And here is the output:

![Show Predictions](farah-show-predictions.png)

The output displays the original images alongside their corresponding segmented cell nuclei. This demonstration provides a visual representation of the model's accuracy and its ability to effectively detect and segment cell nuclei in real-world images.

## Credits
The dataset used in this project is sourced from Kaggle:
https://www.kaggle.com/competitions/data-science-bowl-2018/overview

## Contributing
Contributions to this project are welcome. If you find any issues or have suggestions for improvement, please open an issue or submit a pull request on the GitHub repository.
