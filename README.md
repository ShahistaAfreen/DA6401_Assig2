# DA6401-Assignment 2 
# Part A
# Question 01: CNN on iNaturalist Dataset


 Dataset Setup
- Mounted from Google Drive.
- Extracted from `nature_12K.zip` to '/content/drive/MyDrive/DL_A2_Dataset'
- Splitted the 10% training data to validation


 Model: `CustomNeuralNetwork` <br>
A CNN with 5 conv-activation-maxpool blocks followed by:
- 1 dense layer with `n` neurons
- 1 output layer with `10` neurons (for 10 classes)

 Configurable Parameters
- `m` : number of filters per conv layer  
- `k` : kernel size  
- `n` : neurons in the dense layer  
- Activation function 
- Input image size (default: `224x224x3`)  


Computation Metrics
- `compute_parameters(m, k, n)`: total trainable parameters  
- `compute_computations(m, k)`: total convolution operations
# Question 02: Hyperparameter Tuning with W&B Sweep on iNaturalist 



 Dataset Setup
- Mounted from Google Drive: '/content/drive/MyDrive/DL_A2_Dataset'
- **Randomly sampled 100 images per class** for faster training and experimentation
- Validation split: 10% from training data, **class-balanced**
- **Test set was not used** for hyperparameter tuning



 Model: `NeuralVisionNetwork` <br>
Configurable CNN with
- 5 convolutional blocks
- Flexible filter layouts (same/double/half)
- Built-in augmentation toggle
- BatchNorm, Dropout
- Dense layer with ReLU before final output

Training Setup
- Optimizer: Adam
- Max Epochs: 10
- Early stopping on val_acc (patience = 3)
- Image Size: 224x224
- Batch Size: 64

 Config Parameters
- `base_filter`: base filters (64/128)
- `kernel_size`: 3 or 5
- `activation`: ReLU, GELU, SiLU, Mish
- `filter_type`: same/double/half
- `batch_norm`: True/False
- `augmentation`: True/False
- `dropout`: 0 / 0.1 / 0.2
- `dense_neurons`: 128 / 256 / 512



 W&B Sweep Settings
- **Sweep Method**: Bayesian optimization
- **Goal**: Maximize `val_acc`
- **Count**: 20 trials
- **Project**: 'vision-model-sweep'

```python
sweep_id = wandb.sweep(sweep_settings, project="vision-model-sweep")
wandb.agent(sweep_id, function=launch_training, count=20)
```
# Question 03 & 04 : Wandb report, Final Test Evaluation using Best Model




Test Data Usage
- Dataset path: '/content/drive/MyDrive/DL_A2_Dataset/val'
- Test data was **never used** during model selection or hyperparameter tuning
- All model selection was strictly based on train-validation performance


# Part B
# Question 01 & 02: Answer in markdown cell (PartB.ipynb)
# Question 03 : Fine-Tuning ResNet50 on iNaturalist 12K 

This project fine-tunes a **ResNet50** model using **PyTorch Lightning** on the 10-class variant of the **iNaturalist 12K** dataset. The goal is to explore different transfer learning strategies and identify the most effective approach for fine-tuning. <br>


 Dataset

- **Source**: `nature_12K.zip` (mounted from Google Drive in Colab : '/content/drive/MyDrive/DL_A2_Dataset')
- **Preprocessing**:
- Resize to `224x224` (ImageNet standard)



Model Architecture

- Base model: **ResNet50** with pretrained ImageNet weights
- Modified FC layer:
```python
net.fc = nn.Sequential(
    original_fc,         # 1000-class original output
    nn.ReLU(),
    nn.Linear(1000, 10)  # Adapted for 10-class iNaturalist
)
```
 Training Configuration
- Framework: PyTorch Lightning

- Optimizer: Adam (lr=1e-4)

- Loss: CrossEntropyLoss

- Batch size: 64

- Epochs: 5 <br>

