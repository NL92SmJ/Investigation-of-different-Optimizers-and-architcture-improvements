# Investigation-of-Optimizer, Regularization & Architecture Comparison
This case study analyzing the impact of different optimizers and certain improvements on Fashion-MNIST-dataset

# Motivation & Expactation
## Experiment 1 - Optimizer Benchmark
While convolutional neural networks are often introduced together with the Adam optimizer by default, the practical impact of alternative optimization strategies is rarely investigated under controlled conditions.

Most tutorials focus on achieving a good result as quickly as possible, while the underlying trade-off between convergence speed, stability, computational efficiency and final model quality is often overlooked.

The objective of this experiment was therefore to isolate the optimizer as the only varying component and evaluate its impact under identical conditions.

My expectations were:

**1.) Adaptive optimizers should converge faster**

Algorithms such as Adam, AdamW and RMSprop adapt learning rates on a per-parameter basis and should therefore reach high accuracy significantly faster than SGD with Momentum.

**2.) AdamW should provide better generalization**

Due to its decoupled weight decay implementation, AdamW is frequently reported to generalize better than standard Adam in image classification tasks.

**3.) SGD should optimize more slowly but remain highly stable**

Although SGD with Momentum is expected to require more epochs, literature frequently reports smoother optimization trajectories and strong late-stage convergence.

This benchmark investigates whether these theoretical expectations are observable on Fashion-MNIST using an identical CNN architecture.

## Experiment 2 - Architecture & Regularization Benchmark

After identifying high-performing optimizers, the next step was to investigate whether architectural and training improvements could produce larger gains than optimizer selection alone.

Modern CNN implementations commonly employ additional techniques such as Batch Normalization, Data Augmentation and Learning Rate Scheduling. These methods are often presented as universally beneficial, yet their individual contribution is rarely quantified in isolation.

The goal of this experiment was therefore to measure the practical impact of each technique under controlled conditions.

My expectations were:

**1.) Batch Normalization should accelerate convergence**

By normalizing intermediate feature distributions, BatchNorm should improve gradient flow, stabilize learning and increase validation accuracy.

**2.) Data Augmentation should improve robustness**

Artificially increasing input diversity was expected to reduce overfitting and improve generalization on unseen test samples.

**3.) Learning Rate Scheduling should improve late-stage optimization**

Adaptive reductions of the learning rate should allow coarse optimization during early epochs and fine tuning during later stages.

**4.) The combination of all techniques should achieve the best overall result**

Since each technique targets a different aspect of model optimization, the fully optimized architecture was expected to outperform all individual configurations.

# Test environment
## Hardware
  * CPU: Intel Core i3-12100F (4 cores / 8 threads)
  * GPU: Nvidia RTX 3050 - 6GB VRAM
  * RAM: 16 GB (dual-channel)
## Software
  * OS: Ubuntu 22.04.05 LTS (64bit)
  * Visual Studio Code
  * Python 3.11.15
  * TensorFlow (CUDA-enabled)
  * NumPy
  * Matplotlib
  * Scikit-Learn
  * Pandas
  * Seaborn

# Dataset
Fashion-MNIST
## Features:
  * 60,000 training images
  * 10,000 test images
  * 10  clothing categories
  * grayscale images
  * Resolution: 28x28 pixels
## Classes:
  * T-Shirt/Top
  * Trouser
  * Pullover
  * Dress
  * Coat
  * Sandal
  * Shirt
  * Sneaker
  * Bag
  * Ankle Boot

# Experimental Design
The same CNN architecture was trained using four different optimization algorithms:

  * Adam
  * AdamW
  * SGD with Momentum
  * RMSprop

# Training configuration
  * Epochs
  * AdamW
  * SGD with Momentum
  * RMSprop

  * 
1. Due to low correlation: removing feature sepal.width
2. One-Hot-Encoding (OHE) of the species column
3. Standard scaling of:
   * petal.length
   * petal.width
4. Train-test split:
   * 80% training
   * 20% testing
  
# Architectures
Following section describes the used architectures of this experiment.
## Modelarchitecture
The model architecture stays fixed for all upcoming activation-function tests:
```text
  [ Input ] ──> ( 4 Features )
       │
       ▼
  [ Layer ] ──> Dense (32, Activation)
       │
       ▼
  [ Layer ] ──> Dense (64, Activation)
       │
       ▼
  [ Output ] ──> Dense (1)
```
## Networkarchitecture
The network architecture stays fixed for all upcoming activation-function tests.
  * Dense: 32 - 64 - 1
  * Optimizer: Adam
  * Loss: Mean Absolute Error (MAE)
  * Epochs: 70
  * Batch size: 32

# Experimental Design
Only the activation function was changed - all other parameters remained identical.
Tested activation functions:
  * ReLU
  * Leaky ReLU
  * ELU
  * SeLU
  * GELU
  * Swish
  * Tanh
  * Sigmoid
    
To understand and follow the thesis of the textbooks, I will first compare MAE and R² for each activation function. In the second step, I will compare: error distribution, loss progression, prediction vs reality, and residual plots.

# Results
After running my Python-script, I've got the following results. They are ordered by ascending R²-values.

| Activiation-function | MAE   | R²-Score |
|----------------------|-------|----------|
| ELU                  | 0,259 | 0,849    |
| SeLU                 | 0,270 | 0,837    |
| Swish                | 0,284 | 0,815    |
| Tanh                 | 0,262 | 0,815    |
| ReLU                 | 0,304 | 0,802    |
| Leaky_ReLU           | 0,320 | 0,793    |
| GELU                 | 0,332 | 0,767    |
| Sigmoid              | 0,482 | 0,419    |

Theory says: MAE is more robust to outliers and gives a directly interpretable average deviation. R² is useful for communicating explained variance, but it can be misleading if looked at on its own: a high R² doesn’t automatically mean good predictions on new data, and R² can even be negative if a model performs worse than just predicting the mean.
Just by viewing MAE and R²-score for each activation-function, it can be postulated: for the chosen architecture and training configuration, ELU achieved the highest R² score. Sigmoid performed significantly worse than all other activation funtions. Interstingly, GELU did not outperform ReLU in this experiment, despite its popularity in modern deep learning architectures.
The results suggest that activation-function performance is highly dependent on the dataset and network architecture.

The final evaluation is not based solely on MAE and R².
To better understand model behaviour and generalization, the following diagnostic plots were analyzed:
 * Loss progression
 * Prediction vs. Reality
 * Error distribution
 * Residual plots
These visualizations help identify overfitting, underfitting, systematic prediction bias and overall prediction quality.

## loss-progression
For easy classification:
 * ideal case: Both curves go down and stay close to each other --> model learns and generalizes well to unseen data.
 * overfitting: training goes down, validation goes up: training keeps getting better; Validation is getting worse --> model is momorizing the training data
 * underfitting: both curves go down, but on high level --> model is to simple or short trained

<img width="800" height="500" alt="loss_tanh" src="https://github.com/user-attachments/assets/78f74647-f035-401d-a924-33cd0e823084" />
<img width="800" height="500" alt="loss_swish" src="https://github.com/user-attachments/assets/41cd16cd-5e5e-4b6c-997a-f5ba973edebe" />
<img width="800" height="500" alt="loss_sigmoid" src="https://github.com/user-attachments/assets/25b1b39f-c5d8-43ee-9c4a-271ca2929050" />
<img width="800" height="500" alt="loss_selu" src="https://github.com/user-attachments/assets/95d58f69-e1ed-45cd-bb6d-ba7fba4333b7" />
<img width="800" height="500" alt="loss_relu" src="https://github.com/user-attachments/assets/615555c8-1b3f-42fa-ac7f-ae2eb2d9ad43" />
<img width="800" height="500" alt="loss_leaky_relu" src="https://github.com/user-attachments/assets/3533bc3e-4e1d-445e-bdec-904ce97243e9" />
<img width="800" height="500" alt="loss_gelu" src="https://github.com/user-attachments/assets/33c2db54-1376-4e45-9bef-31d8ae101443" />
<img width="800" height="500" alt="loss_elu" src="https://github.com/user-attachments/assets/e5d02bbb-aa84-46e8-bf37-85106da57988" />

A good loss progression is characterized by decreasing training and validation loss curves that remain close to each other.
A widening gap between both curves indicates overfitting, whereas consistently high losses suggest underfitting.

## prediction vs. reality
For easy classification:
 * the red line symbolizes the ideal line
 * perfect model: all data-points on the diagonal --> "prediction equals reality."
 * good model: all data-points lie close around the diagonal --> "prediction follow the real values with small deviations."
 * bad model: large spread --> "predictions are inaccurate and deviate significantly from the real targets."

<img width="600" height="600" alt="pred_tanh" src="https://github.com/user-attachments/assets/f1795b7a-f88f-41a3-9011-f8b4ee2d65a5" />
<img width="600" height="600" alt="pred_swish" src="https://github.com/user-attachments/assets/49236655-710c-444b-9a8d-fe9a6f89989a" />
<img width="600" height="600" alt="pred_sigmoid" src="https://github.com/user-attachments/assets/2f1b9429-fca3-4c31-8d91-b45fb464bed9" />
<img width="600" height="600" alt="pred_selu" src="https://github.com/user-attachments/assets/6840749c-b60a-4bf3-9a94-75dac6dec8c2" />
<img width="600" height="600" alt="pred_relu" src="https://github.com/user-attachments/assets/c4249777-a321-4287-902d-1eb5c900fc73" />
<img width="600" height="600" alt="pred_leaky_relu" src="https://github.com/user-attachments/assets/d9b022c1-9a18-41a1-bd56-85bfa903e519" />
<img width="600" height="600" alt="pred_gelu" src="https://github.com/user-attachments/assets/b2893f07-4e3e-461f-9674-3a9655450180" />
<img width="600" height="600" alt="pred_elu" src="https://github.com/user-attachments/assets/ce757244-e052-4533-a923-dca6d67d7afd" />

## error history
For easy classification:
 * ideal case: maximum at 0 --> The model overestimates and underestimates about equally often.
 * shift case: maximum not at 0 --> The model has a systematic error (bias).
 * very wide distribution case: big prediction errors.
   
<img width="800" height="400" alt="error_hist_tanh" src="https://github.com/user-attachments/assets/96e64fbb-4726-45ad-8abb-965dad9c7e86" />
<img width="800" height="400" alt="error_hist_swish" src="https://github.com/user-attachments/assets/884f35df-d169-46b6-92e8-5056112aa127" />
<img width="800" height="400" alt="error_hist_sigmoid" src="https://github.com/user-attachments/assets/478977fc-17c9-4ceb-bac8-d3af05125b2e" />
<img width="800" height="400" alt="error_hist_selu" src="https://github.com/user-attachments/assets/304d288e-a72f-4ce1-ab8a-90db661c9a1f" />
<img width="800" height="400" alt="error_hist_relu" src="https://github.com/user-attachments/assets/9ca0c224-f15c-47f0-9360-d41b2de51aa8" />
<img width="800" height="400" alt="error_hist_leaky_relu" src="https://github.com/user-attachments/assets/c2d9c67c-8a39-42ea-927e-9aa54ca84b8f" />
<img width="800" height="400" alt="error_hist_gelu" src="https://github.com/user-attachments/assets/d4bdcee7-be91-4c80-9f9a-edd264498e00" />
<img width="800" height="400" alt="error_hist_elu" src="https://github.com/user-attachments/assets/5111f210-20c6-40fd-a84f-5ba12c7545cb" />

A well-performing model should produce a centered error distribution around zero.
Wide distributions indicate larger prediction errors, while shifted distributions may reveal systematic prediction bias.

## residual-plot
For easy classification:
 * good case: random scatter around 0 / no patterns. --> interpretation: no systematic error pattern.
 * bad case 1 "curves" --> Model didn't understand the context.
 * bad case 2 "funnel shapes" -->  Errors get bigger as the values increase.
 * bad case 3 "cluster" --> There are groups in the dataset that are treated differently.

<img width="800" height="400" alt="residual_tanh" src="https://github.com/user-attachments/assets/2a4d5f2c-e03a-4f0a-b32f-a13a9e350a4a" />
<img width="800" height="400" alt="residual_swish" src="https://github.com/user-attachments/assets/1760e943-a352-4c3e-8f2e-33151ba4559e" />
<img width="800" height="400" alt="residual_sigmoid" src="https://github.com/user-attachments/assets/12de109b-efa2-47af-b2d0-befa5003ca16" />
<img width="800" height="400" alt="residual_selu" src="https://github.com/user-attachments/assets/cc230429-ed63-41c5-b296-2ad0b4d20964" />
<img width="800" height="400" alt="residual_relu" src="https://github.com/user-attachments/assets/fb92db62-2621-418f-8b48-363832233bb9" />
<img width="800" height="400" alt="residual_leaky_relu" src="https://github.com/user-attachments/assets/000fda07-fe01-4543-a51c-002158bacf48" />
<img width="800" height="400" alt="residual_gelu" src="https://github.com/user-attachments/assets/418eb1c7-7635-4fd6-aacf-4cdb652b11b9" />
<img width="800" height="400" alt="residual_elu" src="https://github.com/user-attachments/assets/b1ef1a58-9f8f-4363-9fde-4359429febd2" />

# Interpretation
The conducted experiments show that the choice of activation function can significantly influence the performance of a neural network, even when all other training parameters remain unchanged.
ELU achieved the highest R² score (0.849) and the lowest MAE (0.259), indicating the best overall prediction quality among the tested activation functions.
The prediction-versus-reality plots reveal that ELU follows the ideal prediction line considerably better than Sigmoid. While ELU is able to reproduce the overall trend of the target values, the Sigmoid-based model compresses predictions into a narrow value range and fails to represent larger target values accurately.
This behavior is also visible in the residual plots. The residuals of the ELU model are distributed relatively randomly around zero, suggesting that no strong systematic prediction errors are present. In contrast, the Sigmoid model exhibits a clear pattern: prediction errors become increasingly negative for larger predictions, indicating a systematic underestimation of larger target values.
The error histograms further support these observations. ELU produces a narrower error distribution centered around zero, whereas Sigmoid shows a broader distribution with larger negative prediction errors.
Overall, the results suggest that modern activation functions such as ELU and SeLU are better suited for this regression task than the classical Sigmoid function.

# Next steps / experiments

This experiment focused on the influence of activation functions while keeping all other parameters constant.
Future investigations could examine additional factors that affect neural network performance:

## Network Architecture
 * Compare shallow and deep networks
 * Compare narrow and wide network structures
 * Evaluate symmetric architectures (e.g. 32-64-32) versus asymmetric architectures (e.g. 32-64)
## Optimizers
 * Compare Adam, SGD and RMSProp
 * Analyze convergence speed and prediction quality
 * Batch Size
 * Investigate the impact of different batch sizes on training stability and model performance
## Learning Rate
 * Evaluate how learning rate tuning influences convergence and final prediction accuracy
## Regularization
 * Introduce Dropout layers to study their effect on overfitting and generalization
## Classification Task
 * Perform a similar comparison using the original Iris classification problem instead of a regression task

The long-term goal is to identify which hyperparameters have the strongest influence on model performance and whether the observed activation-function ranking remains stable under different network configurations.

