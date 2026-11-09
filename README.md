# Investigation-of-Optimizer, Regularization & Architecture Comparison
This case study analyzing the impact of different optimizers and certain improvements on Fashion-MNIST-dataset

# Motivation & Expactation

Convolutional Neural Networks (CNNs) are commonly introduced together with standard training pipelines, where choices such as optimizers, Batch Normalization, learning rate scheduling or Data Augmentation are often presented as generally beneficial practices. However, systematic comparisons of these techniques under controlled conditions are less common, particularly for beginners trying to understand why certain design choices improve model behaviour.

The objective of this project was therefore not simply to achieve the highest possible Fashion-MNIST accuracy, but to empirically investigate how individual optimization and regularization techniques affect convergence speed, training stability, model generalization and final classification performance.

To answer these questions, two dedicated experiments were conducted.

## Experiment 1 - Optimizer Benchmark
The first experiment isolates the optimizer as the only changing variable.

My expectations were:

**1.) Adaptive optimizers should converge faster**

Algorithms such as Adam, AdamW and RMSprop adapt learning rates on a per-parameter basis and should therefore reach high accuracy significantly faster than SGD with Momentum.

**2.) AdamW should provide better generalization**

Due to its decoupled weight decay implementation, AdamW is frequently reported to generalize better than standard Adam in image classification tasks.

**3.) SGD should optimize more slowly but remain highly stable**

Although SGD with Momentum is expected to require more epochs, literature frequently reports smoother optimization trajectories and strong late-stage convergence.

This benchmark investigates whether these theoretical expectations are observable on Fashion-MNIST using an identical CNN architecture.

## Experiment 2 - Architecture & Regularization Benchmark

The second experiment investigates common CNN improvement techniques.

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
Fashion-MNIST is a grayscale image classification dataset designed as a more challenging replacement for the handwritten MNIST digit dataset.
## chracteristics:
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

## Preprocessing:
  * Pixel normalization from [0,255] to [0,1]
  * Additional channel dimension added for CNN compatibility
  * Validation split: 20%
  * No class balancing applied

# Base CNN Architecture
The benchmark uses a custom CNN architecture consisting of:

## Preprocessing:
  * Conv2D (64)
  * Conv2D (64)
  * MaxPooling
  * Conv2D (128)
  * Conv2D (128)
  * MaxPooling

# classification head:
  * Flatten
  * Dense (128)
  * Dropout (0.3)
  * Dense (10 Softmax)

The same architecture serves as the foundation for all optimizer and architecture benchmark experiments.

# Experiment 1: Optimizer Benchmark
## Experimental Design
The optimizer benchmark compares four optimization algorithms while keeping all other parameters fixed.
## Tested Optimizers
   * Adam
   * AdamW
   * SGD with Momentum
   * RMSprop
## Shared Hyperparameters
   * Epochs: 12
   * Batch Size: 64
   * Early Stopping (patience = 3)
   * Dropout = 0.3
   * Sparse Categorical Crossentropy
   * Identical CNN Architecture

# Results
## Test Accurancy Overview

| Optimizer         | Test Accuracy |
|-------------------|---------------|
| AdamW             | 92,50%        |
| Adam              | 92,42         |
| RMSProp           | 92,10         |
| SGD with Momentum | 91,98%        |

The overall accuracy differences are relatively small, indicating that optimizer choice has a stronger influence on training dynamics than on absolute performance.

## Convergance Analysis
**Adam**

Adam demonstrated rapid convergence and reached competitive accuracy within only a few epochs.
The Training Loss continuously decreased while Validation Accuracy stabilized around 93%.
This behaviour confirms why Adam is often considered the default optimizer for CNN-based classification problems.

**Strengths**
  * Fast convergence
  * Stable training
  * Strong accuracy
    
**Weaknesses**
  * Slight increase in Validation Loss after reaching its minimum

This suggests diminishing returns during later optimization stages.


**AdamW**

AdamW achieved the highest overall test accuracy while maintaining highly stable learning curves.
Compared to standard Adam, Validation Loss remained slightly more controlled, indicating improved generalization.
The decoupled weight decay mechanism appears beneficial even on a relatively small dataset such as Fashion-MNIST.

**Strengths**
  * Highest accuracy
  * Stable convergence
  * Strong generalization

AdamW delivered the most balanced performance across all measured criteria.

**SGD with Momentum**

SGD exhibited the slowest convergence of all tested optimizers.
The model required considerably more epochs before reaching competitive Validation Accuracy levels.
However, the optimization trajectory remained remarkably smooth throughout training.

**Strengths**
  * Stable optimization
  * Predictable convergence
  * Highest throughput

**Weaknesses**
  * Slow learning
  * Lowest final accuracy

The results suggest that SGD would likely benefit from substantially longer training schedules.

**RMSprop**

RMSprop achieved performance comparable to Adam and AdamW but showed noticeably stronger fluctuations in Validation Loss and Validation Accuracy.
The optimizer consistently found performant solutions but demonstrated less stability near convergence.

**Strengths**
  * Competitive accuracy
  * Rapid initial learning

**Weaknesses**
  * Increased metric oscillations
  * Less stable convergence

# Loss Curve Interpretation

An important observation is the slight increase in Validation Loss after several epochs for the adaptive optimizers.
At first glance, this behaviour may appear to indicate overfitting. However, Validation Accuracy remains largely unchanged during this phase. This suggests that the models continue making approximately the same number of correct predictions while becoming increasingly confident in individual decisions. Since Cross-Entropy Loss strongly penalizes highly confident mistakes, Validation Loss can increase even when classification accuracy remains stable.

Consequently, the observed behaviour should be interpreted as normal optimization dynamics rather than severe overfitting.

# Confusion Matrix Analysis
The confusion matrix reveals a highly structured error pattern.

**Best Performing Classes**

Several classes achieved exceptionally high classification rates:

| Class      | Approximate Accuracy |
|------------|----------------------|
| Trouser    | ~99%                 |
| Bag        | ~99%                 |
| Sandal     | ~99%                 |
| Sneaker    | ~98%                 |
| Ankle Boot | ~96-97%              |

These categories possess highly distinctive geometric characteristics, making separation relatively straightforward.

**Most challenging Classes**
The majority of classification errors occurred among:

  * Shirt
  * T-Shirt/Top
  * Pullover
  * Coat

The Shirt category consistently produced the highest error rates.
Most errors occurred because Shirts were predicted as:

  * T-Shirt/Top
  * Coat
  * Pullover

Given the limited 28×28 image resolution and grayscale format, these categories share substantial visual overlap.
The resulting confusion appears to be primarily dataset-driven rather than architecture-driven.

## **Key Findings**

**Confirmed Expectations**

  * Adaptive optimizers converged significantly faster than SGD.
  * AdamW achieved the strongest overall performance.
  * SGD displayed highly stable optimization behaviour.
  * Final accuracy differences remained relatively small.

**Unexpected Findings**

  * Confusion primarily originated from visually similar garment categories rather than optimization weaknesses.
  * Remaining model limitations were largely dataset-related.


# Experiment 2: Architecture & Regularization Benchmark

## Experimental Design
Five architectural configurations were compared using the Adam optimizer.

## Tested Configurations
   * Baseline_Adam
   * BatchNorm
   * LearningRate_Scheduler
   * Full_Optimzed
     
## Full Optimized Includes
   * Batch Normalization
   * Data Augmentation
   * Learning Rate Scheduler
   * Early Stopping
   * Dropout

# Results
## Validation Accuracy Ranking

The strongest overall results were achieved by:

  1.) BatchNorm
  
  2.) LR Scheduler
  
  3.) Baseline
  
  4.) Data Augmentation
  
  5.) Full Optimized
  
## Individual Analysis
**Baseline Model**

The baseline architecture already achieved approximately 93% Validation Accuracy.
This demonstrates that the underlying CNN design is inherently well suited for Fashion-MNIST.
The baseline therefore represents a strong starting point rather than a weak reference model.

**Batch Normalization**

Batch Normalization proved to be the most effective individual improvement.
Observed benefits:
  * Faster convergence
  * Lower Validation Loss
  * Improved stability
  * Higher accuracy

The learning curves remained smooth throughout training and consistently reached the highest performance levels.

**Interpretation**

Batch Normalization improves gradient flow and reduces internal covariate shift, allowing the optimizer to explore parameter space more efficiently.

**Learning Rate Scheduler**

Learning Rate Scheduling produced results nearly identical to BatchNorm.
The scheduler initially allowed larger optimization steps and later reduced the learning rate to perform fine-tuning near convergence.

**Benefits Observed**

   * Smooth Validation Loss
   * Stable convergence
   * Strong late-stage optimization

Its performance confirms the importance of dynamic learning rate adaptation.

**Data Augmentation**

Data Augmentation improved robustness, but the performance gain remained smaller than expected.
Fashion-MNIST contains highly standardized and centered images.

As a consequence, transformations such as:

   * Rotation
   * Zoom
   * Translation

introduce limited new information compared to more diverse natural image datasets.

**Interpretation**

Data Augmentation provides value, but its impact is constrained by the dataset itself.

**Full Optimized**

The most surprising result emerged from the Full Optimized configuration.
Contrary to expectations, the combined model failed to outperform BatchNorm or LR Scheduler.
Validation Accuracy remained noticeably lower, while Validation Loss stayed elevated throughout much of training.

**Interpretation**

The evidence suggests mild underfitting caused by excessive regularization.
Several regularization techniques were simultaneously active:

   * BatchNorm
   * Data Augmentation
   * ReduceLROnPlateau
   * Dropout
   * Early Stopping

While each mechanism can improve generalization individually, their combined effect appears to have limited the effective learning capacity of the network.

This finding highlights an important practical lesson:
More optimization techniques do not automatically result in a better model.


## Confusion Matrix Analysis

The confusion matrix of the best architecture produced similar observations to Experiment 1.

**Strong Classes**
   * Trouser
   * Bag
   * Sneaker
   * Sandal

These classes typically exceeded 98% classification accuracy.

**Weak Classes**

The most difficult categories remained:

   * Shirt
   * Coat
   * Pullover

Notably, T-Shirt/Top classification improved significantly compared to earlier runs.

However, part of this improvement appears to come at the expense of Shirt classification performance, indicating a shift of decision boundaries between these visually similar classes.

## Key Findings

**Confirmed Expectations**

   * Batch Normalization improved convergence and stability.
   * Learning Rate Scheduling improved optimization quality.
   * The baseline architecture already performed surprisingly well.

**Rejected Expectations**

   * Combining all techniques did not create the strongest model.
   * Data Augmentation produced smaller gains than expected.

**Unexpeted Findings**

   * The Full Optimized configuration showed signs of over-regularization and mild underfitting.
   * Individual improvements proved more effective than maximum optimization complexity.

# Overall Conclusion

Across both experiments, the CNN consistently achieved classification accuracies around 92–93%, placing the model within the expected performance range for a relatively lightweight Fashion-MNIST architecture.
The optimizer benchmark demonstrated that AdamW provides the best balance between convergence speed, stability and generalization, while SGD remains a viable alternative for longer training schedules.
The architecture benchmark revealed that Batch Normalization and Learning Rate Scheduling contribute more meaningfully to performance than Data Augmentation on this dataset.
Perhaps the most valuable insight is that model improvement is not a linear process.
The strongest model was not the most complex configuration. Instead, carefully selected individual improvements outperformed the fully optimized setup.

This study therefore reinforces a key machine learning principle:
Effective model design is driven by empirical validation, not by the sheer number of applied optimization techniques.

# Next Steps

This benchmark opens several interesting directions for future experiments:

   * Residual CNN architectures (ResNet-style blocks)
   * Deeper convolutional networks
   * Transfer learning with pretrained feature extractors
   * Hyperparameter tuning of learning rates and weight decay
   * Alternative regularization methods (L2, Label Smoothing)
   * CIFAR-10 comparison study
   * Fashion-MNIST ensemble models

The complete source code, benchmark results, CSV exports, training curves and confusion matrices are available in this repository.
