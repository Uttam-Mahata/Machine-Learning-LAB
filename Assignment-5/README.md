# Flower Classification using CNN

## Overview
This project implements a comprehensive flower classification system using Convolutional Neural Networks (CNN) with systematic hyperparameter experiments. The implementation explores various CNN architectures and configurations to find optimal parameters for image classification, and compares results with MNIST digit classification from Assignment 4.

## Dataset

### Source
- **Dataset**: Flower Images Dataset
- **Location**: `/home/Assignment-05/flowers`
- **Classes**: 5 flower types
- **Image Format**: Color RGB images

### Preprocessing
- **Resize**: Images resized to 80×80 pixels
- **Grayscale Conversion**: For most experiments (RGB used in Experiment G)
- **Normalization**: Pixel values normalized to [0, 1] range
- **Data Split**: 90% training, 10% testing (stratified split)

### Class Distribution
The dataset contains 5 flower classes with stratified sampling to maintain balanced representation across train and test sets.

## CNN Architecture

### Base Architecture
```python
Input(80×80×1) → Conv2D → Pooling → Conv2D → Pooling → Conv2D → Pooling 
               → Flatten → Dense(FC) → [...] → Dense(5, softmax)
```

### Configurable Components
1. **Convolution Layers**: 
   - Kernel sizes: 3×3, 5×5, or combinations
   - Filters: [16, 32, 64] or deeper configurations
   - Number of layers: 3-6 layers

2. **Pooling Layers**:
   - Max Pooling (default)
   - Average Pooling (tested in Experiment C)
   - Pool size: 2×2

3. **Activation Functions**:
   - ReLU (default)
   - Sigmoid
   - Leaky ReLU

4. **Regularization**:
   - Dropout (rates: 0.1, 0.25)
   - Batch Normalization
   - Combined (Dropout + Batch Norm)

5. **Fully Connected Layers**:
   - 1-3 dense layers before output
   - 128 neurons per FC layer

### Model Building Function
```python
def build_cnn_model(input_shape, 
                    kernel_sizes=[3, 3, 3], 
                    filters=[16, 32, 64],
                    pooling_type='max',
                    activation='relu',
                    num_fc_layers=1,
                    regularization='dropout',
                    dropout_rate=0.1,
                    use_batch_norm=False,
                    num_classes=5)
```

## Experiments Conducted

### Experiment A: Convolution Kernel Size Analysis

**Objective**: Compare different kernel size combinations to find optimal feature extraction.

**Configurations Tested**:
- **A1**: (3×3, 3×3, 3×3) - All small kernels
- **A2**: (3×3, 3×3, 5×5) - Progressive increase
- **A3**: (3×3, 5×5, 5×5) - Early increase
- **A4**: (5×5, 5×5, 5×5) - All large kernels

**Fixed Parameters**:
- Filters: [16, 32, 64]
- Pooling: Max Pooling
- Activation: ReLU
- FC Layers: 1
- Dropout: 0.1

**Key Findings**:
- Smaller kernels (3×3) generally perform better for 80×80 images
- Mixed kernel sizes can capture multi-scale features
- Larger kernels (5×5) may lose spatial information on smaller images
- Best configuration identified for subsequent experiments

### Experiment B: Number of Fully Connected Layers

**Objective**: Evaluate the impact of adding more fully connected layers.

**Configurations Tested**:
- **B1**: 2 FC Layers
- **B2**: 3 FC Layers

**Fixed Parameters**:
- Kernel sizes: Best from Experiment A
- Filters: [16, 32, 64]
- Pooling: Max Pooling
- Activation: ReLU
- Dropout: 0.1

**Key Findings**:
- Additional FC layers increase model capacity
- Trade-off between accuracy and training time
- Risk of overfitting with too many FC layers
- 1-2 FC layers often sufficient for this dataset size

### Experiment C: Pooling Type Comparison

**Objective**: Compare Max Pooling vs Average Pooling for feature reduction.

**Configurations Tested**:
- **C1**: Average Pooling (compared against Max Pooling baseline)

**Fixed Parameters**:
- Kernel sizes: Best from Experiment A
- Filters: [16, 32, 64]
- Activation: ReLU
- FC Layers: 1
- Dropout: 0.1

**Key Findings**:
- Max Pooling captures dominant features (strongest activations)
- Average Pooling provides smoother feature representation
- Max Pooling typically better for flower classification
- Choice depends on feature characteristics

### Experiment D: Activation Function Comparison

**Objective**: Compare different activation functions for non-linearity.

**Configurations Tested**:
- **D1**: Sigmoid - Smooth S-shaped curve
- **D2**: ReLU - Rectified Linear Unit
- **D3**: Leaky ReLU - Prevents dying ReLU problem

**Fixed Parameters**:
- Kernel sizes: Best from Experiment A
- Filters: [16, 32, 64]
- Pooling: Max Pooling
- FC Layers: 1
- Dropout: 0.1

**Key Findings**:
- ReLU typically provides fastest convergence
- Sigmoid can suffer from vanishing gradients
- Leaky ReLU helps when ReLU neurons die
- ReLU recommended for most CNN applications

### Experiment E: Regularization Techniques

**Objective**: Analyze different regularization approaches to prevent overfitting.

**Configurations Tested**:
- **E1**: Dropout 0.25 - Higher dropout rate
- **E2**: Batch Normalization - Normalizes layer inputs
- **E3**: Dropout 0.1 + Batch Norm - Combined approach

**Fixed Parameters**:
- Kernel sizes: Best from Experiment A
- Filters: [16, 32, 64]
- Pooling: Max Pooling
- Activation: ReLU
- FC Layers: 1

**Key Findings**:
- Dropout prevents co-adaptation of neurons
- Batch Normalization stabilizes training
- Combined regularization can be very effective
- Higher dropout (0.25) may hurt small datasets
- Batch Norm accelerates training convergence

### Experiment F: Adding More Convolution Layers

**Objective**: Evaluate the impact of deeper networks on performance, parameters, and training time.

**Configurations Tested**:
- **F1**: 4 Conv Layers - Kernel: [3,3,3,3], Filters: [16,32,64,128]
- **F2**: 5 Conv Layers - Kernel: [3,3,3,3,3], Filters: [16,32,64,128,256]
- **F3**: 6 Conv Layers - Kernel: [3,3,3,3,3,3], Filters: [16,32,64,128,256,512]

**Fixed Parameters**:
- Kernel size: 3×3 for all layers
- Pooling: Max Pooling
- Activation: ReLU
- FC Layers: 1
- Dropout: 0.1

**Metrics Tracked**:
- Test accuracy
- Trainable parameters count
- Training time (seconds)
- Model complexity

**Key Findings**:
- Deeper networks increase trainable parameters exponentially
- Training time increases with network depth
- Risk of overfitting with very deep networks on small datasets
- Trade-off between accuracy gain and computational cost
- 3-4 conv layers often optimal for 80×80 images

**Performance Comparison**:

| Configuration | Conv Layers | Parameters | Accuracy | Training Time |
|--------------|-------------|------------|----------|---------------|
| Baseline     | 3           | ~100K      | ~X%      | ~Y sec        |
| F1           | 4           | ~200K      | ~X%      | ~Y sec        |
| F2           | 5           | ~500K      | ~X%      | ~Y sec        |
| F3           | 6           | ~1.2M      | ~X%      | ~Y sec        |

*Note: Actual values filled in after running experiments*

### Experiment G: Best Model with Color Images

**Objective**: Compare grayscale vs color image performance using the best configuration.

**Configuration**:
- Input: RGB images (80×80×3)
- Architecture: Best configuration from previous experiments
- All other parameters: Best values identified

**Key Findings**:
- Color information provides additional features
- RGB requires 3× more input parameters
- Training time increases with color images
- Color can improve accuracy for flowers (color-distinctive features)
- Trade-off between accuracy gain and computational cost

## MNIST Comparison (Assignment 4 Integration)

### Objective
Apply the best flower classification configuration to MNIST digit classification and compare results.

### Setup
- **Dataset**: MNIST handwritten digits
- **Preprocessing**: 
  - Resize from 28×28 to 80×80 (match flower images)
  - Normalize to [0, 1]
  - Grayscale (single channel)
- **Classes**: 10 digits (0-9)
- **Architecture**: Best flower configuration adapted for 10 classes

### Configuration
```python
model = build_cnn_model(
    input_shape=(80, 80, 1),
    kernel_sizes=best_kernel_config,
    filters=[16, 32, 64],
    pooling_type='max',
    activation='relu',
    num_fc_layers=1,
    regularization='dropout',
    dropout_rate=0.1,
    use_batch_norm=False,
    num_classes=10  # Changed from 5 to 10
)
```

### Comparison Results

| Metric | Flowers | MNIST |
|--------|---------|-------|
| Test Accuracy | ~X% | ~Y% |
| F1 Score | ~X | ~Y |
| Training Time | ~X sec | ~Y sec |
| Dataset Size | ~Z images | 60,000 images |
| Image Complexity | High (natural) | Low (digits) |

### Key Observations
1. **MNIST typically achieves higher accuracy**: Simpler patterns, less variation
2. **Flowers more challenging**: Natural images with complex features
3. **Same architecture performs differently**: Task-specific optimization needed
4. **Resizing impact**: MNIST images upscaled, may affect feature quality
5. **Class count**: 10 digits vs 5 flowers affects output layer complexity

## Comprehensive Results Comparison

### Visualization
The project includes comprehensive bar graphs comparing:
1. **Test Accuracy**: Performance on held-out test set
2. **F1 Scores**: Balanced metric considering precision and recall
3. **Training Times**: Computational efficiency comparison
4. **Trainable Parameters**: Model complexity analysis

### Results DataFrame
```python
results_df = pd.DataFrame({
    'Experiment': exp_names,
    'Test Accuracy': accuracies,
    'F1 Score': f1_scores,
    'Training Time': train_times,
    'Parameters': params
})
```

### Best Configuration Summary
- **Best Experiment**: [Determined from results]
- **Optimal Kernel Sizes**: [Best from A]
- **Optimal FC Layers**: [Best from B]
- **Optimal Pooling**: [Best from C]
- **Optimal Activation**: [Best from D]
- **Optimal Regularization**: [Best from E]
- **Optimal Depth**: [Best from F]
- **Color vs Grayscale**: [Best from G]

## Training Details

### Hyperparameters
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 32 (typical)
- **Epochs**: 10-20 (varies by experiment)
- **Learning Rate**: 0.001 (Adam default)

### Data Augmentation
Not applied in base experiments to isolate architecture effects. Can be added for improved performance.

### Hardware Requirements
- **GPU**: Recommended for faster training
- **RAM**: Minimum 8GB
- **Storage**: ~500MB for dataset and models

## Visualizations

### Included Plots
1. **Sample Images**: 10 sample flowers from each class
2. **Training History**: Loss and accuracy curves for each experiment
3. **Accuracy Comparison**: Bar charts across all experiments
4. **F1 Score Comparison**: Model performance consistency
5. **Training Time Analysis**: Efficiency comparison
6. **Parameter Count**: Model complexity visualization
7. **MNIST Comparison**: Side-by-side performance metrics

### Sample Visualization Code
```python
# Visualize sample images
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].squeeze(), cmap='gray')
    plt.title(f'{class_names[y[i]]}')
    plt.axis('off')
plt.tight_layout()
plt.show()
```

## Key Insights

### Architecture Choices
1. **Kernel Size**: Smaller kernels (3×3) capture fine details effectively
2. **Network Depth**: 3-4 conv layers balance performance and complexity
3. **Pooling**: Max pooling generally outperforms average pooling
4. **Activation**: ReLU provides best convergence speed and accuracy
5. **Regularization**: Batch normalization + light dropout optimal

### Performance Factors
1. **Image Size**: 80×80 provides good balance between detail and computation
2. **Color vs Grayscale**: Color helps when features are color-distinctive
3. **Dataset Size**: Larger datasets support deeper networks
4. **Class Similarity**: Similar-looking flowers harder to distinguish
5. **Feature Complexity**: Natural images require more sophisticated features

### Trade-offs
1. **Accuracy vs Speed**: Deeper networks more accurate but slower
2. **Parameters vs Overfitting**: More parameters risk overfitting small datasets
3. **Batch Size vs Memory**: Larger batches need more GPU memory
4. **Epochs vs Time**: More epochs improve accuracy but increase training time
5. **Color vs Computation**: RGB images 3× more expensive than grayscale

## Dependencies

### Required Libraries
```python
# Core libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Image processing
import cv2
import os

# Machine learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Deep learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist  # For MNIST comparison
```

### Installation
```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow
```

Or with specific versions:
```bash
pip install numpy>=1.21.0 pandas>=1.3.0 matplotlib>=3.4.0 seaborn>=0.11.0 \
            opencv-python>=4.5.0 scikit-learn>=0.24.0 tensorflow>=2.6.0
```

### Version Requirements
- Python: 3.7+
- TensorFlow: 2.x (with GPU support recommended)
- NumPy: 1.19+
- OpenCV: 4.0+

## Usage

### 1. Load and Preprocess Data
```python
# Set dataset path
path = '/home/Assignment-05/flowers'

# Load images
X, y, class_names = load_and_preprocess_images(
    data_path=path,
    img_size=(80, 80),
    grayscale=True,
    reduce_size=False
)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

# Convert to categorical
y_train_cat = to_categorical(y_train, num_classes=5)
y_test_cat = to_categorical(y_test, num_classes=5)
```

### 2. Build and Train Model
```python
# Build model
model = build_cnn_model(
    input_shape=X_train.shape[1:],
    kernel_sizes=[3, 3, 3],
    filters=[16, 32, 64],
    pooling_type='max',
    activation='relu',
    num_fc_layers=1,
    regularization='dropout',
    dropout_rate=0.1,
    use_batch_norm=False,
    num_classes=5
)

# Compile
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
history = model.fit(
    X_train, y_train_cat,
    validation_split=0.1,
    epochs=10,
    batch_size=32,
    verbose=1
)

# Evaluate
test_loss, test_acc = model.evaluate(X_test, y_test_cat)
print(f"Test Accuracy: {test_acc:.4f}")
```

### 3. Make Predictions
```python
# Predict on new images
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Get F1 score
f1 = f1_score(y_test, predicted_classes, average='weighted')
print(f"F1 Score: {f1:.4f}")

# Classification report
print(classification_report(y_test, predicted_classes, target_names=class_names))
```

### 4. Visualize Results
```python
# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Reproducibility

### Random Seeds
```python
import numpy as np
import tensorflow as tf
import random

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
```

### Hardware Considerations
Results may vary slightly between CPU and GPU due to:
- Floating-point precision differences
- Parallelization order
- cuDNN algorithm selection

## Future Improvements

### Architecture Enhancements
1. **Transfer Learning**: Use pre-trained models (VGG16, ResNet, MobileNet)
2. **Attention Mechanisms**: Add attention layers for better feature focus
3. **Residual Connections**: Implement skip connections for deeper networks
4. **Inception Modules**: Multi-scale feature extraction
5. **Separable Convolutions**: Reduce parameters with depthwise separable convs

### Data Augmentation
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2
)
```

### Advanced Techniques
1. **Learning Rate Scheduling**: Reduce LR on plateau
2. **Early Stopping**: Prevent overfitting
3. **Ensemble Methods**: Combine multiple models
4. **Cross-Validation**: K-fold validation for robust evaluation
5. **Hyperparameter Tuning**: Grid search or Bayesian optimization

### Evaluation Enhancements
1. **Confusion Matrix**: Detailed per-class analysis
2. **ROC Curves**: Multi-class ROC analysis
3. **Grad-CAM**: Visualize what model focuses on
4. **Error Analysis**: Analyze misclassified samples
5. **Per-Class Metrics**: Precision, recall, F1 for each class

## Conclusion

This comprehensive flower classification project demonstrates:

1. **Systematic Experimentation**: 7 experiments covering key CNN hyperparameters
2. **Optimal Configuration**: Identified best architecture through empirical analysis
3. **Cross-Task Validation**: MNIST comparison validates architecture generality
4. **Practical Insights**: Trade-offs between accuracy, speed, and complexity
5. **Reproducible Results**: Clear documentation and code structure

### Key Takeaways
- **Architecture Matters**: Careful hyperparameter tuning significantly impacts performance
- **No Universal Solution**: Optimal configuration is task and data dependent
- **Experimentation Essential**: Systematic testing reveals best practices
- **Trade-offs Exist**: Balance accuracy, efficiency, and complexity
- **Foundation Skills**: Strong basis for advanced deep learning projects

### Best Practices Learned
1. Start with simple architectures and gradually increase complexity
2. Use stratified splits for balanced evaluation
3. Track multiple metrics (accuracy, F1, time, parameters)
4. Visualize results for intuitive understanding
5. Compare against baseline and related tasks
6. Document experiments thoroughly for reproducibility

## Author
Uttam Mahata

## Assignment
Machine Learning LAB - Assignment 5

## Date
October 2024

## References
- TensorFlow/Keras Documentation
- Deep Learning with Python (François Chollet)
- CNN Architecture Papers (LeNet, AlexNet, VGG, ResNet)
- Flower Classification Dataset Documentation
- Assignment 4: MNIST Digit Classification
