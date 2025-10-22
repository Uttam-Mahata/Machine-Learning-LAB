# MNIST Digit Classification with Neural Networks

## Overview
This project implements a comprehensive neural network experiment for handwritten digit classification using the MNIST dataset. The implementation includes RBF (Radial Basis Function) transformation to convert 28×28 images to 32×32, and conducts multiple experiments to find optimal hyperparameters.

## Dataset
- **Source**: MNIST handwritten digit dataset
- **Training Size**: 6,000 samples (1/10 of original 60,000)
- **Test Size**: 10,000 samples
- **Image Dimensions**: 
  - Original: 28×28 pixels
  - Transformed: 32×32 pixels (using RBF transformation)
- **Classes**: 10 digits (0-9)

## Data Preprocessing

### RBF Transformation
The images are transformed from 28×28 to 32×32 using two RBF methods:

1. **Gaussian Filter RBF**: Applies a Gaussian-based transformation
   - Pads images with 2 pixels on each side
   - Applies RBF factor based on image center and standard deviation
   
2. **SciPy RBF Interpolation**: Uses true RBF interpolation
   - Creates interpolator from original image data
   - Applies to new 32×32 grid
   - Uses 'linear' kernel for better image preservation

### Data Splitting
- **Training**: 80% (4,800 samples)
- **Validation**: 10% (600 samples)
- **Test**: 10% (600 samples)
- **Original MNIST Test**: 10,000 samples (for final evaluation)

## Neural Network Architecture

### Base Architecture
- **Input Layer**: 1024 neurons (32×32 flattened)
- **Hidden Layers**: Configurable (experiments with [16], [16,32], [16,32,64])
- **Output Layer**: 10 neurons (one per digit class)
- **Activation Functions**: Sigmoid, Tanh, ReLU

### Model Implementation
```python
Input(1024) → Dense(hidden_layers) → [Dropout] → ... → Dense(10, softmax)
```

## Experiments Conducted

### Experiment 1: Optimizer and Architecture Comparison
**Objective**: Compare SGD vs Adam optimizers with different hidden layer configurations

**Configurations**:
- Optimizers: SGD, Adam
- Hidden Layers: [16], [16, 32], [16, 32, 64]
- Loss Function: Categorical Crossentropy
- Epochs: 10
- Activation: Sigmoid

**Key Findings**:
- Adam optimizer consistently outperforms SGD
- Deeper networks (more hidden layers) generally perform better
- Best accuracy achieved with Adam optimizer

### Experiment 2: Activation Function Comparison
**Objective**: Compare different activation functions

**Configurations**:
- Activation Functions: Sigmoid, Tanh, ReLU
- Hidden Layers: [16, 32, 64]
- Optimizer: Adam (learning rate: 0.001)
- Epochs: 10

**Key Findings**:
- ReLU activation typically provides best performance
- Faster convergence compared to Sigmoid and Tanh
- Better gradient flow in deeper networks

### Experiment 3: Dropout Regularization
**Objective**: Evaluate impact of dropout on model performance

**Configurations**:
- Dropout Rates: 0.0, 0.25, 0.5, 0.75
- Hidden Layers: [16, 32, 64]
- Activation: ReLU
- Optimizer: Adam
- Epochs: 10

**Key Findings**:
- Moderate dropout (0.25-0.5) helps prevent overfitting
- Too much dropout (0.75) can hurt performance
- No dropout works well with small dataset

### Experiment 4: Learning Rate Optimization
**Objective**: Find optimal learning rate for fastest and best convergence

**Configurations**:
- Learning Rates: 0.01, 0.001, 0.005, 0.0001
- Hidden Layers: [16, 32, 64]
- Activation: ReLU
- Epochs: 15

**Metrics Tracked**:
- Test accuracy
- Training time
- Time to achieve best validation accuracy
- Best validation accuracy

**Key Findings**:
- Learning rate of 0.001 often provides good balance
- Higher learning rates converge faster but may be unstable
- Lower learning rates are more stable but slower

## Results Summary

### Best Model Configuration
- **Architecture**: [16, 32, 64] hidden layers
- **Activation**: ReLU
- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Dropout**: 0.0
- **Test Accuracy**: ~81%

### Training Performance
- **Training Set Accuracy**: ~98.5% (after 50 epochs)
- **Validation Accuracy**: ~80%
- **Test Accuracy**: ~81%
- **Training Time**: ~100 seconds (50 epochs)

### Handwritten Digit Recognition Test
Tested on 5 real MNIST samples (digits 0-4):
- **Overall Accuracy**: 80% (4/5 correct)
- **Correct Predictions**: 
  - Digit 0: ✓ (100.0% confidence)
  - Digit 1: ✓ (94.8% confidence)
  - Digit 2: ✓ (100.0% confidence)
  - Digit 4: ✓ (67.2% confidence)
- **Incorrect Predictions**:
  - Digit 3 → Predicted as 8 (96.0% confidence)
    - Note: 3 and 8 are visually similar in handwriting

## Visualizations

The project includes comprehensive visualizations:

1. **Training Curves**: Loss and accuracy over epochs for each experiment
2. **Comparison Plots**: Side-by-side comparison of different configurations
3. **Digit Recognition Results**: 
   - Original 28×28 images
   - Transformed 32×32 images with predictions
   - Prediction probability distributions

## Key Observations

### Strengths
- Effective RBF transformation maintains image quality
- Good generalization despite reduced training size (1/10 of original)
- High confidence predictions for most digits
- Fast training with moderate network size

### Challenges
- Confusion between visually similar digits (3 vs 8, 4 vs 9)
- Overfitting evident from training vs validation accuracy gap
- RBF transformation adds computational overhead

### Improvements Achieved
- Fixed synthetic digit generation by using real MNIST samples
- Improved from 20% to 80% accuracy on handwritten digit test
- Added comprehensive evaluation metrics and visualizations
- Implemented proper train-validation-test split

## Dependencies

```python
- tensorflow/keras: Deep learning framework
- numpy: Numerical computations
- matplotlib: Visualization
- scikit-learn: Data splitting and metrics
- scipy: RBF interpolation
```

## Usage

### Training the Model
```python
# Load and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Apply RBF transformation
x_train_32x32 = convert_28x28_to_32x32_rbf(x_train)

# Create and train model
model = create_model([16, 32, 64], activation='relu')
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])
model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50)
```

### Making Predictions
```python
# Prepare test image
test_image_32x32 = convert_28x28_to_32x32_rbf(test_image)
test_image_flat = test_image_32x32.reshape(1, -1)

# Predict
prediction = model.predict(test_image_flat)
predicted_digit = np.argmax(prediction)
confidence = np.max(prediction) * 100
```

## Future Enhancements

1. **Model Architecture**:
   - Implement Convolutional Neural Networks (CNNs)
   - Try different RBF kernel functions
   - Experiment with batch normalization

2. **Data Augmentation**:
   - Add rotation, scaling, and translation
   - Implement elastic deformations
   - Synthetic data generation

3. **Optimization**:
   - Learning rate scheduling
   - Early stopping implementation
   - Ensemble methods

4. **Evaluation**:
   - Confusion matrix analysis
   - Per-class performance metrics
   - Error analysis on misclassified samples

## Conclusion

This project successfully demonstrates:
- Effective neural network training on MNIST dataset
- Systematic hyperparameter tuning through multiple experiments
- RBF transformation for image resizing
- Comprehensive evaluation and visualization

The final model achieves **~81% test accuracy** with a relatively simple architecture, trained on only 1/10 of the original dataset. The model shows strong performance on most digit classes, with expected confusion on visually similar digits.

## Author
Uttam Mahata

## Date
October 16, 2025
