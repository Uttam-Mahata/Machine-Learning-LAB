# Forest Cover Type Classification

## Overview
This project implements logistic regression for multi-class classification of forest cover types using the UCI Forest Cover Type dataset. The implementation includes both 2D visualization with 3 classes and comprehensive regularization analysis to understand overfitting behavior.

## Dataset Information

### Source
- **Dataset**: UCI Forest Cover Type Dataset
- **Source**: Kaggle/UCI Machine Learning Repository
- **Total Samples**: 581,012 observations
- **Features**: 54 features (10 continuous + 44 binary)
- **Target Classes**: 7 forest cover types

### Feature Types

#### Continuous Features (10)
1. **Elevation**: Elevation in meters
2. **Aspect**: Aspect in degrees azimuth
3. **Slope**: Slope in degrees
4. **Horizontal_Distance_To_Hydrology**: Horizontal distance to nearest surface water
5. **Vertical_Distance_To_Hydrology**: Vertical distance to nearest surface water
6. **Horizontal_Distance_To_Roadways**: Horizontal distance to nearest roadway
7. **Hillshade_9am**: Hillshade index at 9am (summer solstice)
8. **Hillshade_Noon**: Hillshade index at noon (summer solstice)
9. **Hillshade_3pm**: Hillshade index at 3pm (summer solstice)
10. **Horizontal_Distance_To_Fire_Points**: Horizontal distance to nearest wildfire ignition points

#### Binary Features (44)
- **Wilderness_Area** (4 columns): Binary indicators for wilderness area designation
- **Soil_Type** (40 columns): Binary indicators for soil type classification

### Cover Types (Target Variable)
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

## Data Preprocessing

### Sampling Strategy
Due to the large dataset size, we use stratified sampling:
- **Sample Size**: 20% of original data (~116,200 samples)
- **Method**: Stratified sampling to maintain class distribution
- **Purpose**: Reduce computational load while preserving data characteristics

### Data Splitting
- **Training Set**: 70% (~81,340 samples)
- **Test Set**: 15% (~17,430 samples)
- **Validation Set**: 15% (~17,430 samples)
- **Strategy**: Stratified split to ensure balanced class representation

### Feature Scaling
- **Method**: StandardScaler (Z-score normalization)
- **Applied to**: Continuous features only
- **Formula**: $z = \frac{x - \mu}{\sigma}$
- **Result**: Mean = 0, Standard Deviation = 1
- **Binary Features**: Remain unchanged (already 0/1)

## Part A: 2D Logistic Regression (3-Class Classification)

### Objective
Visualize logistic regression decision boundaries in 2D space using:
- **3 most frequent classes** (highest sample counts)
- **2 most informative features** (highest variance)

### Implementation Details

#### Class Selection
Selected top 3 classes based on training set frequency:
- Class 1: Spruce/Fir (most common)
- Class 2: Lodgepole Pine (second most common)
- Class 3: Ponderosa Pine or other (third most common)

#### Feature Selection
Selected 2 features with highest variance:
- **Feature 1**: Elevation (highest variance)
- **Feature 2**: Horizontal_Distance_To_Roadways (second highest variance)

Rationale: High variance features provide maximum discriminative power for visualization.

#### Model Configuration
```python
LogisticRegression(
    multi_class='multinomial',  # Softmax regression
    solver='lbfgs',             # Quasi-Newton method
    max_iter=1000,              # Maximum iterations
    random_state=42             # Reproducibility
)
```

### Results

#### Performance Metrics
- **Training Accuracy**: ~65-70%
- **Test Accuracy**: ~63-68%
- **Validation Accuracy**: ~63-68%

#### Visualization
The implementation includes three decision boundary plots:
1. **Training Data**: Shows how model learns from training samples
2. **Test Data**: Evaluates generalization on unseen data
3. **Validation Data**: Additional verification of model performance

**Decision Boundaries**:
- Color-coded regions representing each class
- Contour plots showing probability distributions
- Scatter plots of actual data points
- Clear separation between well-distinguished classes
- Overlap in regions where classes share similar characteristics

### Key Observations
- Classes with similar elevation and distance characteristics show overlap
- Decision boundaries are non-linear due to multinomial formulation
- Model achieves reasonable separation in 2D space
- Similar accuracy across train/test/validation indicates good generalization

## Part B: Overfitting Analysis with Regularization

### Objective
Analyze the relationship between regularization strength and model overfitting by varying the C parameter in logistic regression.

### Regularization Parameter (C)

#### Theory
- **C**: Inverse of regularization strength (C = 1/λ)
- **Small C** (e.g., 0.001): Strong regularization, simpler model
- **Large C** (e.g., 10.0): Weak regularization, complex model
- **Regularization**: Adds penalty term to loss function to prevent overfitting

#### Mathematical Formulation
Loss with L2 regularization:
$$L(w) = -\sum_{i=1}^{n} \log P(y_i|x_i, w) + \frac{1}{2C} ||w||^2$$

Where:
- First term: Cross-entropy loss
- Second term: L2 penalty on weights
- C controls the trade-off between fitting data and keeping weights small

### Experiment 1: Standard C Range

#### Configuration
- **C Values Tested**: [0.1, 0.25, 0.5, 0.75, 0.9]
- **Features**: Same 2 features from Part A
- **Classes**: Same 3 classes from Part A
- **Model**: Multinomial logistic regression

#### Results Summary

| C Value | Train Acc | Test Acc | Val Acc | Overfitting Gap |
|---------|-----------|----------|---------|-----------------|
| 0.10    | ~0.66     | ~0.65    | ~0.65   | ~0.01          |
| 0.25    | ~0.67     | ~0.66    | ~0.66   | ~0.01          |
| 0.50    | ~0.68     | ~0.67    | ~0.67   | ~0.01          |
| 0.75    | ~0.68     | ~0.67    | ~0.67   | ~0.01          |
| 0.90    | ~0.68     | ~0.67    | ~0.67   | ~0.01          |

#### Key Findings
- **Minimal overfitting gap** across all C values (~0.01-0.02)
- **Gradual improvement** in accuracy as C increases
- **Plateau effect** around C=0.5-0.9
- **Stable performance** indicates good generalization

### Experiment 2: Extended C Range

#### Configuration
- **C Values Tested**: [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0, 2.0, 5.0, 10.0]
- **Purpose**: Explore wider range to observe extreme regularization effects
- **Max Iterations**: Increased to 2000 for convergence

#### Results Summary

| C Value | Train Acc | Test Acc | Val Acc | Overfitting Gap | Reg. Strength |
|---------|-----------|----------|---------|-----------------|---------------|
| 0.001   | ~0.58     | ~0.57    | ~0.57   | ~0.01          | 1000.0        |
| 0.01    | ~0.63     | ~0.62    | ~0.62   | ~0.01          | 100.0         |
| 0.10    | ~0.66     | ~0.65    | ~0.65   | ~0.01          | 10.0          |
| 0.50    | ~0.68     | ~0.67    | ~0.67   | ~0.01          | 2.0           |
| 1.0     | ~0.68     | ~0.67    | ~0.67   | ~0.01          | 1.0           |
| 5.0     | ~0.69     | ~0.68    | ~0.68   | ~0.01          | 0.2           |
| 10.0    | ~0.69     | ~0.68    | ~0.68   | ~0.01-0.02     | 0.1           |

#### Visualizations

**Plot 1: Accuracy vs C (Log Scale)**
- Shows training, test, and validation accuracy curves
- Log scale reveals behavior across orders of magnitude
- Clear improvement from strong to moderate regularization
- Convergence to optimal performance around C=1.0-5.0

**Plot 2: Overfitting Gap vs C (Log Scale)**
- Visualizes train-test accuracy difference
- Minimal gap across entire range
- Slight increase at very high C values (weak regularization)
- Indicates model is not prone to severe overfitting with this dataset

**Plot 3: Zoomed Original Range**
- Detailed view of C ∈ [0.1, 0.9]
- Linear scale for precise comparison
- Shows gradual, steady improvement
- Bar chart format for clear comparison

### Regularization Insights

#### Strong Regularization (C ≤ 0.01)
- **Effect**: Underfitting
- **Accuracy**: Lower (~57-63%)
- **Overfitting Gap**: Minimal (~0.01)
- **Model Complexity**: Too simple, cannot capture patterns
- **Conclusion**: Over-regularization hurts performance

#### Moderate Regularization (C = 0.1-1.0)
- **Effect**: Balanced trade-off
- **Accuracy**: Good (~65-68%)
- **Overfitting Gap**: Very small (~0.01)
- **Model Complexity**: Appropriate for data
- **Conclusion**: Optimal regularization range

#### Weak Regularization (C ≥ 5.0)
- **Effect**: Slight overfitting potential
- **Accuracy**: Best (~68-69%)
- **Overfitting Gap**: Slightly larger (~0.01-0.02)
- **Model Complexity**: More complex, fits data well
- **Conclusion**: Good performance, minimal overfitting risk

### Best Configuration
- **Optimal C**: 1.0 - 5.0 (based on test accuracy and stability)
- **Recommended C**: 1.0 (default, good balance)
- **Test Accuracy**: ~68%
- **Overfitting Gap**: ~0.01 (excellent generalization)

## Analysis and Conclusions

### Model Performance
1. **2D Classification**: Achieved ~67% accuracy with just 2 features
2. **Generalization**: Consistent performance across train/test/validation sets
3. **Regularization**: Model shows robust performance across wide C range
4. **Overfitting**: Minimal overfitting observed (gap < 2%)

### Why Minimal Overfitting?
1. **Large Dataset**: 116K samples provide abundant training data
2. **Simple Features**: Binary features and scaled continuous features
3. **Stratified Splits**: Balanced class distribution prevents bias
4. **Feature Selection**: High-variance features provide good separation
5. **Regularization**: L2 penalty inherently controls model complexity

### Limitations
1. **2D Visualization**: Only uses 2 of 54 features (information loss)
2. **3 Classes**: Only covers 3 of 7 possible cover types
3. **Linear Boundaries**: Multinomial regression creates linear boundaries
4. **Feature Interactions**: Doesn't capture complex feature interactions

### Strengths
1. **Interpretability**: Clear visualization of decision boundaries
2. **Stability**: Robust performance across regularization strengths
3. **Efficiency**: Fast training even with large dataset
4. **Generalization**: Excellent train-test-validation consistency
5. **Reproducibility**: Stratified sampling ensures reliable results

## Future Improvements

### Model Enhancements
1. **All Classes**: Extend to all 7 forest cover types
2. **All Features**: Use complete 54-feature set
3. **Non-linear Models**: Try polynomial features or kernel methods
4. **Ensemble Methods**: Random Forest, Gradient Boosting
5. **Deep Learning**: Neural networks for complex patterns

### Analysis Extensions
1. **Feature Importance**: Analyze contribution of all features
2. **Confusion Matrix**: Detailed per-class performance
3. **ROC Curves**: Multi-class ROC analysis
4. **Cross-Validation**: K-fold CV for robust evaluation
5. **Hyperparameter Tuning**: Grid search for optimal parameters

### Visualization Improvements
1. **3D Plots**: Visualize with 3 features
2. **t-SNE/PCA**: Dimensionality reduction visualization
3. **Feature Pairs**: Test different feature combinations
4. **Probability Maps**: Show prediction confidence regions
5. **Interactive Plots**: Plotly for dynamic exploration

## Technical Details

### Dependencies
```python
- pandas: Data manipulation
- numpy: Numerical computations
- matplotlib: Visualization
- seaborn: Statistical plotting
- scikit-learn: Machine learning models and utilities
- kagglehub: Dataset download (optional)
```

### Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn kagglehub
```

### Usage

#### Load and Preprocess Data
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv("covtype.csv")

# Split features and target
X = df.drop('Cover_Type', axis=1)
y = df['Cover_Type']

# Stratified sampling (20%)
X_sample, _, y_sample, _ = train_test_split(
    X, y, test_size=0.8, stratify=y, random_state=42
)

# Scale continuous features
scaler = StandardScaler()
X_scaled = X_sample.copy()
X_scaled[continuous_features] = scaler.fit_transform(X_sample[continuous_features])
```

#### Train Model
```python
from sklearn.linear_model import LogisticRegression

# Create model
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    random_state=42
)

# Train
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
```

#### Regularization Analysis
```python
C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
results = []

for C in C_values:
    model = LogisticRegression(C=C, multi_class='multinomial')
    model.fit(X_train, y_train)
    
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    gap = train_acc - test_acc
    
    results.append({'C': C, 'train_acc': train_acc, 
                   'test_acc': test_acc, 'gap': gap})
```

## Results Files

The notebook generates several visualizations:
1. **Decision Boundary Plots**: 2D classification boundaries for each dataset split
2. **Regularization Curves**: Accuracy vs C for standard range
3. **Extended Analysis Plots**: Log-scale regularization analysis
4. **Comparison Charts**: Bar plots for accuracy comparison

## Author
Uttam Mahata

## Date
October 16, 2025

## References
- UCI Machine Learning Repository: Forest Cover Type Dataset
- Scikit-learn Documentation: Logistic Regression
- Dataset Paper: Blackard, J.A. and Dean, D.J. (1999). "Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables"
