# Machine Learning LAB

A comprehensive collection of machine learning experiments and assignments covering fundamental to advanced concepts in supervised learning, deep learning, and neural networks.

## 📚 Repository Overview

This repository contains practical implementations and experiments from a Machine Learning laboratory course. Each assignment explores different aspects of machine learning, from classical linear regression to modern convolutional neural networks.

## 🗂️ Repository Structure

```
Machine-Learning-LAB/
├── Assignment 1 - Linear Regression/     # House price prediction
├── Assignment-2/                          # Breast cancer classification
├── Assignment-3/                          # Forest cover type classification
├── Assignment-4/                          # MNIST digit recognition
├── Assignment-5/                          # Flower classification with CNN
├── requirements.txt                       # Python dependencies
├── pyproject.toml                        # Project configuration
└── README.md                             # This file
```

## 📋 Assignments

### Assignment 1: Linear Regression - House Price Prediction

**Dataset**: Ames Housing Dataset  
**Techniques**: Linear Regression, Feature Engineering, Data Preprocessing  
**Key Files**: `house_price_classification.ipynb`

**Description**:
- Implements linear regression for predicting house prices
- Explores feature relationships and correlations
- Includes comprehensive data visualization and analysis
- Dataset contains 80+ features describing house characteristics

**Key Features**:
- Data preprocessing and cleaning
- Feature selection and engineering
- Model training and evaluation
- Visualization of predictions vs actual values

---

### Assignment 2: Logistic Regression - Breast Cancer Diagnosis

**Dataset**: Wisconsin Diagnostic Breast Cancer (WDBC)  
**Techniques**: Logistic Regression, Binary Classification, Feature Scaling  
**Key Files**: `wisconsin.ipynb`, `wisconsin.py`

**Description**:
- Binary classification for breast cancer diagnosis (Malignant vs Benign)
- Uses 30 features computed from digitized images of fine needle aspirate (FNA)
- Implements logistic regression from scratch and with scikit-learn

**Dataset Details**:
- 569 samples with 30 numerical features
- Features include radius, texture, perimeter, area, smoothness, etc.
- Each feature computed as mean, standard error, and "worst" value

**Key Features**:
- Feature standardization and normalization
- Model accuracy analysis
- Confusion matrix and classification metrics
- Comparison of different regularization parameters

---

### Assignment 3: Multi-class Logistic Regression - Forest Cover Type

**Dataset**: UCI Forest Cover Type Dataset  
**Techniques**: Multinomial Logistic Regression, Regularization, 2D Visualization  
**Key Files**: `forest_cover.ipynb`, `README.md`

**Description**:
- Multi-class classification of forest cover types (7 classes)
- 581,012 observations with 54 features (10 continuous + 44 binary)
- Includes overfitting analysis with regularization parameter tuning

**Key Components**:

**Part A - 2D Visualization**:
- 3-class classification using 2 most informative features
- Decision boundary visualization
- Accuracy: ~65-68% on test set

**Part B - Regularization Analysis**:
- Tests regularization parameter C from 0.001 to 10.0
- Analyzes overfitting gap between train and test accuracy
- Minimal overfitting observed (gap < 2%)
- Optimal C range: 1.0 - 5.0

**Key Insights**:
- Strong regularization (C ≤ 0.01): Underfitting (~57-63% accuracy)
- Moderate regularization (C = 0.1-1.0): Balanced (~65-68% accuracy)
- Weak regularization (C ≥ 5.0): Best performance (~68-69% accuracy)

---

### Assignment 4: Neural Networks - MNIST Digit Classification

**Dataset**: MNIST Handwritten Digits  
**Techniques**: Neural Networks, RBF Transformation, Hyperparameter Tuning  
**Key Files**: `mnist.ipynb`, `main.ipynb`, `README.md`

**Description**:
- Implements neural networks for handwritten digit recognition
- Uses RBF (Radial Basis Function) transformation to convert 28×28 to 32×32 images
- Training on 6,000 samples, testing on 10,000 samples

**Experiments Conducted**:

1. **Optimizer Comparison**: SGD vs Adam with different architectures
2. **Activation Functions**: Sigmoid, Tanh, ReLU comparison
3. **Dropout Regularization**: Tests dropout rates from 0.0 to 0.75
4. **Learning Rate Optimization**: Tests rates from 0.0001 to 0.01

**Best Configuration**:
- Architecture: [16, 32, 64] hidden layers
- Activation: ReLU
- Optimizer: Adam (learning rate: 0.001)
- Test Accuracy: ~81%

**Key Features**:
- RBF-based image transformation
- Comprehensive hyperparameter experiments
- Visualization of training curves
- Real digit prediction with confidence scores

---

### Assignment 5: CNN - Flower Classification

**Dataset**: Custom Flower Images Dataset  
**Techniques**: Convolutional Neural Networks, Transfer Learning Concepts  
**Key Files**: `flower_classification.ipynb`, `README.md`

**Description**:
- Implements CNN for flower classification (5 classes)
- Systematic hyperparameter experiments to find optimal configuration
- Includes comparison with MNIST dataset using same architecture

**CNN Architecture Experiments**:

1. **Experiment A**: Convolution kernel size analysis (3×3, 5×5 combinations)
2. **Experiment B**: Number of fully connected layers (1-3 layers)
3. **Experiment C**: Pooling type comparison (Max vs Average)
4. **Experiment D**: Activation functions (Sigmoid, ReLU, Leaky ReLU)
5. **Experiment E**: Regularization techniques (Dropout, Batch Norm, Combined)
6. **Experiment F**: Network depth (3-6 convolutional layers)
7. **Experiment G**: Color vs Grayscale images

**Key Results**:
- Image size: 80×80 pixels
- Best accuracy achieved with optimized configuration
- Comprehensive comparison of training time vs accuracy
- Parameter count analysis for each configuration

**MNIST Comparison**:
- Applied best flower configuration to MNIST
- Resized MNIST from 28×28 to 80×80
- Cross-task validation of architecture effectiveness

---

## 🛠️ Technologies & Libraries

### Core Libraries
- **Python**: 3.12+
- **NumPy**: Numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **Matplotlib & Seaborn**: Data visualization and plotting

### Machine Learning
- **scikit-learn**: Classical ML algorithms and preprocessing
  - Linear/Logistic Regression
  - StandardScaler, train_test_split
  - Metrics and evaluation tools
- **TensorFlow/Keras**: Deep learning frameworks
  - Neural network layers and models
  - Optimizers (SGD, Adam)
  - Callbacks and training utilities

### Additional Tools
- **Jupyter Notebook/Lab**: Interactive development environment
- **SciPy**: Scientific computing (RBF interpolation)
- **OpenCV**: Image processing (Assignment 5)
- **KaggleHub**: Dataset downloading
- **UCI ML Repo**: Dataset access

---

## 📦 Installation & Setup

### Prerequisites
- Python 3.12 or higher
- pip or uv package manager

### Option 1: Using pip

```bash
# Clone the repository
git clone https://github.com/Uttam-Mahata/Machine-Learning-LAB.git
cd Machine-Learning-LAB

# Install dependencies
pip install -r requirements.txt
```

### Option 2: Using uv (faster)

```bash
# Clone the repository
git clone https://github.com/Uttam-Mahata/Machine-Learning-LAB.git
cd Machine-Learning-LAB

# Install with uv
uv sync
```

### Starting Jupyter

```bash
# Launch Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

---

## 🚀 Usage

### Running Individual Assignments

Each assignment is self-contained in its directory. Navigate to the assignment folder and open the Jupyter notebook:

```bash
# Example: Assignment 1
cd "Assignment 1 - Linear Regression"
jupyter notebook house_price_classification.ipynb

# Example: Assignment 3
cd Assignment-3
jupyter notebook forest_cover.ipynb

# Example: Assignment 5
cd Assignment-5
jupyter notebook flower_classification.ipynb
```

### Running Python Scripts

Some assignments include standalone Python scripts:

```bash
# Assignment 2
cd Assignment-2
python wisconsin.py
```

---

## 📊 Datasets

### 1. Ames Housing Dataset
- **Source**: Kaggle / Ames, Iowa housing data
- **Samples**: ~1,460 houses
- **Features**: 80+ features
- **Task**: Regression (house price prediction)

### 2. Wisconsin Breast Cancer Dataset (WDBC)
- **Source**: UCI Machine Learning Repository
- **Samples**: 569 patients
- **Features**: 30 numerical features
- **Task**: Binary classification (Malignant vs Benign)

### 3. Forest Cover Type Dataset
- **Source**: UCI ML Repository / Kaggle
- **Samples**: 581,012 observations
- **Features**: 54 features (10 continuous + 44 binary)
- **Task**: Multi-class classification (7 forest types)

### 4. MNIST Handwritten Digits
- **Source**: MNIST Database
- **Samples**: 60,000 training, 10,000 test
- **Features**: 28×28 grayscale images
- **Task**: 10-class digit recognition

### 5. Flower Images
- **Source**: Custom flower dataset
- **Classes**: 5 flower types
- **Features**: 80×80 RGB/Grayscale images
- **Task**: Multi-class image classification

---

## 📈 Key Learning Outcomes

### Supervised Learning
- Linear and logistic regression implementation
- Feature engineering and selection
- Regularization techniques (L1, L2)
- Overfitting analysis and prevention

### Deep Learning
- Neural network architecture design
- Activation functions and their impacts
- Optimization algorithms (SGD, Adam)
- Hyperparameter tuning strategies

### Convolutional Neural Networks
- CNN layer configurations
- Pooling strategies (Max, Average)
- Regularization (Dropout, Batch Normalization)
- Transfer learning concepts

### Model Evaluation
- Train-validation-test split strategies
- Cross-validation techniques
- Performance metrics (Accuracy, F1-score, Confusion Matrix)
- Overfitting detection and mitigation

### Data Processing
- Feature scaling and normalization
- Stratified sampling
- Image preprocessing and augmentation
- RBF transformations

---

## 📝 Assignment Reports

Several assignments include comprehensive PDF reports and LaTeX source files:
- Assignment 1: `ML_LAB_ASSIGNMENT_01.pdf`
- Assignment 2: `main.tex`
- Assignment 3: `main.tex`, detailed `README.md`
- Assignment 4: Extensive `README.md` with experiment results
- Assignment 5: Comprehensive `README.md` with all experiments

---

## 🤝 Contributing

This is an academic project repository. For issues or suggestions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 👤 Author

**Uttam Mahata**

---

## 📄 License

This project is created for educational purposes as part of a Machine Learning laboratory course.

---

## 🙏 Acknowledgments

- UCI Machine Learning Repository for datasets
- Kaggle for dataset hosting and community
- Course instructors and teaching assistants
- Open-source ML/DL community (scikit-learn, TensorFlow, Keras)

---

## 📚 References

### Academic Papers
- Blackard, J.A. and Dean, D.J. (1999). "Comparative Accuracies of Artificial Neural Networks and Discriminant Analysis in Predicting Forest Cover Types from Cartographic Variables"
- W.N. Street, W.H. Wolberg and O.L. Mangasarian. "Nuclear feature extraction for breast tumor diagnosis" (1993)

### Datasets
- [UCI ML Repository](https://archive.ics.uci.edu/ml/index.php)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [MNIST Database](http://yann.lecun.com/exdb/mnist/)

### Documentation
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [TensorFlow/Keras Documentation](https://www.tensorflow.org/)
- [NumPy Documentation](https://numpy.org/)
- [Pandas Documentation](https://pandas.pydata.org/)

---

## 📞 Contact

For questions or collaborations related to this repository, please open an issue on GitHub.

---

**Last Updated**: October 2025

**Repository**: [https://github.com/Uttam-Mahata/Machine-Learning-LAB](https://github.com/Uttam-Mahata/Machine-Learning-LAB)
