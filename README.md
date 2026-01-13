# Machine Learning Portfolio
**CS 4641 – Georgia Tech** | Python, NumPy, scikit-learn, PyTorch

A comprehensive collection of machine learning implementations spanning classical algorithms to modern deep learning architectures.

---

## 📊 Supervised Learning

### **Regression & Optimization**
- **Linear & Ridge Regression**: Closed-form solutions, gradient descent (GD/SGD/MBGD), L2 regularization
- **Cross-validation**: Hyperparameter tuning, k-fold validation
- **Polynomial feature engineering**: Basis expansion for non-linear modeling

### **Classification**
- **Logistic Regression**: Binary classification, text sentiment analysis, gradient-based optimization
- **Naive Bayes**: Probabilistic classification with Gaussian distributions
- **Support Vector Machines (SVM)**: Custom kernels, RBF kernels, kernel trick implementation
- **Neural Networks**: Two-layer fully-connected networks with softmax/softsign activations, dropout regularization, backpropagation, Adam optimizer

### **Model Evaluation & Selection**
- **Metrics**: ROC curves, AUC, confusion matrices, accuracy, RMSE
- **Feature selection**: Forward selection, backward elimination
- **Class imbalance**: SMOTE oversampling technique

---

## 🔍 Unsupervised Learning

### **Clustering**
- **K-Means/K-Means++**: Vectorized implementation, convergence criteria, clustering metrics (Silhouette coefficient, Adjusted Rand Index)
- **Hierarchical clustering**: Agglomerative bottom-up approach, dendrogram visualization
- **Gaussian Mixture Models (GMM)**: EM algorithm, soft clustering, image compression via pixel clustering

### **Dimensionality Reduction**
- **SVD**: Audio compression, collaborative filtering recommender systems
- **PCA**: Eigenface decomposition for facial recognition, variance retention analysis

---

## 🤖 Deep Learning & Computer Vision

### **Convolutional Neural Networks (CNN)**
- **Image classification**: Multi-class tumor detection
- **Data augmentation**: Random rotations, flips, normalization
- **Regularization**: Dropout, batch normalization concepts
- **Framework**: PyTorch implementation with custom architectures

### **Recurrent Neural Networks**
- **RNN/LSTM**: Next-character prediction, sequential text generation
- **Vanishing gradient mitigation**: LSTM memory cells

---

## 📐 Mathematical Foundations

### **Linear Algebra**
- Matrix operations, eigenvalue decomposition, SVD factorization
- Computational efficiency via broadcasting (no explicit loops)

### **Probability & Statistics**
- Maximum Likelihood Estimation (MLE), Bayesian inference
- Expectation, covariance, correlation, statistical independence
- Information theory: Entropy, mutual information, KL divergence

### **Optimization Theory**
- KKT conditions for constrained optimization
- Gradient-based methods, convergence analysis
- Numerical stability (logsumexp tricks, softmax normalization)

---

## 🛠️ Technical Skills Demonstrated
**Languages**: Python  
**Libraries**: NumPy, pandas, scikit-learn, PyTorch, matplotlib, scipy  
**Techniques**: Vectorized computation, object-oriented design, algorithm implementation from scratch  
**Applications**: Image/audio processing, NLP sentiment analysis, recommender systems, medical imaging

---

## 🌟 Notable Projects
- **Semi-supervised learning**: Messy data imputation with k-NN and EM for gamma ray telescope classification
- **Image compression**: GMM-based lossy compression in RGB color space
- **Eigenfaces**: PCA-driven facial recognition system
- **Text generation**: Character-level RNN/LSTM for sequence modeling

---

## 📂 Repository Structure
- `cs4641-hw1/`: Mathematical foundations (linear algebra, probability, optimization, information theory)
- `cs4641-hw2/`: Clustering algorithms (K-Means, hierarchical, GMM), EM algorithm
- `cs4641-hw3/`: SVD applications, PCA, regression models, classification, feature selection
- `cs4641-hw4/`: Neural networks, CNNs, SVMs, RNN/LSTM
