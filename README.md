# Multilayer Perceptron - Diabetes Predictor

I created a multilayer perceptron (Deep Neural Network) to determine whether or not an individual has diabetes based on 8 characteristics relating to their health information.

The goal of this project was to examine how the variation in Neural Network architecture can yeild higher accuracy. Utilizing relevant statistical concepts was imperative in producing an accurate classifier while considering how to best suit my model to make predictions for the dataset.

---

## Table of Contents
1. [Installation & Usage](#installation-and-usage)  
2. [Data Visualization](#data-visualization)
3. [Network Architecture](#network-architecture)
4. [Features](#features)  
5. [Contact Info](#contact-info)  
6. [Versions Quick Reference](#versions-quick-reference)


---
## Installation and Usage
1. The `DiabetesPredictor.py` script runs various user defined hyperparameters to produce optimal Deep Neural networks to make an accurate diabetes classifier.
   
   *Running the script will:*
   
   **a.** Automatically check for libraries required to run this project.
   
   **b.** Ask if you want to install them; entering `Y` in the terminal will install them.
   
   **c.** Ask you what loss function you want to use. Automatically select appropriate activation function for the optimizer

   **d.** Train a model with every combination of the hyperparameters defined in the `hyperparam_grid` using the validation dataset.

   **e.** Reconstruct the network that had the highest accuracy and test it using the testing dataset

   **f.** Display the loss and accuracy of the network.
   
   **g.** Display a confusion matrix and ROC curve in a window pop-up.
   
   *Closing the confusion matrix window will display the ROC curve of the network.*
  
2. Utilizing `tensorflow-gpu` for this project is optional, but because the training dataset is relatively small and the architecture is simple, it will train quickly on the cpu. It may be useful when adding several hyperparameters as it tests every combination, although on a smaller validation set.  

3. It is recommended that you create a Python virtual environment so that any pre-existing versions of these libraries are not updated.  
   
---
## DATA VISUALIZATION 

### CONFUSION MATRIX

My script produces a Confusion Matrix to display Actual/Predicted against True/False classifications.  
At a glance, you can see the rate of correct classifications or Type 1 or Type 2 errors.

### ROC CURVE GRAPH
The ROC (Receiver Operating Characteristic) curve measures the True Positive Rate against the False Positive Rate.  
My script creates an ROC curve to measure the model's performance.  
This is important to evaluate the validity of high-probability predictions for the model.

#### Relevant Equations  ;  T=True F=False P=Positive N=Negative
- **TRUE POSITIVE RATE (TPR = TP / (TP + FN))**  
- **FALSE POSITIVE RATE (FPR = FP / (FP + TN))**  
- **MODEL ACCURACY (Accuracy = (TP + TN) / SAMPLES)**

### AREA UNDER ROC CURVE (TPR / FPR)
- Provides insight into how well the model distinguishes between classes.  
- Every single model produced by the varied architectures had a ROC > 0.5, meaning that it is better than random.  
- The model also reaches a TPR of 1 quicker than the random prediction line, so it is effective at making distinctions between features in the dataset.

### Preprocess Data to Maximize Model Effectiveness
1. Replace NaN values with 0.  
2. Normalize the features' numerical values:  
   - Scaling the data so that the entire dataset has a standard deviation of 1 and mean of 0.
   - This allows for faster convergence and equal consideration of different features during training.

---

## Network Architecture

The design of the network architecture is key to producing an accurate classifier.
#### GOAL:
#### Accuratly classifly whether or not someone has diabetes.

#### Design choices:

- **Input Features** : A total of 8 characteristics were provided in the dataset.
- **Size of Input Layer** : There must be an input layer of 8 neurons to recieve accurate data.

  *It is possible to reduce the number of nodes in the input layer using PCA, but this is usaully reserved for datasets with many more characteristics. PCA also results in some loss in data, but preserves the components with the greatest variance.*

- **Number of Hidden layers**

  The number of hidden layers is determined by the scale of the training dataset and the complexity of the classification problem. Only 491 rows of data were used for the explicit purpose of training and the classification was binary, so 1-2 layers were sufficient for this task.
- **Loss Function**

  I used the **Hinge Loss Function** and **Binary Crossentropy function** because they are both well suited for binary classifiers (Diebetes or No Diabetes). **You will be asked to select one or the other.**
- **Output Neuron Activation Function**
  
  The choice of activation function needs to consider the loss function's parameters.

  *When the loss function is:*
  - Binary Crossentropy, the network will utilize the **Sigmoid Activation Function**.

    Binary cossentropy (BCE) requires a single predicted probability from 0 to 1, so sigmoid is appropriate for this as all outputs will be within this range (sigmoid(wx+b) *denoted as y-hat*). The activation of the output neuron, processed by Sigmoid is utilized in BCE. Assuming one pass, the BCE loss is computed as −(y⋅log(p​)+(1−y)⋅log(1−p​)). Where the true label (y) is 0 or 1 (predefined when enumerating the training set). If the true label is 1 and the predicted probability (p sigmoid output) is high, then the right side cancels out and you are left with -(1-log(.99)) (-ln(.99) is .01), meaning low loss. But, smaller predicted values for the true class are exponentially more penalized, since -ln(0) approaches infinity.
    
  - Hinge Loss, the network will use a **linear activation function**.
    
    This is because the scale of the misclassification (how wrong the prediection was) matters when constructing the gradient vector and using it in updating the weights and biases. A linear activation function reflects the magnitude of the incorrect prediction better than sigmoid because the output is preserved as a raw score (not a probability between 0 and 1).

    **The selection of activation happens automatically based on choice of loss function**
- **Optimizer** (Weight Update Function)

  I chose **Stochastic Gradient Descent (SGD)** for because It was the introduced to me initially when I was studying neural network architecture. I also use **Adaptive Movement Estimation (ADAM)** because it utilizes adaptive learning rates through the utilization of *momentum* and *velocity* variables. *Both of these are hyperparameters that you can vary. You can add other optimizers if you update the code in `hyperparam_grid` and custom logic for hard-coded parameters for the optimizers.

  
#### Creation of training, testing, validation sets
1. Split the original dataset into **80% Training Set** and **20% Testing Set**
2. Take **20% of the Training Set** to use as the **Validation set (16% of original network)**

##### My reasoning
    **Training Set (64%)** : Used in training the network with the highest accuracy on validation set.
    
    **Validation Set (16%)** : Used to train hyper parameters and training is sped up with smaller sets.

    **Testing Set (20%)** : Used in the final evaluation of the model. The model has never seen this data prior.

##### Training
1. **Glorot Uniform Initalization** of weights.
  
   *This is just the default for keras fully connected networks.*
   
   *Other methods exist such as:*
   *HeNormal/HeUniform (forReLu activation) ; RandomNormal/Uniform (custom params)*
   
  
2. **Create input layer**

   The Input dimensions of the data are the features of the dataset.

   *This dataset has an individual's characteristics such as blood pressure, glucose, etc.*
   
3. **Define Hidden Layers**

   The hidden layers and number of neurons for every combination is defined in the `hyperparam_grid`.

   I was taught to follow three rules regarding hidden layer structure:
   
   *Total number of hidden neurons should be:*
   1. (input neurons + output neurons) * 2/3
   2. Between the input and output layer size
   3. never larger than twice the size of the input layer.

   These are only general rules and it really depends on the dataset and type of classification problem.
4. **Define Output Layer**

   The desired output of the classification is **Diabetes** or **No Diabetes**
   
   *The dot product of the final layer of neurons with the weights is passed through the sigmoid function to determine the activation of the **output neuron**.*

5. **Training**

   The overarching goal of this process is to minimize the total loss of the network and update the weights such that new data will fall into the accurate classification threshold.
   
   **Feed-Foward**

   The loss of the network after a batch (or single) number of training examples is computed at the output layer.
   
   *As computed through the **binary cross-entropy** or **Hinge** functions within this script.*

   **Back-Propagation**

   The next step is to compute the gradient vector, which is the contribution of each individual weight and bias to the loss of the overall network. 
   ##### This is done by utilizing the chain rule of calculus to solve for the change in cost relative to the preactivation functions of deeper layers.

   *Solve for:*
   1. The derivative of the loss (L) with respect to the activation neuron (a^L)
   2. The derivative of the activation neuron (a^L) with respect to the Activation_Function(pre-activation (z^L))
   3. The derivative of the pre-activation (z^L) with respect to the weight (w^L)
   4. The derivative of the pre-activation (z^L) with respect to the bias (b^L)
      
      *By this step you utilize the chain rule of calculus to compute the gradient of the loss with respect to weight at layer L (w^L) and gradient of the loss with respect to the bias at layer L (b^L)*

   **Repeating this process through previous layers results in the construction of the *gradient vector*, which is used in updating the weights with the goal of minimizing the cost of the network.**

6. **Update The Weights and Biases**

   ***Adaptive Movement Estimation (ADAM)** and **Stochastic Gradient Descent (SGD)** are used in these scripts for Back-Propagation as they are further optimized *

   **[Stochastic Gradient Descent](https://keras.io/api/optimizers/sgd/):** Computed as `w = w + momentum * velocity - learning_rate * g` 

   *Where*:

   - `velocity = momentum * velocity - learning_rate * g`
   - g is the gradient for that weight (use corresponding gradient value in gradient vector)
   - *Learning Rate* is equal to 0.01 by default
   - *Momentum* is equal to 0 by default

   **[Adaptive Movement Estimation](https://keras.io/api/optimizers/adam/):** Is too lengthy for the scope of this document, but linked is the keras documentation for this optimizer.

7. **Number of Epochs**
   
   The number of times steps 5 - 6 repeats over entire the dataset is one epoch. There are optimizations here with mini-batching because it would take an insane amount of compute to do this with every individual training example.

   **The number of epochs is defined by the user, but you will notice that the loss will converge to a constant-ish value. This is "convergence" and a good indicator that there are diminishing returns for additional epochs.** Setting `verbose=1` will display the loss after every epoch.

10. Final Prediction

    The weights and biases of the neural network are set and no longer updated.
    
    **The classification threshold varies based on the activation function of the output neuron:**
    
    **Sigmoid**: 0 (no diabetes) if output neuron activation is <.5 and 1 (diabetes) if >=.5*

    **Linear**: Creates a boundary in higher dimensional space where the dataset is linearly seperable.

---

## Contact Info
**Email:** [juandev32@gmail.com](mailto:juandev32@gmail.com)  
**LinkedIn:** [Juan Chavira's Profile](https://www.linkedin.com/in/juan-chavira/)

---

## VERSIONS QUICK REFERENCE
- `python==3.11.*`  
- `keras==3.7.0`  
- `matplotlib==3.8.1`  
- `numpy==2.2.1`  
- `pandas==2.2.3`  
- `scikit_learn==1.6.0`  
- `seaborn==0.13.2`  
- `tensorflow==2.18.0`  