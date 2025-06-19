import tensorflow as tf

import pandas as pd                     #data manipulation used to create data frame
from matplotlib import pyplot as plt    #create visuals with dataframes
import seaborn as sns                   #generate density plots
import numpy as np                      #mathematical operations; used NaN 
from sklearn import preprocessing       #for scale() to preform centering and covariance equalization
from sklearn.model_selection import train_test_split    #randomly splits data set
from keras.models import Sequential
from keras.layers import Dense,Input

from sklearn.metrics import confusion_matrix
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from sklearn.model_selection import ParameterGrid

#need theese to verify you have all the depencies in your env (should use python virtual env)
import subprocess
import sys
import importlib

"""
Configuration
Code to ensure user has nessesary libraries as in requirements.txt file.
Setting up tensorflow to work with Cuda is optional, 
the logic will tell you if it properly recognizes your gpu.
"""
#map package in requirements.txt to their import name if deviation
package_name_map= {
    "scikit_learn":"sklearn"
}

#ensure youre running cuda 11.8 if you want to enable gpu for faster model training
try:
    print(tf.sysconfig.get_build_info()["cuda_version"])
except KeyError as k:
    print(f"Cuda version is not recognized by tensorflow, probably cuda version >11.8")
except Exception as e:
    print(f"Error has occured while trying to get the CUDA version {str(e)}")

#ensure all the dependencies in the requiremnets.txt file are installed
try:
    with open("./requirements.txt",'r') as reqfile:
        #read lines from requirement 
        for line in reqfile:
            
            package = line.split('==')[0].strip()
            #map the names of any required files to their names in pip (if they exist if not then it retruns default value through 2nd arg)
            import_name_in_pip=package_name_map.get(package,package)
            print(package)
            try:
                importlib.import_module(import_name_in_pip)
            except ImportError:
                print(package)
                promptInstall= input(f"{package} is not installed. Do you want to install it Y/N : ").strip().lower()
                if promptInstall in ['y',"yes","ok","ye"]:
                    print(f"Installing package {package} ...")
                    subprocess.check_call([sys.executable,'-m','pip','install',line.strip()])
                    print(f"{package} has been installed")
                    break
                elif promptInstall in ['n',"no","nope","on"]:
                    print("You probably need this dependency to train the neural network, later exceptions may occur")
                    break
                else:
                    print("Enter a valid input 'Y' or 'N'.")
except FileNotFoundError:
    print("requirements.txt file not found. \nDo not move the file from the Multilayer Perceptron - Diabetes predictor Parent directory")
except Exception as e:
    print(f'An exception as occured: {str(e)}')
finally:
    print("All depencencies are installed with the correct version.")
"""
Data Pre-Processing
1. Output descriptive statistics of dataset
2. Fill in NaN values with 0
3. Normalize values (stdev=1 mean=0)
4. Display data again to confirm this
5. Create Training, Validation, Testing Dataset
"""
#create dataframe object of csv
df=pd.read_csv("./diabetes.csv")

#check data for null values and print result
print(df.isnull().any())

#Generate descriptive statistics for dataset and print result
pd.set_option("display.max_columns", 500)
print(df.describe(include='all'))   

#count occurences of 0 in columns
print("Number of rows with 0 values for each variable:")
for col in df.columns:
    missing_rows = df.loc[df[col] == 0].shape[0]
    print(col + ": " + str(missing_rows))

df['Glucose'] = df['Glucose'].replace(0, np.nan)
df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
df['Insulin'] = df['Insulin'].replace(0, np.nan)
df['BMI'] = df['BMI'].replace(0, np.nan)

#Replace NaN values with the mean of the column. This is why the functions are called in this manner. To create seperate means
df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

#Data Normalization
print("Centering the data...")
df_scaled = preprocessing.scale(df)
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
df_scaled['Outcome'] = df['Outcome']
df = df_scaled
print(df.describe().loc[['mean', 'std','max'],].round(2).abs())

#Generating Training, Validation, and Testing Set
X = df.loc[:, df.columns != 'Outcome']
y = df.loc[:, 'Outcome']

# split input matrix to create the training set (80%) and testing set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# second split on training set to create the validation set (validation set is 20% of training set; so 16% of orignal dataset)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

"""
Constructing Neural Network
1. Choose how to vary the hyper parameters
2. Allow users to decide between Binary CrossEntropy or Hinge loss functions, vary the activation functions based on choice.
3. Dynamically construct the model based on what is defiend in hyperparam_grid
4. Run the the varying combinations of model hyperparameters on the validation testing set.
5. Save the highest accuracy hyperparameters and re-contruct the model architecture.
6. Display ROC curve and Confusion Matrix for the Highest Accuracy Model

"""

#create the hyperparameters to tune within the model
hyperparam_grid = {
    'learning_rate': [0.01,.1],
    'batch_size': [16,24],
    'epochs': [ 150,200],
    'num_layers': [2,3],
    'num_neurons': [4,6,8],
    'optimizer': ["Adam","SGD"]
}

#create selection process for user to pick loss function, this is used later to determine the activation function of output neuron
loss_functions=[
    ["binary_crossentropy",tf.keras.losses.BinaryCrossentropy()],
    ["hinge",tf.keras.losses.Hinge()]
    ]

#Create dynamic menue to scale with updates to loss functions if I want to add more in future
print("\n")
for idx, loss_entry in enumerate(loss_functions):
   
    print(f"{idx+1}. {loss_entry[0]}")

loss_function_selected=input("Select the loss function that you want to use for the network: ")

#cast user input to integer and subtract 1, just preference, i could have started at 0
loss_function_selected=int(loss_function_selected)-1

#no longer need to reference name of loss function, just the tensorflow module
loss_function_keras=loss_functions[loss_function_selected][1]
print(f"{loss_function_keras}")

#Store the best parameters and highest accuracy, validate with ROC curve and confusion matrix output.
best_accuracy= 0
best_hyperparams = None
activation_function=None

for params in ParameterGrid(hyperparam_grid):
    #display the combination of hyperparameters
    print(f"Testing hyperparameters: {params}")
    
    #configure input layer + first hidden layer
    model=Sequential()
    model.add(Input(shape=(8,)))

    #add additional hidden layers based on num_layers, vary the number of neurons per layer
    for _ in range(params['num_layers']):
        model.add(Dense(params['num_neurons'],activation='relu'))

    # Select optimizer
    if params["optimizer"] == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    elif params["optimizer"] == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate=params["learning_rate"],nesterov=True)
    else:
        raise Exception("Issue in the optimizer selection logic or change with learning rate")

    #Select the apropriate activation function for output neuron for the loss function selected
    if(loss_functions[loss_function_selected][0].lower()=="hinge"):
        activation_function="linear"
    elif(loss_functions[loss_function_selected][0].lower()=="binary_crossentropy"):
        activation_function="sigmoid"
    else:
        raise Exception("The loss function could not utilize the proper activation function for the output neuron.")

    #output neuron, varies the activation function based on loss function
    model.add(Dense(1,activation=activation_function))

    #compiling the Network
    model.compile(optimizer=params['optimizer'], loss=loss_function_keras,metrics=['accuracy'])


    #Training the Network; verbose=0 supresses the training progress terminal output; set verbose=1 to see loss after every epoch
    history=model.fit(X_train,y_train,validation_data=(X_val, y_val),batch_size=params['batch_size'],epochs=params['epochs'],verbose=0)

    val_accuracy= history.history['val_accuracy'][-1] #most recent validation accuracy
    val_loss=history.history['loss'][-1] #most recent validation loss 
    print(f"Validation Accuracy: {val_accuracy}\tValidation Loss: {val_loss}")

    #store the hyperparameters of the model with the highest accuracy with the validation set
    if (val_accuracy > best_accuracy):
        best_accuracy = val_accuracy
        best_hyperparams=params

print(f"Best hyperparameters: {best_hyperparams}")
print(f"Best validation accuracy: {best_accuracy}")


"""
Use the hyperparameters that produced the highest accuracy on the validation set.
1. Copy the same model creation logic
2. Use the single hyperparameters that i found earlier

"""
# train best hyperparams on training set
 #configure input layer + first hidden layer
best_model=Sequential()
best_model.add(Input(shape=(8,)))

#add additional hidden layers based on num_layers, vary the number of neurons per layer aswell
for _ in range (best_hyperparams['num_layers']):
    best_model.add(Dense(best_hyperparams['num_neurons'],activation='relu'))

# Use the optimizer that worked best for the training set
if best_hyperparams["optimizer"] == "Adam":
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_hyperparams["learning_rate"])
elif best_hyperparams["optimizer"] == "SGD":
    optimizer = tf.keras.optimizers.SGD(learning_rate=best_hyperparams["learning_rate"],nesterov=True)
else:
    raise Exception("Issue in the optimizer selection logic or change with learning rate")

# The loss function is decided by the user; 
# I wanted to demonstrate that I could program user flexibility in any parameter for loss func and understand impact on other params like act_func
if(loss_functions[loss_function_selected][0].lower()=="hinge"):
    activation_function="linear"
elif(loss_functions[loss_function_selected][0].lower()=="binary_crossentropy"):
    activation_function="sigmoid"
else:
    raise Exception("The loss function could not utilize the proper activation function for the output neuron.")

#output neuron, varies the activation function based on loss function
best_model.add(Dense(1,activation=activation_function))

#compiling the Network
best_model.compile(optimizer=best_hyperparams['optimizer'], loss=loss_function_keras,metrics=['accuracy'])


#Training the Network; verbose=0 supresses the training progress terminal output
history=best_model.fit(X_train,y_train,validation_data=(X_val, y_val),batch_size=best_hyperparams['batch_size'],epochs=best_hyperparams['epochs'],verbose=1)

"""
Testing the Network
    Test network with training and testing set data (good indicator to see if overfitting took place).
    Evaluate with:
        1. Testing accuracy
        2. Confusion Matrix
        3. Using a Receiver Operating Characteristic curve (ROC curve)
"""


# Evaluate the accuracy with respect to the training set
train_loss, train_accuracy = best_model.evaluate(X_train, y_train)
print(f"Training Accuracy: {train_accuracy}\nTraining Loss: {train_loss}\n")

# Evaluate the final model on the test set
test_loss, test_accuracy = best_model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy}\nTest Loss: {test_loss}\n")

path_to_trained= f"./Trained_Models/Diabetes_Predictor_Model_Accuracy_{test_accuracy:.2f}_{loss_functions[loss_function_selected][0]}_Loss_{test_loss:.2f}.keras"
best_model.save(path_to_trained)
#Construct a confusion matrix
y_test_pred = (best_model.predict(X_test) > 0.5).astype("int32")
c_matrix = confusion_matrix(y_test, y_test_pred)
ax = sns.heatmap(c_matrix, annot=True,
xticklabels=['No Diabetes','Diabetes'],
yticklabels=['No Diabetes','Diabetes'],
cbar=False, cmap='Blues')
ax.set_xlabel("Prediction")
ax.set_ylabel("Actual")
plt.show()

#Constrct ROC Curve
y_test_pred_probs = best_model.predict(X_test)
FPR, TPR, _ = roc_curve(y_test, y_test_pred_probs)
plt.plot(FPR, TPR)
plt.plot([0,1],[0,1],'--', color='black') #diagonal line
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()