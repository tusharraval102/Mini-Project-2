from tensorflow import keras
from keras.models import Sequential
from keras import layers
from keras.layers import Dense
from keras.layers import Input

import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import plotly
import plotly.express as px
import plotly.graph_objects as go

print("\n")
print("Tensorflow version: ", keras.__version__)
print("Pandas version: ", pd.__version__)
print("Numpy version: ", np.__version__)
print("Scikit-learn version: ", sklearn.__version__)
print("Plotly version: ", plotly.__version__)
print("\n")

# Load the dataset
df = pd.read_csv("./Dataset.csv")
ts = pd.read_csv("./Testset.csv")

# Print the dataset
print("-------------------- Dataset --------------------")
print(df)
print("-------------------- Testset --------------------")
print(ts)

# STEP 1: Selecting data for modeling
X_train = df[['A', 'B', 'C', 'D', 'E']]
y_train = df['Output'].values

# STEP 2: Create training and test sets
X_test = ts[['A', 'B', 'C', 'D', 'E']]
y_test = ts['Output'].values


# STEP 3: Specify the structure of a Neural Network
model = Sequential(name="Neural-Network")
model.add(Input(shape=(5,), name="Input-Layer"))
model.add(Dense(16, activation='relu', name="Hidden-Layer"))
model.add(Dense(16, activation='relu', name="Hidden-Layer2"))
model.add(Dense(1, activation='sigmoid', name="Output-Layer"))

# STEP 4: Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['Accuracy'],
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
)

# STEP 5: Fit the keras model on the dataset
model.fit(X_train,
          y_train,
          batch_size=10,
          epochs=3,
          verbose='auto',
          callbacks=None,
          validation_split=0.2,
          validation_data=None,
          shuffle=True,
          class_weight={0: 0.3, 1: 0.7},
          steps_per_epoch=None,
          validation_freq=3)

# STEP 6: Use the model to make predictions
pred_lables_train = (model.predict(X_train) > 0.5).astype(int)
pred_lables_test = (model.predict(X_test) > 0.5).astype(int)

# Step 7 - Model Performance Summary
print("")
print('-------------------- Model Summary --------------------')
model.summary()  # print model summary
print("")
print('-------------------- Weights and Biases --------------------')
for layer in model.layers:
    print("Layer: ", layer.name)  # print layer name
    print("  --Kernels (Weights): ", layer.get_weights()[0])  # weights
    print("  --Biases: ", layer.get_weights()[1])  # biases

print("")
print('---------- Evaluation on Training Data ----------')
print(classification_report(y_train, pred_lables_train))
print("")

print('---------- Evaluation on Test Data ----------')
print(classification_report(y_test, pred_lables_test))
print("")
