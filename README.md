# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The term neural network refers to a group of interconnected units called neurons that send signals to each other. While individual neurons are simple, many of them together in a network can perform complex tasks. Neural networks are flexible and can be used for both classification and regression. In this article, we will see how neural networks can be applied to regression problems.

Regression is a method for understanding the relationship between independent variables or features and a dependent variable or outcome. Outcomes can then be predicted once the relationship between independent and dependent variables has been estimated.

First import the libraries which we will going to use and Import the dataset and check the types of the columns and Now build your training and test set from the dataset Here we are making the neural network 2 hidden layer with activation layer as relu and with their nodes in them. Now we will fit our dataset and then predict the value.

## Neural Network Model

![image](https://github.com/user-attachments/assets/37315130-72a4-403d-af35-dbee816e8d60)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: ISREAL MOSES B
### Register Number: 212221040060
```python


from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp 1').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 6, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()


loss_df = pd.DataFrame(AI_Brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

AI_Brain.evaluate(X_test1,y_test)

X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)


```
## Dataset Information

![image](https://github.com/user-attachments/assets/b3b14c12-9293-4f98-b395-d783d683b50a)


## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/ddfca0c8-f4c5-4d36-963e-0de574608726)


### Test Data Root Mean Squared Error

![image](https://github.com/user-attachments/assets/13bebeba-df22-413e-9f87-70de75ce58f6)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/fb590836-c410-45dc-adef-0bcf2d51401c)


## RESULT

To develop a neural network regression model for the given dataset is created sucessfully.

