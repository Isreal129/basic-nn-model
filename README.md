# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error.


## Neural Network Model

![359044031-151f56b9-8129-4253-a9c3-744ab9c77732](https://github.com/user-attachments/assets/426329b6-bdf8-4489-a9fb-de764308bd23)


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
```
 Name: ISREAL MOSES
 Register Number: 212221040060
```
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import auth
import gspread
from google.auth import default

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd  

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('Deeplearning').sheet1
data = worksheet.get_all_values()
dataset1 = pd.DataFrame(data[1:], columns=data[0])
dataset1 = dataset1.astype({'Input': 'int', 'Output': 'int'})
dataset1.head()

x = dataset1[['Input']].values
y = dataset1[['Output']].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=33)

Scaler = MinMaxScaler()
Scaler.fit(x_train)
x_train1 = Scaler.transform(x_train)

ai_brain = Sequential([
    Dense(8,activation = 'relu'),
    Dense(10,activation = 'relu'),
    Dense(1)
])

ai_brain.compile(optimizer = 'rmsprop', loss = 'mse')
ai_brain.fit(x_train1,y_train,epochs = 1000)

loss_df = pd.DataFrame(ai_brain.history.history)
loss_df.plot()
x_test1 = Scaler.transform(x_test)
ai_brain.evaluate(x_test1,y_test)
x_n1=[[4]]
x_n1_1 = Scaler.transform(x_n1)
ai_brain.predict(x_n1_1)

```
## OUTPUT:

## Dataset Information

![image](https://github.com/user-attachments/assets/22ed739c-818d-41c7-a317-cce21330db53)


### Training Loss Vs Iteration Plot
![image](https://github.com/user-attachments/assets/3c7a53b8-cdf3-4e18-82ca-7cb2efdc2033)


### Test Data Root Mean Squared Error
![image](https://github.com/user-attachments/assets/a84fb875-d12b-435c-aec1-f1486cb8585c)

![image](https://github.com/user-attachments/assets/73ba7850-c0b4-48cb-8349-2754e7dfddf3)



### New Sample Data Prediction
![image](https://github.com/user-attachments/assets/e4aec84c-5232-47c9-aec2-b8defacb7761)


## RESULT:

Thus a Neural Network regression model for the given dataset is written and executed successfully


