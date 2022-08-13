from base64 import b16decode
from tokenize import Single
#from re import X
#from tkinter import Y
import numpy as np
import pandas as pd
#from Football_data_scraping_Single_Year import df
#from Youtube_Scraping import match_df

matches = pd.read_csv("matches.csv", index_col=0)



#Clean Data




#machine learning can only work with float64 or int64, does not work with objects
#matches.dtypes
#matches = match_df

del matches["comp"]
del matches["notes"]
del matches["date"]
del matches["attendance"]
del matches["dist"]
del matches["round"]
del matches["day"]
del matches["captain"]
del matches["formation"]
del matches["referee"]
del matches["match report"]

#overriding existing col with datetime data

#matches["date"] = pd.to_datetime(matches["date"])
matches["target"] = (matches["result"] == "W").astype("int")
#print(matches)



#Creating predictors

#away or home game, converting to a number so can be used in algo, .cat.codes converts to integer

matches["venue_code"] = matches["venue"].astype("category").cat.codes

#Unique code for each opponent
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["team"] = matches["team"].astype("category").cat.codes

#we want to just have the hour rather than hour:min, so is an int, need a number input
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")

matches["gf"] = matches["gf"].astype("int")
matches["ga"] = matches["ga"].astype("int")
#changing each day of the week to a number
#matches["day_code"] = matches["date"].dt.dayofweek
#matches

#We want the target to be to predict this, change this to a number as well, 0 if lost or drew and 1 otherwise


del matches["result"]
del matches["venue"]
del matches["opponent"]
del matches["time"]




#turn all the floats into int

#matches["gf"] = matches["gf"].astype("int")
#matches["ga"] = matches["ga"].astype("int")
#matches["xg"] = matches["xg"].astype("int")
#matches["xga"] = matches["xga"].astype("int")
#matches["poss"] = matches["gf"].astype("int")
#matches["sh"] = matches["sh"].astype("int")
#matches["sot"] = matches["sot"].astype("int")
#matches["dist"] = matches["dist"].astype("int")
#matches["fk"] = matches["fk"].astype("int")
#matches["pk"] = matches["pk"].astype("int")
#matches["pkatt"] = matches["pkatt"].astype("int")


#Start Neural Net



#data = df
data = matches
#print("dtypes")
#print(data.dtypes)
#print(data)
data = np.array(data)
#print(data)
#print(data.shape)


m, n = data.shape 
#shuffling data
np.random.shuffle(data)

data_dev = data[0:int(m*0.2)].T
#13 is where the 'target' is
#data_dev[[13, 0]] = data_dev[[0, 13]]
Y_dev = data_dev[0]
X_dev = data_dev[0:n]

print(Y_dev.shape)
data_train = data[int(m*0.2):m].T
#data_train[[13, 0]] = data_train[[0, 13]]
Y_train  = data_train[0]
X_train = data_train[0:n]


#print(X_train)
#print(X_train.shape)
#print(Y_train)


#print(Y_train)

#print(X_train)
#print(X_train[:, 0].shape)
#print(X_train[:, 0].shape)



#initially creating random weights and bias
#how many nodes do we want, done 304-> 10 -> 10 -> 1

def init_params():
    W1 = 10*np.random.rand(10, n) - 0.5
    b1 = 10*np.random.rand(10, 1) - 0.5
    W2 = 10*np.random.rand(10, 10) - 0.5
    b2 = 10*np.random.rand(10, 1) - 0.5
    W3 = np.random.rand(2, 10) - 0.5
    b3 = np.random.rand(2, 1) - 0.5
    return W1, b1, W2, b2, W3,b3

#changing the - to a +
def elu(Z):
    Z = np.where(Z<0,0.05*np.exp(Z)-1 ,Z)
    return Z 

#def ReLU(Z):
    return np.maximum(Z, 0)
#def elu(Z):
    return Z if Z >= 0 else elu_func2(Z)
#need to fix this, not clear if true or false
"""
def softmax(Z):
    Z = Z[~np.isnan(Z).any(axis=1), :]
    #Z = Z[:,~pd.isna(Z).any(axis=0)]
    Z = np.array(Z, dtype=int)
    result = np.any(Z == 0)
    if result:
        A = Z
    A = np.exp(Z) / sum(np.exp(Z))
    return A
"""
def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = elu(Z1)
    #print("A1")
    #print(A1)
    #print(A1)
    Z2 = W2.dot(A1) + b2
    #print("Z2")
    #print(Z2)
    #A2 = softmax(Z2)
    #A2 = ReLU(Z2)
    A2 = elu(Z2)
    Z3 = W3.dot(A2) +b3
    A3 = elu(Z3)
    #print(A2)
    return Z1, A1, Z2, A2, Z3, A3

#acts as a boolean so prints 0 or 1
#def ReLU_deriv(Z):
    return Z > 0

#def elu_deriv(Z):
    return 1 if Z > 0 else 0.01

def elu_deriv(Z):
    Z = np.where(Z>0,1,0.05)
    return Z 

#creates a new matrix that is correctly sized, moving backwards, Y is the column of outputs, in our case win, lose or draw
"""
def one_hot(Y):
    Y = np.array(Y, dtype=int)
    #one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y = np.zeros((Y.size,Y.max() + 6))
    #for each row go to the col specified by the label in Y and set it to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
"""



def backward_prop(Z1, A1, Z2, A2,Z3, A3, W1, W2, W3, X, Y):
    m = Y.size
    #MSE = np.square(np.subtract(Y,A2))
    #print(MSE)
    #one_hot_Y = one_hot(Y)
    #print(one_hot_Y.shape)
    #print(one_hot_Y)
    #print(A2.shape)
    #print("A2")
    #print(A2)
    #dZ2 = A2 - one_hot_Y
    #dZ2 = A2 - Y
    dZ3 = A3-Y
    print("Y shape")
    print(Y.shape)
    #print("dZ2")
    #print(dZ2)
    dW3 = 1 / m * dZ3.dot(A2.T)
    db3 = 1 / m * np.sum(dZ3,axis=1, keepdims=True)
    dZ2 = W3.T.dot(dZ3) * elu_deriv(Z2)
    dW2 = 1 / m * dZ2.dot(A1.T)
    #dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2,axis = 1, keepdims=True)
    #dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dZ1 = W2.T.dot(dZ2) * elu_deriv(Z1)
    #dZ1 = np.dot(W2.T,dZ2)*ReLU_deriv(Z1)
    dW1 = (1 / m) * dZ1.dot(A1.T)
    #dW1 = 1 / m * dZ1.dot(X.T)
    #dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1,axis=1,keepdims=True)
    return dW1, db1, dW2, db2, dW3, db3


#change - to +  
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 = W1 - (alpha * dW1)
    b1 = b1 - (alpha * db1)    
    W2 = W2 - (alpha * dW2) 
    b2 = b2 - (alpha * db2)
    W3 = W3 - (alpha * dW3)
    b3 = b3 - (alpha * db3) 
    return W1, b1, W2, b2, W3, b3



#Now do gradient decsent



def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        #every 10th iteration we will print 
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Accuracy" , get_accuracy(predictions, Y))
    return W1, b1, W2, b2, W3, b3




W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, Y_train, 10, 100)
def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _,_,_,_,_,A2 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index]
    #current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index], W1, b1, W2, b2, W3, b3)
    #prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    print(current_image)


#cross-val accuracy
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2, W3, b3)
print(get_accuracy(dev_predictions, Y_dev))

#combined = dev_predictions.merge(dev_predictions[["date", "team", "opponent", "result"]], left_index=True, right_index=True)

print(test_prediction(110,W1,b1,W2,b2,W3,b3))