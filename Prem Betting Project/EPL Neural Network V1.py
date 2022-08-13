from base64 import b16decode
from tokenize import Single
#from re import X
#from tkinter import Y
import numpy as np
import pandas as pd
#from Football_data_scraping_Single_Year import df
from Youtube_Scraping import match_df



#Clean Data




#machine learning can only work with float64 or int64, does not work with objects
#matches.dtypes
matches = match_df

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

data = np.array(data)

#print(data.shape)


m, n = data.shape
#shuffling data
np.random.shuffle(data)

data_dev = data[0:int(m*0.8)].T
Y_dev = data_dev[0]
X_dev = data_dev[2:n]

data_train = data[int(m*0.8):m].T
Y_train  = data_train[2]
X_train = data_train[2:n]


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
    W1 = np.random.rand(10, n-2) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    Z = Z[:,~pd.isna(Z).any(axis=0)]
    Z = np.array(Z, dtype=int)
    A = np.exp(Z) / sum(np.exp(Z))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    #Z1 = np.dot(W1,X) + b1
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    print(A1)
    Z2 = W2.dot(A1) + b2
    #Z2 = np.dot(W2,A1) + b2
    A2 = softmax(Z2)
    print(A2)
    return Z1, A1, Z2, A2

#acts as a boolean so prints 0 or 1
def ReLU_deriv(Z):
    return Z > 0


#creates a new matrix that is correctly sized, moving backwards, Y is the column of outputs, in our case win, lose or draw
def one_hot(Y):
    Y = np.array(Y, dtype=int)
    #one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y = np.zeros((Y.size,10))
    #for each row go to the col specified by the label in Y and set it to 1
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y




def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    m = Y.size
    one_hot_Y = one_hot(Y)
    #print(one_hot_Y.shape)
    #print(one_hot_Y)
    #print(A2.shape)
    #print(A2)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    #dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    #dZ1 = np.dot(W2.T,dZ2)*ReLU_deriv(Z1)
    #dW1 = 1 / m * dZ1.dot(X.T)
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2



def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2



#Now do gradient decsent



def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        #every 10th iteration we will print 
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print(get_accuracy(predictions, Y))
    return W1, b1, W2, b2


W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 0.20, 100)

def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    print(current_image)


#cross-val accuracy
dev_predictions = make_predictions(X_dev, W1, b1, W2, b2)
print(get_accuracy(dev_predictions, Y_dev))



"""

#my attempt 

#W1 = np.random.rand((380,380)) * 0.01
W1 = np.random.random((380,380))
b1 = np.zeros((380,1))
W2 = np.random.random((1,380)) * 0.01
b2 = 0

#learning rate (hyperparam)
alpha = 0.01

print(W1)
#Z1 = W1.dot(df)+b1

#print(Z1)
"""""
"""
#ReLU
def g(z):
    return max(0,z)

#ReLU derivative
def g1_dash(z):
    if z<0:
        return(0)
    else:
        return(1)

#Creating X, A1, Z1 etc from x, a1, z1
for i in range(m):
    z1(i) = W1*x(i)+b1
    a1(i) = g(z1(i))
    z2(i) = W2*a1(i)+b2
    a2(i) = g(z2(i))


#Big X is the input (games), all vectorised
#A0 = X
Z1 = W1*X + b1
A1 = g(Z1)
# Z1 = [z11, z12, z13, ..., z1m]
Z2 = W2*A1+b2
A2 = g(Z2)

#derivatives for gradient descent
dZ2 = A2-Y
dW2 = (1/m)*dZ2*((A1).T)
db2 = (1/m)*np.sum(dZ2,axis=1,keepdims=True)
dZ1 = (W2).T*dZ2*g1_dash(Z1)
dW1 = (1/m)*dZ1*(X.T)
db1 = (1/m)*np.sum(dZ1,axis=1, keepdims=True)



#Updating params

W1 = W1 - alpha*(dW1)
b1 = b1 - alpha*(db1)
W2 = W2 - alpha*(dW2)
b2 = b2 - alpha*(db2)

"""