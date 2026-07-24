import numpy as np
import pandas as pd
data=pd.read_csv('train_quantized.csv')
data=np.array(data)
np.random.shuffle(data)
m,n=data.shape
data_train=data[2000:m]
data_test=data[0:2000]
m-=2000
data_train=data_train.T
data_test=data_test.T
trainY=(data_train[0])
trainX=(data_train[1:n])/255.0
testY=data_test[0]
testX=data_test[1:n]/255.0

def weights():
    W1 = np.random.randn(64, 784) * np.sqrt(2.0 / 784)
    W2 = np.random.randn(32, 64) * np.sqrt(2.0 / 64)
    W3 = np.random.randn(10, 32) * np.sqrt(2.0 / 32)
    b1=np.random.randn(64,1)
    b2=np.random.randn(32,1)
    b3=np.random.randn(10,1)
    return W1,W2,W3,b1,b2,b3

def relu(X):
    return np.maximum(0,X)

def forwardprop(trainX,W1,W2,W3,b1,b2,b3):
    Z1=W1.dot(trainX)+b1
    A1=relu(Z1)
    Z2=W2.dot(A1)+b2
    A2=relu(Z2)
    Z3=W3.dot(A2)+b3
    A3=softmax(Z3)
    return Z1,A1,Z2,A2,Z3,A3

def softmax(x):
    shiftx = x - np.max(x, axis=0, keepdims=True)
    exps = (np.exp(shiftx))
    return exps / np.sum(exps, axis=0, keepdims=True)

def labeltoarr(trainY):
    labels=np.zeros((trainY.shape[0],10))
    labels[np.arange(trainY.shape[0]),trainY]=1
    return labels.T


def backprop(X,Y,W1,W2,W3,b1,b2,b3,Z1,Z2,Z3,A1,A2,A3):
    dZ3=A3-Y
    dW3=(1/m)*dZ3.dot(A2.T)
    db3=(1/m)*np.sum(dZ3,axis=1,keepdims=True)

    dZ2 = W3.T.dot(dZ3) * (Z2 > 0)  # Derivative of ReLU is 1 if Z > 0, else 0
    dW2 = (1 / m) * dZ2.dot(A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dZ1=W2.T.dot(dZ2)*(Z1>0)
    dW1=(1/m)*dZ1.dot(X.T)
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)
    return dW1,db1,dW2,db2,dW3,db3

def updateweights(dW1,db1,dW2,db2,dW3,db3,W1,W2,W3,b1,b2,b3,alpha):
    W2-=alpha*dW2
    b2-=alpha*db2
    W1-=alpha*dW1
    b1-=alpha*db1
    W3-=alpha*dW3
    b3-=alpha*db3

def accuracy(A2):
    predict=A2.argmax(axis=0)
    return np.sum(predict == trainY)/len(trainY)

def main():
    '''
    model_data = np.load('mnist_model_weights.npz')

    # Access them like a dictionary using the keys you defined
    W1 = model_data['W1']
    W2 = model_data['W2']
    b1 = model_data['b1']
    b2 = model_data['b2']
    W3 = model_data['W3']
    b3 = model_data['b3']'''
    W1,W2,W3,b1,b2,b3=weights()
    Y = labeltoarr(trainY)
    for iteration in range(100000):
        Z1, A1, Z2, A2,Z3,A3 = forwardprop(trainX, W1, W2,W3, b1, b2,b3)
        dW1,db1,dW2,db2,dW3,db3 = backprop(trainX,Y,W1,W2,W3,b1,b2,b3,Z1,Z2,Z3,A1,A2,A3)
        updateweights(dW1,db1,dW2,db2,dW3,db3,W1,W2,W3,b1,b2,b3,0.1)

        print("iteration:", iteration)
        print("accuracy:", accuracy(A3))
        if(iteration%100==0 and iteration>0):
            np.savez_compressed('mnist_model_weights.npz', W1=W1, W2=W2, b1=b1, b2=b2, W3=W3, b3=b3)


main()
