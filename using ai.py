import numpy as np
import pandas as pd
import tkinter as tk
data=pd.read_csv('train.csv')
data=np.array(data)
np.random.shuffle(data)
m,n=data.shape
data_test=data[0:2000]

testY = data_test[:, 0]
testX = data_test[:, 1:] / 255.0

model_data = np.load('mnist_model_weights.npz')
W1 = model_data['W1']
W2 = model_data['W2']
b1 = model_data['b1']
b2 = model_data['b2']
W3 = model_data['W3']
b3 = model_data['b3']

def relu(X):
    return np.maximum(0,X)

def forwardprop(trainX,W1,W2,W3,b1,b2,b3):
    Z1=W1.dot(trainX)+b1
    A1=relu(Z1)
    Z2=W2.dot(A1)+b2
    A2=relu(Z2)
    Z3=W3.dot(A2)+b3
    A3=softmax(Z3)
    return A3.argmax(axis=0)

def softmax(x):
    shiftx = x - np.max(x, axis=0, keepdims=True)
    exps = (np.exp(shiftx))
    return exps / np.sum(exps, axis=0, keepdims=True)


solving=False
def draw(event):
    print(screenpixel)
    global solving
    x,y = event.x,event.y
    x=int(x/20)
    y=int(y/20)
    if screenpixel[x,y]==1:
        return
    solving = True
    for i in range(x-1,x+2):
        for j in range(y-1,y+2):

            if(i<0 or j<0 or i>=28 or j>=28):
                continue
            if screenpixel[j,i]>(1-(((abs(i-x)+abs(j-y))**2)/5)):
                continue

            screenpixel[j,i]=1-((abs(i-x)+abs(j-y))**2)/5

            if (abs(i-x)+abs(j-y))==0:
                col="FFFFFF"
            elif (abs(i-x)+abs(j-y))==1:
                col="ABABAB"
            else:
                col="575757"
            canvas.create_rectangle(i*20,j*20,i*20+20,j*20+20,fill="#"+col)



def refresh(event):
    screenpixel[:]=0
    canvas.create_rectangle(0,0,560,560,fill="#000000")

def solve():
    global solving
    global pred
    if not solving:
        window.after(100, solve)
        return
    #print(screenpixel)
    x = screenpixel.reshape(784, 1)
    prediction = forwardprop(x, W1, W2, W3, b1, b2,b3)
    print("Prediction:", prediction[0])
    label1.config(text="Number is"+str(prediction[0]))
    solving = False
    window.after(100, solve)
    return

window =tk.Tk()
screenpixel=np.zeros((28,28))
window.title("MNIST Digit Classifier")
window.geometry("560x580")
canvas=tk.Canvas(window,width=560,height=560,bg="black")
canvas.bind("<B1-Motion>", draw)
canvas.pack()
button=tk.Button(window,text="refresh")
button.bind("<Button-1>",refresh)
button.pack()
pred=0
label1=tk.Label(window,text="PREDICTED NUMBER:"+str(pred) )
label1.pack()
window.after(100,solve)
window.mainloop()


#main()
