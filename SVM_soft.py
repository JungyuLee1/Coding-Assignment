# +
import numpy as np
from sklearn import datasets
from numpy import random
import matplotlib.pyplot as plt
from utils import *

iris=datasets.load_iris()
X=iris.data[50:,[2,3]]
y=iris.target[50:]



"""
Using below code.
class SSVM:
    def __init__(self,  ):
        pass
    def fit(self, X, y):
        pass
    def predict(self, X):
        pass

def computeClassificationAcc(gt_y,pred_y):
    pass
"""


# +
y[y!=1] = -1

plt.plot(X[y==1,0],X[y==1,1],'bo')
plt.plot(X[y==-1,0],X[y==-1,1],'ro')
plt.xlabel('X 1')
plt.ylabel('X 2')
plt.show()


# +
# Defind hinge loss fuctions

def hinge_loss(X, y, weights, bias):
    loss = 1 - y*(weights@X.transpose()+bias)
    loss[loss<0] = 0
    return loss

def hinge_loss_gradient(X, y, weights, bias):
    # Find partial derivatives
    dw = -y[...,np.newaxis]*X
    db = -y
    
    # Apply the hindge (Erase where y*f(x) > 1)
    yfx = y*(weights@X.transpose()+bias)
    if isinstance(yfx, np.ndarray):
        dw[yfx>1,:] = 0
        db[yfx>1] = 0
    else:
        if yfx > 1:
            dw = 0
            db = 0
    
    # Add along the data points
    dw = np.sum(dw,axis=0)
    db = np.sum(db,axis=0)
    return dw, db


# +
# Algorithm 1: SVM with normal gradient descent method
# With large C, it works as hard margin SVM. with small C, it works as soft margin SVM.
def softGD(X, y, learning_rate, epochs):
    # hyper parameter
    C = 1
    
    # Initialize vparameters
    w = np.array([0,0])
    b = 0
    
    # Conduct gradient descent method : iter : w - dw
    for t in range(epochs):
        hdw, hdb = hinge_loss_gradient(X, y, w, b)
        dw = w + C*hdw
        db = C*hdb
        w = w - learning_rate*dw
        b = b - learning_rate*db
        
#         # Loop beaking condition
#         if ((y*(w@X.transpose()+b) > 1).all()):
#             break
            
    # Output
    weights = w
    bias = b
    return weights, bias


# -

# Train model
w, b = softGD(X, y, 0.001, 1000000)

# +
# Plot the result
print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')


plotFunction(w,b)
plotFunction(w,b+1,color = 'green')
plotFunction(w,b-1,color = 'green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -



# +
# Algorithm 4 : SVM of dual problem with coordinate descent method
def Lagrangian_softCGD(X, y, learning_rate, epochs):
    # Define a hyperparameter C
    C = 1
    
    # Initialization
    w = np.array([0,0])
    a = np.zeros([len(y)])
    
    for t in range(epochs):
        # Choose one dimentsion
        i = np.random.randint(0, len(y))
        ai = a[i]
        
        # Calculate the gradient
        gD = w@(y[i]*X[i,:])-1
        # Projection
        af = np.median(np.array([0,C,ai-learning_rate*gD/(X[i,:]@X[i,:])]))
        # Update parameters
        a[i] = af
        w = w + (af-ai)*y[i]*X[i,:]
        
# #         Loop bearking condition
#         b = np.mean(-w@X[a>0].transpose())
#         if (y*(w@X.transpose()+b) > 1).all():
#             break

    # Calculate w with the result paremters
    b = np.mean(-w@X[a>0].transpose())
    
    # Output
    weights = w
    bias = b
    return weights, bias


# -

w, b = Lagrangian_softCGD(X, y, 0.001, 1000000)

# +
# Plot the result
print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plotFunction(w,b+1,color='green')
plotFunction(w,b-1,color='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -



# Check the official result
from sklearn.svm import SVC
softSVM = SVC(kernel='linear', C=1)
softSVM.fit(X,y)
w = softSVM.coef_[0]
b = softSVM.intercept_[0]

# +
print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plotFunction(w,b+1,color='green')
plotFunction(w,b-1,color='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
