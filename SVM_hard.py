# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from utils import *

# +
# Data import

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100]

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
    C = 10000
    
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
w, b = softGD(X, y, 0.001, 10000)

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
# Algorithm 2 : SVM with coordinate descent method
# In this method, updates only one dimension of w for each iteration

def softCGD(X, y, learning_rate, epochs):

    # Define a hyperparameter C
    C = 10000
    
    # Initialization
    w = np.array([0,0])
    b = 0
    
    # Coordinate descent algorithm
    for t in range(epochs):
        # Choose one dimension
        i = np.random.randint(0, len(w))
        
        # Find gradient
        hdw, hdb = hinge_loss_gradient(X, y, w, b)
        dw = w + C*hdw
        db = C*hdb
        
        # Update parameter
        w[i] = w[i] - learning_rate*dw[i]
        b = b - learning_rate*db
        
#         # Breaking condition (for hard SVM)
#         if (y*(w@X.transpose()+b) > 1).all():
#             break
    
    # Output
    weights = w
    bias = b
    return weights, bias


# -

# Train model
w, b = softCGD(X, y, 0.001, 10000)

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



# Algorithm 3 : Pegasos algorithm for primal problem (Update susing one data for each iteration)
def Pegasos(X, y, reg, epochs):
    
    # Initialization
    w = np.array([0,0])
    b = 0
    
    for t in range(1,epochs+1):
        # Set learning rate
        lr = 1/(reg*t)
        
        # Choose one data
        n = np.random.randint(0, len(y))
        Xt = X[n,:]
        yt = y[n]
        
        # Update parameters
        if yt*(w@Xt.transpose()+b) < 1:
            w = (1-lr*reg)*w + (lr*yt)*Xt
            b = b + lr*yt
        else:
            w = (1-lr*reg)*w
            
    # Output
    return w,b


w, b = Pegasos(X, y, 0.1, 10000)

# +
import matplotlib.pyplot as plt

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



# Algorithm 4 : SVM of dual problem with coordinate descent method
def Lagrangian_softCGD(X, y, learning_rate, epochs):
    # Define a hyperparameter C
    C = 10000
    
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
        af = np.median(np.array([0,C,ai-gD/(X[i,:]@X[i,:])]))
        # Update parameters
        a[i] = af
        w = w + (af-ai)*y[i]*X[i,:]
        
        # Loop bearking condition
#         b = np.mean(-w@X[a>0].transpose())
#         if (y*(w@X.transpose()+b) > 1).all():
#             break

    # Calculate w with the result paremters
    b = np.mean(-w@X[a>0].transpose())
    
    # Output
    weights = w
    bias = b
    return weights, bias


w, b = Lagrangian_softCGD(X, y, 0.01, 10000)

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



# Algorithm extra : Pegasos algorithm for dual problem
def Pegasos_Dual(X, y, reg, epochs):
    # Initialize
    a = np.zeros(len(y))
    
    for t in range(1,epochs+1):
        # set learning rate
        lr = 1/(reg*t)
        
        # Choose one dimension
        i = np.random.randint(0, len(a))
        
        # Update parameters (Lagrangian multiplier)
        if y[i]*(1/lr)*np.sum(a*y*(X[i]@X.transpose())) < 1:
            a[i] = a[i] + 1
        else:
            a[i] = a[i]

    # Calculate the parameters
    w = np.sum(a*y*X.transpose(),axis=-1)
    b = -w@np.mean(X,axis=0)
    
    # Output
    return w,b


w, b = Pegasos_Dual(X, y, 10, 10000)

# +
w, b = Pegasos_Dual(X, y, 10, 10000)
import matplotlib.pyplot as plt

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



# +
# Check the official result
from sklearn.svm import SVC
import matplotlib.pyplot as plt

softSVM = SVC(kernel='linear', C=1e6)
softSVM.fit(X,y)
w = softSVM.coef_[0]
b = softSVM.intercept_[0]

# +
import matplotlib.pyplot as plt

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


# +
def plot_decision_boundary(model, X, y):
    h = .02  # step size in the mesh
    X1_min, X1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    X2_min, X2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    X1, X2 = np.meshgrid(np.arange(X1_min, X1_max, h), np.arange(X2_min, X2_max, h))
    yPred = model.predict(np.c_[X1.ravel(), X2.ravel()])
    yPred = yPred.reshape(X1.shape)
    
    plt.contourf(X1, X2, yPred, alpha=0.2, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.show()

def findAcc(model, X, y):
    yPred = model.predict(X)
    wrong = np.sum(np.abs(yPred - y))/2
    acc = 1 - wrong/len(y)
    return acc

print("Accuracy :",findAcc(softSVM, X, y))
plot_decision_boundary(softSVM, X, y)
plt.xlabel('x1')
plt.ylabel('x2')
