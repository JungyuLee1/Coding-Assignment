import numpy as np
from sklearn import datasets, model_selection
from numpy import random
import matplotlib.pyplot as plt

dataset = datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset[0], dataset[1], test_size = 0.3, random_state = 100)

y_train[y_train!=1]=-1
y_test[y_test!=1]=-1


def Lagrangian_softCGD_withKernel(X, y, learning_rate, epochs, kernel):
    # Define a hyperparameter C
    C = 0.1
    
    lr = learning_rate
    a = np.zeros([len(y)])
    for t in range(epochs):
        i = np.random.randint(0, len(y))
        ai = a[i]*1
        da = 1 - ai*y[i]*y[i]*kernel(X[i,:],X[i,:])
        af = np.median(np.array([0,C,ai+lr*da]))
        a[i] = af
    
    b = np.mean(- np.sum(a[a>0]*y[a>0]*kernel(X[a>0,:],X[a>0,:]),axis=-1))
    
    return a, b


# +
def SVMftn(X,X_train,y_train,a,b,kernel):
    y = np.zeros(X.shape[0])
    for i in range(len(y)):
        y[i] = np.sign(np.sum(a*y_train*kernel(X[i,:],X_train[:,:]),axis=-1)+b)
    
#     y = np.sign(np.sum(a[a>0]*y_train[a>0]*kernel(X,X_train[a>0,:]),axis=-1)+b)
    return y


# -

def prediction_kernel(X,y,X_train,y_train,a,b,kernel):
    result = y-SVMftn(X,X_train,y_train,a,b,kernel)
    count = np.sum(np.abs(result))/2
    acc = 1-count/len(y)
    return acc


def plot_decision_boundary(X, y, a, b, kernel):
    C = 0.1

    X1_min, X1_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    X2_min, X2_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    X1, X2 = np.meshgrid(np.linspace(X1_min, X1_max, 100), np.linspace(X2_min, X2_max, 100))
    
    X_test = np.vstack([X1.flatten(),X2.flatten()]).transpose()
    yPred = SVMftn(X_test,X,y,a,b,kernel)
    yPred = yPred.reshape(X1.shape)
    
    plt.contourf(X1, X2, yPred, alpha=0.2, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.scatter(X[(a>0), 0], X[(a>0), 1], s= 3, c='yellow')


# +
def Kernel_Linear(X1,X2):
    return X1@X2.transpose()

def Kernel_ExactPoly(X1,X2):
    return (X1@X2.transpose())**2

def Kernel_UpPoly(X1,X2):
    return (X1@X2.transpose()+1)**2

def Kernel_UpPoly4(X1,X2):
    return (X1@X2.transpose()+1)**4

def Kernel_Gaussian(X1,X2):
    gamma = 1 # Hyper parameter
    return np.exp(-gamma*np.sum((X1-X2)**2,axis=-1))

def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

def sigmoid(x1, x2, n=0.1, v=0):
    return np.tanh(n*x1@x2.transpose()+v)


# -

kernel = Kernel_UpPoly4

a, b = Lagrangian_softCGD_withKernel(X_train, y_train, 0.01, 1000, kernel)

print("Train acc :",prediction_kernel(X_train,y_train,X_train,y_train,a,b,kernel))
print("Test acc :",prediction_kernel(X_test,y_test,X_train,y_train,a,b,kernel))
plot_decision_boundary(X_train, y_train, a, b, kernel)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()



# +
# Check the official result
from sklearn.svm import SVC
import matplotlib.pyplot as plt

softSVM = SVC(kernel='poly', C=0.1)
softSVM.fit(X_train,y_train)


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

def findAcc(model, X, y):
    yPred = model.predict(X)
    wrong = np.sum(np.abs(yPred - y))/2
    acc = 1 - wrong/len(y)
    return acc

print("Accuracy train :",findAcc(softSVM, X_train, y_train))
print("Accuracy test :",findAcc(softSVM, X_test, y_test))
plot_decision_boundary(softSVM, X_train, y_train)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
# -


