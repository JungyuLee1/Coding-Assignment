import numpy as np
import matplotlib.pyplot as plt

# you can modify util functions here what you need
# this python file will not be included in the grading

# This function write line with inputs w and b
def plotFunction(w,w0,color='orange'):
    x1 = np.linspace(-10, 10, 400)
    x2 = (-w0 - w[0] * x1) / w[1]

    plt.plot(x1, x2, color=color)


# This function predict class and calculate the accuracy with input X, y, w and b
def prediction(X,y,w,b):
    yp = np.sign(w@X.transpose()+b)
    acc = np.sum(yp == y)/len(y)
    return acc


