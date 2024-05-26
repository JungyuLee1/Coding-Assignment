[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/m2D4CCO6)
# Coding-Assignment

## Introduction

This assignment focuses on understanding and applying the concepts of soft and hard margins, kernel tricks, and optimization methods in Support Vector Machines (SVM). By solving the given problems and compiling a report, you will demonstrate your grasp of these important machine learning techniques.  

For questions about each question, please email  
Hard margin SVM, Extra Credit : wonhyeok316@gm.gist.ac.kr  
Soft margin SVM : sk000514@gm.gist.ac.kr  
Kernel Tricks : as584868@gm.gist.ac.kr    
  
## Deadline
May 26, 2024 11:59PM KST (One day delay is permitted with linear scale score deduction. (20% per day))

### Submission checklist
* Push your code to Git
* Submit your report to LMS (+Please include a link to your Git code at the beginning of your report)
In your report, you should write down how you solved each question and the visualization that the question requires. It's a good idea to capture code snippets from each step of the process to make your explanation easier to understand.
   
**Both CODE and REPORT must be present for a specific question to count as a score**

## Files
Files you will edit:
* `SVM_hard.py` : You need to modify this file to implement SVM with hard margin.
* `SVM_soft.py` : You need to modify this file to implement SVM with soft margin.
* `SVM_kernel.py` : You need to modify this file to implement kernels which will be used to soft margin.
* `utils.py` : A bunch of utility functions!
* `test.py` : A testing code! I will run this code to run your models.

**You should not use the SVM library provided by sklearn except when loading data.**


## What to submit

### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

## Hard margin SVM (20%)
### Load Dataset
Be sure to load and use the data in the following way (using a subset of IRIS data for completely isolated data). We only use 2 features of data.

`REPORT1`: Draw a decision boundary that perfectly separates the two datasets. Implement the process of finding the optimal decision boundary using **hinge loss** and **coordinate gradient descent**(What you learned in the lesson!). The report should include a reasoned explanation of how the decision boundary was drawn. The size of the margin does not necessarily have to be the maximum, and will only be evaluated for complete separation of the data relative to the decision boundary.

```python
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[:100, :2]
y = iris.target[:100] 
```
<p>
  <img src = "https://github.com/MLDL-2024-GIST/Coding-Assignment/assets/79001832/026a92cf-81f6-4641-a54d-d11601ca73bf" width="400" height="300">
</p>

### Implement and Visualize 
You can implement the Hard margin SVM to predict the class with the input data. In this case, it's okay to evaluate accuracy with train data.
```
>>> from SVM_hard import *
>>> model = HSVM()
>>> model.fit(tr_x, tr_y)
>>> y_pred = model.predict(tr_x)
>>> acc = computeClassificationAcc(tr_y, y_pred) 
>>> print(acc)
1.00 # for example
```
<p>
  <img src = "https://github.com/MLDL-2024-GIST/Coding-Assignment/assets/79001832/54b4fe1c-0558-4544-bdd3-ba7366c18507" width="600" height="300">
</p>

*Hint : The hinge loss can be used to find the optimal decision boundary using the gradient descent method. Compute LOSS for all data points to find the optimized weight and bias. The constant of hinge loss can be modified to determine the margin of the decision boundary. Make sure the decision boundary divides all data points well.

```python
def hinge_loss_gradient(X, y, weights, bias):
    """Your code"""
    return dw, db

def train_svm(X, y, learning_rate, epochs):
    """Your code"""
    return weights, bias
```

## Soft margin SVM (40%)
### Load Dataset
Be sure to load and use the data in the following way (using a subset of IRIS data for completely isolated data). We only use 2 features of data.


`REPORT2`: Find a decision boundary of two classes by solving dual problem. Also, find a decision boundary of two classes by solving primal problem. Compare two solutions of each problem and analyze differences. Visualize the decision boundary of an SVM and analyze how slack variables allow misclassification. You can also adjust the hyperparameters to find best accuracy.

```python
from sklearn import datasets

iris = datasets.load_iris()
X = iris.data[50:, 2:]
y = iris.target[50:] 
```
### Implement and Visualize 
You can implement the soft margin SVM to predict the class with the input data.
```
>>> from SVM_soft import *
>>> from utils import *
>>> model = SSVM()
>>> model.fit(tr_x, tr_y)
>>> y_pred = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_pred) 
>>> print(acc)
0.96 # for example
```

## Kernel Tricks (30%)
Write a program in python that performs Kernel Trick for SVM. Run your own kernel functions on the given dataset.
Visualize the decision boundaries and support vectors of each kernel filters.

`REPORT3`: Apply various kernel filters to SVM and compare their performance. Also, you have to visualize the decision boundaries and support vectors of SVM with different kernel filters. The report should indicate which kernel used in the SVM performed best, including reasons based on visualized data.

### Load Dataset

```python
import sklearn

dataset = sklearn.datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.3, random_state = 100)
```

<p>
  <img src = "https://github.com/MLDL-2024-GIST/Coding-Assignment/assets/97542056/1562d39f-c48d-47c5-8407-b53e0714f9f5" width="400" height="300">
</p>

### Implement and Visualize

You can implement the soft margin SVM to predict the class with the input data.
```
>>> from SVM_kernel import *
>>> model = KSVM()
>>> model.fit(tr_x, tr_y, kernel=kernel)
>>> y_pred = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_pred) 
>>> print(acc)
0.94 # for example
```

<p>
  <img src = "https://github.com/MLDL-2024-GIST/Coding-Assignment/assets/97542056/a9eb4119-35a3-4407-8170-c739b072f48b" width="400" height="300" >
</p>

## Discussion (10%)
`REPORT4`: Compare your implementation with `sklearn` library with same hyper-parameters.

## Extra Credit (10%)
You can earn extra credit for solving this question.   
   
For the optimization methods in the three questions above, use a method other than the one you learned in class and describe which method you used and what the results were. You must also submit your code on Github.
