import numpy as np
from sklearn import datasets

# +
# iris = datasets.load_iris()
# X = iris.data[:100, :2]
# y = iris.target[:100]
# y[y!=1] = -1
# -

iris = datasets.load_iris()
X = iris.data[50:, 2:]
y = iris.target[50:]
y[y!=1] = -1


# +
# # Kernel Tricks
# from sklearn import datasets, model_selection

# dataset = datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
# X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset[0], dataset[1], test_size = 0.3, random_state = 100)

# y_train[y_train!=1]=-1
# y_test[y_test!=1]=-1
# X = X_train
# y = y_train

# +
def hinge_loss(X, y, weights, bias):
    loss = 1 - y*(weights@X.transpose()+bias)
    loss[loss<0] = 0
    return loss


def hinge_loss_gradient(X, y, weights, bias):
    # Find partial derivatives
    dw = -y[...,np.newaxis]*X
    db = -y
    
    # Apply the hindge
    yfx = y*(weights@X.transpose()+bias)
    if isinstance(yfx, np.ndarray):
        dw[yfx>1,:] = 0
        db[yfx>1] = 0
    else:
        if yfx > 1:
            dw = 0
            db = 0
    
    dw = np.sum(dw,axis=0)
    db = np.sum(db,axis=0)
    return dw, db


def plotFunction(w,w0,color='orange'):
    x1 = np.linspace(-10, 10, 400)
    x2 = (-w0 - w[0] * x1) / w[1]

    plt.plot(x1, x2, color=color)
    
def prediction(X,y,w,b):
    yp = np.sign(w@X.transpose()+b)
    acc = np.sum(yp == y)/len(y)
    return acc


# -

# GD - soft margin
def softGD(X, y, learning_rate, epochs):
    e = 1e-6
    # Define a hyperparameter C
    C = 1
    
    w = np.array([0,0])
    b = 0
    for t in range(epochs):
        hdw, hdb = hinge_loss_gradient(X, y, w, b)
        dw = w + C*hdw
        db = C*hdb
        w = w - learning_rate*dw
        b = b - learning_rate*db
        if ((y*(w@X.transpose()+b) > 1).all()): #| (dw@dw<e):
            break
    weights = w
    bias = b
    return weights, bias


w, b = softGD(X, y, 0.001, 100000)

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


# GD - soft margin
def softCGD(X, y, learning_rate, epochs):
    e = 1e-6
    # Define a hyperparameter C
#     C = 10000 # Hard
    C = 10000 # Soft
    
    w = np.array([0,0])
    b = 0
    for t in range(epochs):
        i = np.random.randint(0, len(w))
        hdw, hdb = hinge_loss_gradient(X, y, w, b)
        dw = w + C*hdw
        db = C*hdb
        w[i] = w[i] - learning_rate*dw[i]
        b = b - learning_rate*db
        if (y*(w@X.transpose()+b) > 1).all() | (dw@dw<e):
            break
    weights = w
    bias = b
    return weights, bias


w, b = softCGD(X, y, 0.01, 10000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -


def hinge_loss_gradient2(X, y, weights, bias):
    # Find partial derivatives
    dw = -y[...,np.newaxis]*X
    db = -y
    
    # Apply the hindge
    yfx = y*(weights@X.transpose())
    if isinstance(yfx, np.ndarray):
        dw[yfx>1,:] = 0
        db[yfx>1] = 0
    else:
        if yfx > 1:
            dw = 0
            db = 0
    
    dw = np.sum(dw,axis=0)
    db = np.sum(db,axis=0)
    return dw, db


# GD - soft margin
def softCGD2(X, y, reg, epochs):
    e = 1
    # Define a hyperparameter C
#     C = 10000 # Hard
    C = 0.01 # Soft
    
    w = np.array([0,0])
    b = 0
    for t in range(epochs):
        lr = 1/(reg*(t+1))
        i = np.random.randint(0, len(y))
        hdw, hdb = hinge_loss_gradient2(X[i,:], y[i], w, b)
        dw = reg*w + C*hdw
        db = C*hdb
        w = w - lr*dw
        b = b - lr*db
        if (y*(w@X.transpose()+b) > 1).all() | (dw@dw<e):
            break
    weights = w
    bias = b
    return weights, bias


w, b = softCGD2(X, y, 0.01, 10000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -


# +
# CGD - hard margin
def Lagrangian_softCGD(X, y, learning_rate, epochs):
    # Define a hyperparameter C
    C = 0.2
    
    w = np.array([0,0])
    a = np.zeros([len(y)])
    for t in range(epochs):
        i = np.random.randint(0, len(y))
        ai = a[i]
        gD = w@(y[i]*X[i,:])-1
        af = np.median(np.array([0,C,ai-gD/(X[i,:]@X[i,:])]))
        a[i] = af
        w = w + (af-ai)*y[i]*X[i,:]
        
#         b = np.mean(-w@X[a>0].transpose())
#         if (y*(w@X.transpose()+b) > 1).all():
#             break
    b = np.mean(-w@X[a>0].transpose())
    
    weights = w
    bias = b
    return weights, bias, a


# -

w, b, a = Lagrangian_softCGD(X, y, 0.01, 100000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')
plt.plot(X[a>0,0],X[a>0,1],'yo',markersize=2)

plotFunction(w,b)
plotFunction(w,b+1,color='green')
plotFunction(w,b-1,color='green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -




# CGD - hard margin
def Lagrangian_softCGD(X, y, learning_rate, epochs):
    # Define a hyperparameter C
    C = 1
    lr = learning_rate
    
    a = np.zeros([len(y)])
    for t in range(epochs):
        i = np.random.randint(0, len(y))
        ai = a[i]
        af = np.median(np.array([0,C,ai + lr*(1-ai*y[i]*y[i]*X[i,:]@X[i,:])]))
        a[i] = af
        
    w = np.sum(a*y*X.transpose(),axis=-1)
    b = np.mean(-w@X[(a>0)&(a<C)].transpose())
    
    weights = w
    bias = b
    return weights, bias


w, b = Lagrangian_softCGD(X, y, 0.01, 10000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plotFunction(w,b+1)
plotFunction(w,b-1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -


def Pegasos(X, y, reg, epochs):
    w = np.array([0,0])
    for t in range(1,epochs+1):
        lr = 1/(reg*t)
        n = np.random.randint(0, len(y))
        Xt = X[n,:]
        yt = y[n]
        if yt*w@Xt.transpose() < 1:
            w = (1-lr*reg)*w + (lr*yt)*Xt
        else:
            w = (1-lr*reg)*w
            
    b = -w@np.mean(X,axis=0)
    
    return w,b


w, b = Pegasos(X, y, 100000, 10000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -


# +
def Pegasos2(X, y, reg, epochs):
    w = np.array([0,0])
    b = 0
    for t in range(1,epochs+1):
        lr = 1/(reg*t)
        n = np.random.randint(0, len(y))
        Xt = X[n,:]
        yt = y[n]
        if yt*(w@Xt.transpose()+b) < 1:
            w = (1-lr*reg)*w + (lr*yt)*Xt
            b = b + lr*yt
        else:
            w = (1-lr*reg)*w
            
#     b = -w@np.mean(X,axis=0)
    
    return w,b


# -

w, b = Pegasos2(X, y, 0.1, 100000)

# +
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -




def Pegasos_Dual(X, y, reg, epochs):
    a = np.zeros(len(y))
    for t in range(1,epochs+1):
        i = np.random.randint(0, len(a))
        lr = 1/(reg*t)
        
        if y[i]*(1/lr)*np.sum(a*y*(X[i]@X.transpose())) < 1:
            a[i] = a[i] + 1
        else:
            a[i] = a[i]

    w = np.sum(a*y*X.transpose(),axis=-1)
    b = -w@np.mean(X,axis=0)
    
    return w,b


w, b = Pegasos_Dual(X, y, 10, 10000)

# +
w, b = Pegasos_Dual(X, y, 10, 10000)
import matplotlib.pyplot as plt

print("Accuracy is :",prediction(X,y,w,b))

plt.plot(X[(y==1),0],X[(y==1),1],'ro')
plt.plot(X[(y!=1),0],X[(y!=1),1],'bo')

plotFunction(w,b)
plt.xlabel('x1')
plt.ylabel('x2')
plt.xlim(np.min(X[:,0])-0.2,np.max(X[:,0])+0.2)
plt.ylim(np.min(X[:,1])-0.2,np.max(X[:,1])+0.2)
plt.show()
# -


# +
# Kernel Tricks
from sklearn import datasets, model_selection

dataset = datasets.make_moons(n_samples = 300, noise = 0.3, random_state = 20) # you can change noise and random_state where noise >= 0.15
X_train, X_test, y_train, y_test = model_selection.train_test_split(dataset[0], dataset[1], test_size = 0.3, random_state = 100)

y_train[y_train!=1]=-1
y_test[y_test!=1]=-1


# -

# CGD - hard margin
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
# def SVMftn(X,X_train,y_train,a,b,kernel):
#     y = np.zeros(X.shape[0])
#     for i in range(len(y)):
#         y[i] = np.sign(np.sum(a[a>0]*y_train[a>0]*kernel(X[i,:],X_train[a>0,:]),axis=-1)+b)
    
# #     y = np.sign(np.sum(a[a>0]*y_train[a>0]*kernel(X,X_train[a>0,:]),axis=-1)+b)
#     return y

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


def plot_decision_boundary(X_train, y_train, a, b, kernel):
    C = 0.1

    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    X_test = np.vstack([xx.flatten(),yy.flatten()]).transpose()
    Z = SVMftn(X_test,X_train,y_train,a,b,kernel)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.2, cmap='coolwarm')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
    plt.scatter(X_train[(a>0), 0], X_train[(a>0), 1], s= 3, c='yellow')


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

kernel = Kernel_Gaussian

a, b = Lagrangian_softCGD_withKernel(X_train, y_train, 0.01, 1000, kernel)

print("Train acc :",prediction_kernel(X_train,y_train,X_train,y_train,a,b,kernel))
print("Test acc :",prediction_kernel(X_test,y_test,X_train,y_train,a,b,kernel))
plot_decision_boundary(X_train, y_train, a, b, kernel)
plt.xlabel('x1')
plt.ylabel('x2')


def Pegasos_Dual_kernel(X, y, reg, epochs,kernel):
    a = np.zeros(len(y))
    for t in range(1,epochs+1):
        i = np.random.randint(0, len(a))
        lr = 1/(reg*t)
        
        if y[i]*(1/lr)*np.sum(a*y*kernel(X[i],X)) < 1:
            a[i] = a[i] + 1
        else:
            a[i] = a[i]

    b = np.mean(- np.sum(a[a>0]*y[a>0]*kernel(X[a>0,:],X[a>0,:]),axis=-1))
    
    return a, b



a, b = Pegasos_Dual_kernel(X_train, y_train, 0.001, 1000, kernel)

print("Train acc :",prediction_kernel(X_train,y_train,X_train,y_train,a,b,kernel))
print("Test acc :",prediction_kernel(X_test,y_test,X_train,y_train,a,b,kernel))
plot_decision_boundary(X_train, y_train, a, b, kernel)
plt.xlabel('x1')
plt.ylabel('x2')



# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# 데이터 생성 (2차원)
np.random.seed(42)
X = X_train
y = y_train

# 가우시안 커널 함수 정의
def gaussian_kernel(x1, x2, sigma=1.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

# 커널 행렬 계산
def compute_kernel_matrix(X, kernel_func):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

# 목적 함수 정의
def objective(alpha, K, y):
    return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)

# 제약 조건 정의
def zerofun(alpha):
    return np.dot(alpha, y)

# 초기화 및 제약 조건 설정
initial_alpha = np.zeros(X.shape[0])
C = 1.0  # Regularization parameter
bounds = [(0, C) for _ in range(X.shape[0])]
constraints = {'type': 'eq', 'fun': zerofun}

# 커널 행렬 계산
K = compute_kernel_matrix(X, gaussian_kernel)

# 최적화 수행
result = minimize(objective, initial_alpha, args=(K, y), bounds=bounds, constraints=constraints)
alpha = result.x

# 서포트 벡터
support_vectors = alpha > 1e-5

# 가중치 벡터 계산 (서포트 벡터를 이용)
w = np.sum(alpha[support_vectors, None] * y[support_vectors, None] * X[support_vectors], axis=0)

# 절편 b 계산
b = np.mean(y[support_vectors] - np.sum((alpha[support_vectors, None] * y[support_vectors, None] * K[support_vectors][:, support_vectors]), axis=0))

# 예측 함수 정의
def predict(X_new, X, alpha, y, b, kernel_func):
    K_new = np.array([np.sum(alpha * y * np.array([kernel_func(x, x_new) for x in X])) for x_new in X_new])
    return np.sign(K_new + b)

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, alpha, b, kernel_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_grid, X, alpha, y, b, kernel_func)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
    plt.title('Kernel SVM with Gaussian Kernel and Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 예측 및 시각화
plot_decision_boundary(X, y, alpha, b, gaussian_kernel)

# +
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (2차원)
np.random.seed(42)
X = X_train
y = y_train

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title('Generated 2D Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 가우시안 커널 함수 정의
# def gaussian_kernel(x1, x2, sigma=1.0):
#     return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (sigma ** 2)))

def gaussian_kernel(x1, x2, sigma=1.0):
    return (np.dot(x1,x2)+1)**2

# 커널 행렬 계산
def compute_kernel_matrix(X, kernel_func):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

# 목적 함수 및 그레디언트 정의
def objective(alpha, K, y):
    return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)

def gradient(alpha, K, y):
    return np.dot(K, alpha * y) * y - 1

# 경사하강법을 이용한 라그랑지 승수 최적화
def gradient_descent(K, y, C, lr=0.01, epochs=1000):
    alpha = np.zeros(len(y))
    for epoch in range(epochs):
        grad = gradient(alpha, K, y)
        alpha = alpha - lr * grad
        alpha = np.clip(alpha, 0, C)  # 제약 조건: 0 <= alpha <= C
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Objective: {objective(alpha, K, y)}')
    return alpha

# 서포트 벡터 및 절편 계산
def compute_b(X, y, alpha, K, kernel_func):
    support_vectors = alpha > 1e-5
    b = np.mean(y[support_vectors] - \
                np.sum((alpha[support_vectors, None] * y[support_vectors, None] * K[support_vectors][:, support_vectors]), axis=0))
    return b

def predict(X_new, X, alpha, y, b, kernel_func):
    K_new = np.array([np.sum(alpha * y * np.array([kernel_func(x, x_new) for x in X])) for x_new in X_new])
    return np.sign(K_new + b)

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, alpha, b, kernel_func):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_grid, X, alpha, y, b, kernel_func)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
    plt.title('Kernel SVM with Gaussian Kernel and Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 커널 행렬 계산
K = compute_kernel_matrix(X, gaussian_kernel)

# 경사하강법을 사용하여 라그랑지 승수 최적화
alpha = gradient_descent(K, y, C=1.0, lr=0.01, epochs=1000)

# 절편 b 계산
b = compute_b(X, y, alpha, K, gaussian_kernel)

# 결정 경계 시각화
plot_decision_boundary(X, y, alpha, b, gaussian_kernel)

# +
import numpy as np
import matplotlib.pyplot as plt

# 데이터 생성 (2차원)
np.random.seed(42)
X = X_train
y = y_train

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.title('Generated 2D Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 시그모이드 커널 함수 정의
def sigmoid_kernel(x1, x2, gamma=0.1, r=0.0):
    return np.exp(-np.linalg.norm(x1 - x2) ** 2 / (2 * (1 ** 2)))
#     return np.tanh(gamma * np.dot(x1, x2) + r)

# 커널 행렬 계산
def compute_kernel_matrix(X, kernel_func):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = kernel_func(X[i], X[j])
    return K

# 목적 함수 및 그레디언트 정의
def objective(alpha, K, y):
    return 0.5 * np.sum((alpha * y)[:, None] * (alpha * y)[None, :] * K) - np.sum(alpha)

def gradient(alpha, K, y):
    return np.dot(K, alpha * y) * y - 1

# 경사하강법을 이용한 라그랑지 승수 최적화
def gradient_descent(K, y, C, lr=0.01, epochs=1000):
    alpha = np.zeros(len(y))
    for epoch in range(epochs):
        grad = gradient(alpha, K, y)
        alpha = alpha - lr * grad
        alpha = np.clip(alpha, 0, C)  # 제약 조건: 0 <= alpha <= C
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Objective: {objective(alpha, K, y)}')
    return alpha

# 서포트 벡터 및 절편 계산
def compute_b(X, y, alpha, K, kernel_func):
    support_vectors = (alpha > 1e-5) & (alpha < C)
    b = np.mean(y[support_vectors] - \
                np.sum((alpha[support_vectors, None] * y[support_vectors, None] * K[support_vectors][:, support_vectors]), axis=0))
    return b

def predict(X_new, X, alpha, y, b, kernel_func):
    K_new = np.array([np.sum(alpha * y * np.array([kernel_func(x, x_new) for x in X])) for x_new in X_new])
    return np.sign(K_new + b)

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, alpha, b, kernel_func, C):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    Z = predict(X_grid, X, alpha, y, b, kernel_func)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, levels=[-1, 0, 1], alpha=0.2, colors=['blue', 'red'])
    
    # 원래 데이터 포인트
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', marker='o')
    
    # 서포트 벡터 표시
    support_vectors = alpha > 1e-5
    plt.scatter(X[support_vectors, 0], X[support_vectors, 1], facecolors='none', edgecolors='k', s=100, label='Support Vectors')
    
    # 마진에 있는 서포트 벡터 표시
    margin_support_vectors = alpha == C
    plt.scatter(X[margin_support_vectors, 0], X[margin_support_vectors, 1], facecolors='none', edgecolors='g', s=100, label='Margin Support Vectors')
    
    plt.title('Kernel SVM with Sigmoid Kernel and Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()

# 커널 행렬 계산
C = 1.0
K = compute_kernel_matrix(X, sigmoid_kernel)

# 경사하강법을 사용하여 라그랑지 승수 최적화
alpha = gradient_descent(K, y, C, lr=0.01, epochs=1000)

# 절편 b 계산
b = compute_b(X, y, alpha, K, sigmoid_kernel)

# 결정 경계 시각화
plot_decision_boundary(X, y, alpha, b, sigmoid_kernel, C)
# -


