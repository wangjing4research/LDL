

import numpy as np
from sklearn.neighbors import NearestNeighbors
from numpy.random import multinomial
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from scipy.optimize import minimize
from scipy.special import softmax



def LDL2SL(Y):
    Y_hat = np.zeros(Y.shape[0], dtype=np.int32)
    for (i, pro) in enumerate(Y):
        Y_hat[i] = np.where(multinomial(1, pro))[0][0]
    
    return Y_hat

def LDL2Bayes(Y):
    return np.argmax(Y, 1)



class bfgs_ldl:

    def __init__(self, x, y, C = 0):
        self.C = C
        self.x = x
        self.y = y
        self.n_features, self.n_outputs = self.x.shape[1], self.y.shape[1]

    def predict(self, x):
        p_yx = np.dot(x, self.W)
        #p_yx = p_yx - np.max(p_yx, 1).reshape(-1, 1)
        #p_yx = normalize(np.exp(p_yx), norm = 'l1')
        p_yx = softmax(p_yx, axis = 1)
        return p_yx

    def object_fun(self, weights):
        W = weights.reshape(self.n_outputs, self.n_features).transpose()
        p_yx = np.dot(self.x, W)
        #p_yx = p_yx - np.max(p_yx, 1).reshape(-1, 1)
        #y_pre = normalize(np.exp(p_yx), norm = 'l1')
        
        y_pre = softmax(p_yx, axis = 1)
        
        func_loss = self.loss(y_pre) + self.C * 0.5 * np.dot(weights, weights) 
        func_grad = self.gradient(y_pre) + self.C * weights
        
        return func_loss, func_grad

    def gradient(self, y_pre):
        grad = np.dot(self.x.T, y_pre - self.y)
        return grad.transpose().reshape(-1, ) 

    def loss(self, y_pre):
        y_true = np.clip(self.y, 1e-7, 1)
        y_pre = np.clip(y_pre, 1e-7, 1)
        return -1 * np.sum(y_true * np.log(y_pre))

    def fit(self):        
        weights = np.random.uniform(-0.1, 0.1, self.n_features * self.n_outputs)
        optimize_result = minimize(self.object_fun, weights, method = 'l-BFGS-b', jac = True,
                                   options = {'gtol':1e-6, 'disp': False, 'maxiter':600 })
        
        weights = optimize_result.x
        self.W = weights.reshape(self.n_outputs, self.n_features).transpose()
        
    def __str__(self):
        name = "bfgs_ldl_" +  str(self.C)
        return name


class AA_KNN:
    def __init__(self,train_X, train_Y, K = 5):
        self.train_X = None
        self.train_Y = None
        self.K = K
        self.model =NearestNeighbors(n_neighbors=self.K, algorithm='brute')
        self.train_X = train_X
        self.train_Y = train_Y
    
    def fit(self, test_X, test_Y):
        self.test_X = test_X
        self.test_Y = test_Y
        
        self.model.fit(self.train_X)
        _, inds = self.model.kneighbors(self.test_X)
        self.inds = inds
        
    def score(self, k):
        self.k = k
        Y_hat = None
        i = 0
        for i in range(self.k):
            ind = self.inds.T[i]
            if Y_hat is None:
                Y_hat = self.train_Y[ind]
            else:
                Y_hat += self.train_Y[ind]
        Y_hat = Y_hat / self.k      
        
        return (zero_one_measure(self.test_Y, Y_hat), error(self.test_Y, Y_hat))
        
    def to_str(self, k):
        return "AA-KNN_K=" + str(k)


class PT_Bayes:
    def __init__(self, train_X, train_Y, toSL=LDL2SL):
        self.train_X = train_X
        self.model = GaussianNB()
        self.toSL = toSL
        self.train_Y = self.toSL(train_Y)
        
    
    def fit(self):

        self.model.fit(self.train_X, self.train_Y)
    
    def score(self, test_X, test_Y):
        Y_hat = self.model.predict(test_X)
        return (zero_one_measure(self.toSL(test_Y), Y_hat), error(test_Y, Y_hat))
    
    def __str__(self):
        if self.toSL is LDL2SL:
            return "PT-Bayes_LDL2SL"
        else:
            return "PT-Bayes_Bayes"


class PT_SVM:
    def __init__(self, train_X, train_Y, C=1, toSL=LDL2SL):
        self.train_X = train_X
        self.C = C
        self.model = SVC(self.C, kernel='rbf', gamma = 'scale')
        self.toSL = toSL
        self.train_Y = self.toSL(train_Y)

    def fit(self):
        self.model.fit(self.train_X, self.train_Y)
        
    def score(self, test_X, test_Y):
        Y_hat = self.model.predict(test_X)
        return (zero_one_measure(self.toSL(test_Y), Y_hat), error(test_Y, Y_hat))
    def __str__(self):
        if self.toSL is LDL2SL:
            return "PT-SVM_LDL2SL_C=" + str(self.C)
        else:
            return "PT-SVM_Bayes_C=" + str(self.C)

