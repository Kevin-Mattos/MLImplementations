# -*- coding: utf-8 -*-

import sklearn.datasets as ds, numpy as np, matplotlib.pyplot as plt

class logReg:
    
    def __init__(self, alpha = 0.05, num_of_iter = 300):
        self.thetas = None
        self.tempThetas = None
        self.numer_of_classes = None
        self.X = None
        self.y = None
        
        self.alpha = alpha
        self.num_of_iter = num_of_iter
        self.J = []
        self.aux = []
        self.actual_pred = 0
        
    def quantity_of_classes(self):
        _, count = np.unique(self.y, return_counts=True)
        return len(count)
        
    def fit(self, X, y):
        self.X = self.add_ones_row(X)
        self.y = np.copy(y)   
        self.numer_of_classes = self.quantity_of_classes()        
        rows  = len(self.X[0, :])  
        
        self.thetas = np.ones(rows)
        for i in range(self.numer_of_classes):  
            self.actual_pred = i
            self.temp_y = np.copy(self.y) == i
            self.gradientDescent(self.temp_y)
            self.aux.append(self.thetas)
    
    def gradientDescent(self, temp_y):
        
        for i in range(self.num_of_iter):
            self.J.append(self.cost())
            self.tempThetas = self.thetas - self.alpha*self.derivated_cost(temp_y)
            self.thetas =  self.tempThetas
        self.plot_cost()
       
        
    def sigmoid(self):
        z = np.dot(self.X, self.thetas)
        return 1/(1 + np.exp(-z))
    
    def sigmoid_pred(self, thetas, X):
        z = np.dot(X, thetas)
        return 1/(1 + np.exp(-z))  
        
    def derivated_cost(self, temp_y):
        haga = self.sigmoid() - temp_y
        haga = np.dot(haga, self.X)        
        return haga
    
    
    def plot_cost(self):
        plt.plot(self.J)
        plt.ylabel('Cost value')
        plt.xlabel('Number Of Iterations')
        plt.title('Cost Function for class number {}'.format(self.actual_pred))
        #plt.legend()
        plt.show()
        self.J = []
    def pred(self, X):
        probs = []
        for i in self.aux:#possibility comparing one to all
            probs.append(self.sigmoid_pred(i, X))
        
        probs = np.array(probs).T       
        prediction = np.argmax(probs, axis = 1 )
        
       
        return prediction

        
    def cost(self):
        left = np.dot(self.temp_y, np.log(self.sigmoid()))
        right = np.dot((1 - self.temp_y), np.log(1 - self.sigmoid()))
                       
        result = -left - right
        
        return result/len(self.temp_y)
    
    def add_ones_row(self, X):
        ones = np.ones(len(X[:,0])).reshape(-1,1)
        return np.concatenate((ones, X),1)
        
def compare(real, pred):
    
    
    concat = np.concatenate((real.reshape(-1,1), pred.reshape(-1,1)), 1)
    print(concat)


X, y = ds.load_iris(return_X_y=True)
# =============================================================================
# X = X[y<2]
# y = y[y<2]
# =============================================================================

splitter = int(len(y)/5)

X_train, Xtest = X[ splitter:,  :], X[: splitter, :]


y_train, y_test = y[ splitter:], y[: splitter]

clasf = logReg()

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)



clasf.fit(X,y)

print('eae')
pred = clasf.pred(clasf.X)
compare(y, pred)

from sklearn.linear_model import LogisticRegression
re = LogisticRegression()
re.fit(X,y)
pred = re.predict(X)

compare(y, pred)

    
    
    
    
    