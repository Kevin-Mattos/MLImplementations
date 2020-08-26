import numpy as np, pandas as pd, matplotlib.pyplot as plt

       
   
   
class MyLinearRegr:
    
    def __init__(self, alpha = 0.08, num_iter = 150):
        self.X = None
        self.y = None
        self.thetas = None
        self.thetas = None
        self.tempThetas = None
        
        self.num_iter = num_iter
        self.alpha = alpha
        self.J = []
        
    
    def fit(self, X, y):
        self.X = self.add_ones_row(X)
        self.y = np.copy(y)   
        
        rows  = len(self.X[0, :])
        self.thetas = np.ones(rows)    
        self.tempThetas = np.copy(self.thetas)        
    
        for i in range(self.num_iter):
            self.gradient_descent()
   
    
    def update(self):
        self.thetas = np.copy(self.tempThetas)      
           
        
    
    def gradient_descent(self):
        self.J.append(self.cost_function())  
        
        derivated_cost_j = self.partial_derivative_cost()
        descent = self.alpha*derivated_cost_j/(len(self.y))
        self.tempThetas = self.thetas - descent      
        self.update()
      
        
    def cost_function(self):
        return self.j_of_theta()
    
    def j_of_theta(self):
        haga = self.h(self.X)
        haga_y = haga - self.y        
        sum_of_quads = np.square(haga_y).sum()/(2/len(self.y))
        return sum_of_quads
    
    def add_ones_row(self, X):
        ones = np.ones(len(X[:,0])).reshape(-1,1)
        return np.concatenate((ones, X),1)
        
    
    def h(self, X):
        return  np.dot(X, self.thetas)
    
    def partial_derivative_cost(self):
        aga = self.h(self.X) - self.y    
        sumOfElements =  np.dot(aga, self.X) 
        result = sumOfElements
        return result

    def predict(self, X):
        return self.h(X)
    
    def plot_cost(self):
        plt.plot(self.J)
        plt.ylabel('Cost value')
        plt.xlabel('Number Of Iterations')
        plt.title('Cost Function')
        plt.legend()
        plt.show()
        
   
def compare(real, pred):
    concat = np.concatenate((real, pred), 1)
    print(concat)





dataset = pd.read_csv('heart.data.csv', index_col = 'id')

y = dataset.iloc[:, 1].values#.reshape(-1,1)
X = dataset.iloc[:, :-1].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X = sc_x.fit_transform(X)

sc_y = StandardScaler()
# =============================================================================
# y = sc_y.fit_transform(y)
# =============================================================================

lr = MyLinearRegr(num_iter=70)
lr.fit(X, y)

pred = lr.predict(lr.X)
    
y_test = y.reshape(-1,1)#sc_y.inverse_transform(y)

y_pred = pred.reshape(-1,1)#sc_y.inverse_transform(pred).reshape(len(y), 1)
lr.plot_cost()

compare(y_test, y_pred)











