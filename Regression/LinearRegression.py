import numpy as np, pandas as pd, matplotlib.pyplot as plt

       
##CUSTO DA FUNCAO REGULARIZADA É DIFERENTE
   
class MyLinearRegr:
    
    def __init__(self, alpha = 0.08, num_iter = 150, isRegularized = False, lambdas = 150, is_normal_equation = False):
        self.X = None
        self.y = None
        self.thetas = None
        self.thetas = None
        self.tempThetas = None
        
        self.is_normal_equation = is_normal_equation
        self.isRegularized = isRegularized
        self.lambdas = lambdas
        self.num_iter = num_iter
        self.alpha = alpha
        self.J = []
        
    
    def fit(self, X, y):
        self.X = self.add_ones_row(X)
        self.y = np.copy(y)
        
        
        self.rows  = len(self.X[:, 0])
        self.columns = len(self.X[0, :])
        self.thetas = np.ones(self.columns)   
        self.tempThetas = np.copy(self.thetas)
                                  
        if(self.isRegularized and not self.is_normal_equation):
            self.set_lambdas()
            self.descent(self.regularized_gradient_descent)
        elif(self.is_normal_equation): 
            self.thetas = self.normal_equation()
        else:
            self.descent(self.gradient_descent)
                
    def descent(self, find_minimal):
        for i in range(self.num_iter):
            find_minimal()
   
    
      
    def set_lambdas(self):   
        self.lambdas *= np.ones(self.columns)
        self.lambdas[0] = 0
        
    def update(self):
        self.thetas = np.copy(self.tempThetas)      
           
        
    
    def gradient_descent(self):
        self.J.append(self.cost_function())  
        
        derivated_cost_j = self.partial_derivative_cost()
        descent = self.alpha*derivated_cost_j/(len(self.y))
        self.tempThetas = self.thetas - descent      
        self.update()
        
    def regularized_gradient_descent(self):
        self.J.append(self.cost_function())  
        
        derivated_cost_j = self.partial_derivative_cost()
        descent = self.alpha*derivated_cost_j/(len(self.y))
        factor = (1 - self.alpha*self.lambdas/ self.rows)
        regularization = self.thetas * factor
        
       
        self.tempThetas = regularization - descent      
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
        plt.show()
        
    def normal_equation(self):# wont work if the features are linearly dependant
        transposeX = np.transpose(self.X)
        inverse = np.dot(transposeX, self.X)
        if(self.isRegularized):
            L = np.identity(self.columns)
            L[0,0] = 0    
            regularization = self.lambdas * L
            inverse += regularization
        inverse = np.linalg.inv(inverse)
        mult = np.dot(inverse, transposeX)
        result =np.dot( mult, self.y)
        print('result: ', result)
        return result
        
        
   
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

lr = MyLinearRegr(num_iter=300, isRegularized=True)
lr.fit(X, y)

pred = lr.predict(lr.X)
    
y_test = y.reshape(-1,1)#sc_y.inverse_transform(y)

y_pred = pred.reshape(-1,1)#sc_y.inverse_transform(pred).reshape(len(y), 1)
lr.plot_cost()

print(lr.thetas)

compare(y_test, y_pred)









