import numpy as np, pandas as pd, matplotlib.pyplot as plt

class Thetas:
    def __init__(self):
        self.thetas = np.array([0.0001, 0.0006, 0.0009])
        self.tempThetas = np.copy(self.thetas)
        self.alpha = 0.08
        self.J = []
        
    def calculate(self, X, y):
        j = cost(X, self.thetas, y)
        self.J.append(self.j_of_theta(X,y ) )       
        custo = self.alpha*j/(len(y))
        self.tempThetas = self.thetas - custo      
        self.update()
        
    def j_of_theta(self, X, y):
        sum_of_quads = np.square(h(X, self.thetas) -y).sum()/(2/len(y))
        return sum_of_quads
        
        
    def update(self):
        self.thetas = np.copy(self.tempThetas)      
   
class MyLinearRegr:
    
    def fit(self, X, y):
        self.X = add_ones_row(X)
    
        for i in range(300):
        thetas.calculate(X, y)
        
        

    
    def add_ones_row(X):
        ones = np.ones(len(X[:,0])).reshape(-1,1)
        return np.concatenate((ones, X),1)
        
    
    def h(X, thetas):
        return  np.dot(X, thetas)
    
    def cost(X, thetas, y):
        aga = h(X, thetas) - y    
        sumOfElements =  np.dot(aga, X) 
        result = sumOfElements
        return result


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

X = add_ones_row(X)
thetas = Thetas()







plt.plot(thetas.J)
plt.show()

print('ultimo: ', thetas.thetas)
print('x: ', X[0])

pred = h(X, thetas.thetas)
    
y_test = y.reshape(-1,1)#sc_y.inverse_transform(y)

y_pred = pred.reshape(-1,1)#sc_y.inverse_transform(pred).reshape(len(y), 1)



# =============================================================================
# print(np.concatenate((y_test, y_pred), 1))
# 
# 
# 
# print(thetas.thetas[0], sc_x.transform([thetas.thetas[1:]]))
# =============================================================================










