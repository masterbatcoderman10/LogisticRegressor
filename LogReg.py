class LogReg:

    def __init__(self, y, X, eta, epochs, momentum=False, alpha=0.0, irls=False):
        self.w = np.random.normal(0.0, 1.0, (X.shape[1],)) 
        self.y = y
        self.X = X
        self.eta = eta
        self.epochs = epochs
        self.momentum = momentum
        #Iterative Re-weighted least squares
        self.irls = irls
        if momentum:
            self.velocity = 0.0
            #Parameter to update velocity
            self.alpha = alpha


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-1 * x))
    
    
    def predict(self, x):
        preds = np.dot(self.w, x)
        return preds
    
    def pred_prob(self, x):
        return self.sigmoid(self.predict(x))

    def full_prob(self, X):

        return self.sigmoid(X @ self.w)
    
    def full_cost(self):

        cost = 0
        for x_i, y_i in zip(self.X, self.y):
            cost += self.calc_cost(x_i, y_i)
        
        return cost / len(self.X)

    
    def calc_cost(self, x, y):

        if y == 1:
            return -1 * np.log(self.sigmoid(self.predict(x)))
        else:
            return -1 * np.log(1 - self.sigmoid(self.predict(x)))
    
    def full_grad(self, lambd=0.0):
        
        #lambda for regularization
        P1 = self.full_prob(self.X)
        return self.X.T @ (P1 - self.y) + (lambd * self.w)

    def calc_hessian(self, lambd=0.0):

        P1 = self.full_prob(self.X)
        diagonals = P1 * (1 - P1)

        n = len(P1)
        R = np.zeros((n,n))
        np.fill_diagonal(R, diagonals)
        
        H = (self.X.T @ R @ self.X) + lambd

        return H
    
    def single_grad(self, x, y, l2=False, lamb=0.0):

        if l2:
            if y == 1:
                return -1 * x * (1 - self.sigmoid(self.predict(x))) + lamb * self.w
            else:
                return x * self.sigmoid(self.predict(x)) + lamb * self.w


        #Gives gradient for single input variable
        if y == 1:
            return -1 * x * (1 - self.sigmoid(self.predict(x)))
        else:
            return x * self.sigmoid(self.predict(x))
    
    def update_weights(self, grad=0, hessian=0):

        if self.momentum:
            self.w = self.w + self.velocity
        elif self.irls:
            self.w = self.w - (hessian * grad)
        else:
            self.w = self.w - (self.eta * grad)

    
    def shuffle_together(self, x, y):
        assert len(x) == len(y)
        p = np.random.permutation(len(x))
        return x[p], y[p]
    
    def gen_batches(self, x, y, batch_size):

        batches = []
        start = 0
        end = batch_size

        while end < len(x):
            batches.append((x[start:end], y[start:end]))
            start = end
            end += batch_size

        return np.array(batches)
    
    def train_gd_eff(self, lambd=0.0):

        n = len(self.X)
        cost = self.full_cost()
        print(f"The loss before training : {cost}")
        for e in range(self.epochs):
            
            grad = self.full_grad(lambd)
            self.update_weights(grad)
        
        cost = self.full_cost()
        print(f"The loss after training : {cost}")

    def train_irls(self, lambd=0.0):

        cost = self.full_cost()
        print(f"The loss before training : {cost}")
        for e in range(self.epochs):
            
            #Calculate the Hessian matrix
            hessian = self.calc_hessian(lambd)
            #Invert the Hessian matrix
            hessian = np.linalg.inv(hessian)

            grad = self.full_grad()
            #Passing in the Hessian since learning rate won't be used
            self.update_weights(grad, hessian)
        
        cost = self.full_cost()
        print(f"The loss after training : {cost}")


    
    def train_sgd(self, l2=False, l=0):
        n = len(self.X)
        for e in range(self.epochs):
            epoch_loss = 0
            for i in range(n):

                i = np.random.randint(0, n)
                x_i = self.X[i]
                y_i = self.y[i]

                loss = self.calc_cost(x_i, y_i)
                grad = self.single_grad(x_i, y_i, l2, l)

                epoch_loss += loss
                # print(f"The observation : {x_i}")
                # print(f"The target : {y_i}")
                # print(f"Previous weights : {self.w}")
                # print(f"The dot product of weight with observation : {self.predict(x_i)}")
                # print(f"Sigmoid of prediction : {self.pred_prob(x_i)}")
                self.update_weights(grad)
                # print(f"Gradient update at epoch {e} : {grad}")
            epoch_loss /= n
            print(f"Loss at epoch {e} : {epoch_loss}")
                # print(f"Weight after update : {self.w}")
                # print("\n")
    def calc_velocity(self, grad):
        
        self.velocity = (self.alpha * self.velocity) - (self.eta * grad)

    def train_gd(self, l2=False, l=0):
        n = len(self.X)
        for e in range(self.epochs):
            total_loss = 0
            avg_grad = np.zeros(self.w.shape)
            #print(f"Previous weights at epoch {e} : {self.w}")
            for i in range(0, n):
                x_i = self.X[i]
                y_i = self.y[i]
                total_loss += self.calc_cost(x_i, y_i)
                avg_grad = avg_grad + self.single_grad(x_i, y_i, l2, l)

            print(f"Loss at epoch {e} : {total_loss / n}")
            print(f"The gradient is : {avg_grad}")
            #print(f"Gradients : {avg_grad}")
            #avg_grad = avg_grad / n

            if self.momentum:
                self.calc_velocity(avg_grad)
                self.update_weights()
            else:
                self.update_weights(avg_grad)
            #print(f"Weight after update at epoch {e} : {self.w}")
            #print("\n")
    def compare(self):
        self.w = np.zeros(6,)
        n = len(self.X)
        for e in range(self.epochs):
            total_loss = 0
            avg_grad = np.zeros(self.w.shape)
            #print(f"Previous weights at epoch {e} : {self.w}")
            for i in range(0, n):
                x_i = self.X[i]
                y_i = self.y[i]
                total_loss += self.calc_cost(x_i, y_i)
                avg_grad = avg_grad + self.single_grad(x_i, y_i)

            #print(f"Loss at epoch {e} : {total_loss / n}")
            #print(f"The gradient is : {avg_grad}")
            #print(f"Gradients : {avg_grad}")
            #avg_grad = avg_grad / n

            if self.momentum:
                self.calc_velocity(avg_grad)
                self.update_weights()
            else:
                self.update_weights(avg_grad)
            print(f"Weight after update at epoch {e} : {self.w}")
            #print("\n")
        print("\n")
        #Comparing with the efficient version 
        self.w = np.zeros(6,)


        for e in range(self.epochs):
            
            total_cost = self.full_cost() / n
            #print(f"Loss at epoch {e} : {total_cost}")
            grad = self.full_grad()
            #print(f"The gradient : {grad}")
            self.update_weights(grad)
            print(f"Weight after update at epoch {e} : {self.w}")



            
        
