import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from scipy import stats
from sklearn.model_selection import KFold


np.random.seed(6)
THRESHOLD = 0.5
W1 = 1
W2 = 20
W3 = 100
W4 = 0.01


def cost_function(true, predicted):
    """
        true: true values in 1D numpy array
        predicted: predicted values in 1D numpy array

        return: float
    """
    cost = (true - predicted) ** 2

    # true above threshold (case 1)
    mask = true > THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(np.logical_and(predicted < true, predicted > THRESHOLD), mask)
    mask_w3 = np.logical_and(predicted < THRESHOLD, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2
    cost[mask_w3] = cost[mask_w3] * W3

    # true value below threshold (case 2)
    mask = true <= THRESHOLD
    mask_w1 = np.logical_and(predicted > true, mask)
    mask_w2 = np.logical_and(predicted < true, mask)

    cost[mask_w1] = cost[mask_w1] * W1
    cost[mask_w2] = cost[mask_w2] * W2

    # reward for correctly identified safe regions
    reward = W4 * np.logical_and(predicted <= THRESHOLD, true <= THRESHOLD)

    return np.mean(cost) - np.mean(reward)


"""
Fill in the methods of the Model. Please do not change the given methods for the checker script to work.
You can add new methods, and make changes. The checker script performs:


    M = Model()
    M.fit_model(train_x,train_y)
    prediction = M.predict(test_x)

It uses predictions to compare to the ground truth using the cost_function above.
"""






class Model():

    def __init__(self):
        # Hyperparemeters
        self.h = 0.05   #self.h = 0.25
        self.s = 1
        pass

    def predict(self, test_x):

        y = []
        for i in range(len(test_x)):
            y.append(self.mu_pos(test_x[i]))    # Set prediction to be the posterior mean
        y = np.array(y)

        return y





    def fit_model(self, train_x, train_y):


        # IMPLEMENTATION OF THE METHOD OF INDUCED POINTS

        # Define main matrices imvolved
        self.X_A = train_x
        self.N = len(train_x)
        self.mu_A = np.zeros((self.N, 1))  # Initialize mean data vector
        self.y_A = np.array(train_y).reshape(self.N, 1)


        # Create grid of induced points
        K = 20      # K = 10
        self.M = K**2
        x1_u = np.linspace(-1, 1, K)
        x2_u = np.linspace(-1, 1, K)
        x1_u, x2_u = np.meshgrid(x1_u, x2_u)
        x1_u = x1_u.reshape(self.M, 1)
        x2_u = x2_u.reshape(self.M, 1)
        self.xu = np.hstack((x1_u, x2_u))

        # Contruct muA
        for i in range(self.N):
            self.mu_A[i, 0] = self.mu_pri(self.X_A[i])


        # Construct Kernel
        self.K_uu = np.zeros((self.M,self.M))
        for i in range(self.M):
            for j in range(self.M):
                self.K_uu[i, j] = self.k(self.xu[i],self.xu[j])

        # Build Kua and Kau
        self.K_uA = np.zeros((self.M, self.N))
        for i in range(self.M):
            for j in range(self.y_A.shape[0]):
                self.K_uA[i, j] = self.k(self.xu[i],self.X_A[j])
        self.K_Au = np.transpose(self.K_uA)

        # Helper (Kua * Kau + s^2 * Kuu)^-1
        self.I_uu = np.linalg.inv(np.matmul(self.K_uA, self.K_Au) + (self.s**2) * self.K_uu)


    # Define prior to be closer to 1 so that in the case that we are uncertain about the prediction, we converge to a value near 1
    # (this is done in order to enhance the predictions when there is a lack of data, so that we avoid the possibility of predicting a lower pollution level)
    def mu_pri(self, x):
        return 0.8


    # Kernel function
    def k(self, x1, x2):
        # return (0.316**2)*np.exp(-(np.linalg.norm(x1-x2))**2/self.h)
        return np.exp(-(np.linalg.norm(x1-x2))**2/self.h)

    # Posterior mean
    def mu_pos(self, x):
        K_xu = np.zeros((1, self.M))
        for i in range(self.M):
            K_xu[0, i] = self.k(x, self.xu[i])
        return self.mu_pri(x) + np.matmul(np.matmul(K_xu, np.matmul(self.I_uu, self.K_uA)), (self.y_A - self.mu_A))[0][0]


    def cross_validation(self, train_x, train_y, numFolds):

        print("Starting cross-validation...")

        X = train_x
        Y = train_y

        folds = KFold(n_splits=numFolds, shuffle=True)

        estimators = []
        results = np.zeros(len(X))
        score = 0.0
        n_fold = 1

        for train_index, test_index in folds.split(X):
            X_train, X_test = X[train_index, :], X[test_index, :]
            Y_train, Y_test = Y[train_index], Y[test_index]

            M_cv = Model()
            M_cv.fit_model(X_train, Y_train)

            prediction_cv = M_cv.predict(X_test)

            score_fold = cost_function(Y_test, prediction_cv)
            score += score_fold

            print("Fold n°", n_fold, " Loss: ", score_fold)
            n_fold += 1

        score /= numFolds
        print("The averaged cross-validation loss is: ", score)





def main():
    train_x_name = "train_x.csv"
    train_y_name = "train_y.csv"

    train_x = np.loadtxt(train_x_name, delimiter=',')
    train_y = np.loadtxt(train_y_name, delimiter=',')

    # load the test dateset
    test_x_name = "test_x.csv"
    test_x = np.loadtxt(test_x_name, delimiter=',')


    # Fit Model
    M = Model()

    M.fit_model(train_x, train_y)

    # M.cross_validation(train_x, train_y, 5)




    x1,x2 = test_x[:, 0], test_x[:, 1]

    # Predict data
    prediction = M.predict(np.transpose(np.array([x1,x2])))
    print(prediction)

    # PLOTTING
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    test_plot = ax.plot_trisurf(x1, x2, prediction, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.8)
    test_plot.set_clim(0, 1)

    # Customize the z axis.
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    fig.colorbar(test_plot, aspect=5)
    plt.show()


if __name__ == "__main__":
    main()
