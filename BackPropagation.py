import numpy as np

# data points
X = np.array([[1., 2.], [3., -3.], [-2., 2.], [2., 5.], [4., -1.]])
y_true = np.array([-1.4, 24.2, -16.2, -5.1, 22.9])
n_points = X.shape[0]

# set update rate
alpha = 0.1

# initialize our guess for w
w = np.array([3., -5.])

# iterate over the data set 20 times
for iter in range(20):

    # iterate over each data point
    for p in range(n_points):
        x = X[p]
        yn = y_true[p]

        # forward pass
        y = w*x
        print(y)
        # backward pass
        #dSdy = ()
       # dydw = # student code goes here
       # dSdw = dSdy * dydw

        # update w
       # w = w - alpha * dSdw