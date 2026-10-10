###############################################################################
# MODULES
###############################################################################
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import numpy.random as rnd
###############################################################################

################################################################################
# PARAMETERS
################################################################################
# Dimension and sample size
p=2
n=1200
# Proportion of sample from classes 0, 1, and outliers
p0 = 3/6
p1 = 2/6
pout = 1/6
# Examples of means/covariances of classes 0, 1 and outliers
mu0 = np.array([-2,-2])
mu1 = np.array([2,2])
muout = np.array([-8,-8])
Sigma_ex1 = np.eye(p)
Sigma_ex2 = np.array([[5, 0.1],
                      [1, 0.5]])
Sigma_ex3 = np.array([[0.5, 1],
                      [1, 5]])
Sigma0 = Sigma_ex1
Sigma1 = Sigma_ex1
Sigmaout = Sigma_ex1
# Regularization coefficient
lamb = 1
################################################################################

################################################################################
# DATA/LABELS GENERATION
################################################################################
# Sample sizes
n0 = int(np.floor(n*p0))
n1 = int(np.floor(n*p1))
nout = int(np.floor(n*pout))
if n0+n1+nout < n:
   n0 += int(n - (n0+n1+nout))
# Data and labels
mu0_mat = mu0.reshape((p,1))@np.ones((1,n0))
mu1_mat = mu1.reshape((p,1))@np.ones((1,n1))
x0 = np.zeros((p,n0+nout))
x0[:,0:n0] = mu0_mat + la.sqrtm(Sigma0)@rnd.randn(p,n0)
x1 = mu1_mat + la.sqrtm(Sigma1)@rnd.randn(p,n1)
if nout > 0:
  muout_mat = muout.reshape((p,1))@np.ones((1,nout))
  x0[:,n0:n0+nout] = muout_mat + la.sqrtm(Sigmaout)@rnd.randn(p,nout)
y = np.concatenate((-np.ones(n0+nout),np.ones(n1)))
X = np.ones((n,p+1))
for i in np.arange(n):
     X[0:n0+nout,1:p+1] = x0.T
     X[n0+nout:n,1:p+1] = x1.T
################################################################################

################################################################################
# PLOTS
################################################################################
fig,ax = plt.subplots()
ax.plot(x0[0,:],x0[1,:],'xb',label='Class 0')
ax.plot(x1[0,:],x1[1,:],'xr',label="Class 1")
ax.legend(loc = "upper left")


# CLASSIFIERS: OLS / RIDGE / LOGISTIC


# ols
theta_ols = la.inv(X.T @ X) @ (X.T @ y)

# Regression Ridge
I = np.eye(p+1)
theta_ridge = la.inv(X.T @ X + n*lamb*I) @ (X.T @ y)

# Regression logistique
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(penalty='l2', C=1/(lamb+1e-8))
logreg.fit(X[:,1:], y)

theta_log = np.zeros(p+1)
theta_log[0] = logreg.intercept_[0]
theta_log[1:] = logreg.coef_[0]


# hyperplans de ridge ols et la regression logistique


xx = np.linspace(-12, 6, 400)

def hyperplane(theta):
    return -(theta[0] + theta[1]*xx) / theta[2]

ax.plot(xx, hyperplane(theta_ols), 'k-',  linewidth=2, label='OLS')
ax.plot(xx, hyperplane(theta_ridge), 'g--', linewidth=2, label='Ridge')
ax.plot(xx, hyperplane(theta_log), 'm-.', linewidth=2, label='Logistic')

ax.legend()
ax.set_title(
    f"n0={n0}, n1={n1}, nout={nout}, lambda={lamb}"
)

plt.show()
