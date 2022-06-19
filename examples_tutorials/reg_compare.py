from emukit.model_wrappers import GPyModelWrapper
import GPy
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

def f(x):
    return np.sin(2 * np.pi * x)

n = 100
x = np.random.rand(n, 1)
y = f(x) + np.random.randn(n, 1) / 5

plt.figure(num='Data')
plt.scatter(x, y)
plt.tight_layout()

x_plot = np.linspace(0, 1, 200)[:, None]
y_exact_plot = f(x_plot)

plt.figure(num='Data + exact')
plt.scatter(x, y)
plt.plot(x_plot, y_exact_plot, 'r--')
plt.tight_layout()

reg = LinearRegression()
reg.fit(x, y)

y_pred_plot = reg.predict(x_plot)

plt.figure(num='Data + exact + lr')
plt.scatter(x, y)
plt.plot(x_plot, y_exact_plot, 'r--')
plt.plot(x_plot, y_pred_plot, 'r')
plt.tight_layout()

model = make_pipeline(PolynomialFeatures(3), LinearRegression())
model.fit(x, y)
y_pol_pred_plot = model.predict(x_plot)

plt.figure(num='Data + exact + pr')
plt.scatter(x, y)
plt.plot(x_plot, y_exact_plot, 'r--')
plt.plot(x_plot, y_pol_pred_plot, 'r')
plt.tight_layout()

gpy_model = GPy.models.GPRegression(x, y, GPy.kern.RBF(1))
# gpy_model.Gaussian_noise.variance.fix(0)
gpy_model.optimize()
emukit_model = GPyModelWrapper(gpy_model)
mu_plot, var_plot = emukit_model.predict(x_plot)

plt.figure(num='Data + exact + gp')
plt.scatter(x, y)
plt.plot(x_plot, y_exact_plot, 'r--')
plt.plot(x_plot, mu_plot, 'r')
plt.fill_between(x_plot.flatten(),
                 (mu_plot - 2 * np.sqrt(np.abs(var_plot))).flatten(),
                 (mu_plot + 2 * np.sqrt(np.abs(var_plot))).flatten(),
                 alpha=.25, color='r')
plt.tight_layout()


plt.show()