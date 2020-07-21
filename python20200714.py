import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

train_size = 20
test_size = 12
train_X = np.random.uniform(low=0, high=1.2, size=train_size)
train_X = np.random.uniform(low=0.1, high=1.3, size=test_size)
train_X = np.sin(train_X * 2 * np.pi) + np.random.normal(0, 0.2, train_size)
train_X = np.sin(test_X * 2 * np.pi) + np.random.normal(0, 0.2, test_size)

poly = PolynomialFeatures(6) #次数は6
train_poly_X = poly.fit_transform(train_X.reshape(train_size, 1))
test_poly_X = poly.fit_transform(test_X.reshape(train_size, 1))

model = Ridge(alpha=1.0)
model.fit(train_poly, train_y)
train_pred_y = model.prdict(train_poly_X)
test_pred_y = model.predict(test_poly_X)

print(mean_squared_error(train_pred_y, train_y))
print(mean_squared_error(test_pred_y, test_y))