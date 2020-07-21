from sklearn.linear_model import LinearRegression

X = [[10.0],[8.0],[13.0],[9.0],[11.0],[14.0],[6.0],[4.0],[12.0],[7.0],[5.0]]
y = [8.04,6.95,7.58,8.81,8.33,9.96,7.24,4.26,10.84,4.82,5.68]

model = LinearRegression()
model.fit(X,y)
print(model.intercept_) # 切片
print(model.coef_) #　傾き
y_pred = model.predict([[0],[1]])
print(y_pred) # x=0,x=1に対する予測結果