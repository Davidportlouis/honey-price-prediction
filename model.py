import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

honey_data = pd.read_csv('./dataset/honeyproduction.csv')

print(honey_data.head())
print(honey_data.describe())

total_production = honey_data.groupby('year').totalprod.mean().reset_index()

print(total_production)

X = total_production['year']
y = total_production['totalprod'].values.reshape(-1,1)

x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.2,train_size=0.8,random_state=10)

x_train = x_train.values.reshape(-1,1)
x_test = x_test.values.reshape(-1,1)

lm = LinearRegression()
lm.fit(x_train,y_train)
y_predict_train = lm.predict(x_train)
y_predict_test = lm.predict(x_test)

print("Train data R2 Score: " + str(r2_score(y_predict_train,y_train)))
print("Test data R2 Score: " + str(r2_score(y_predict_test, y_test)))

# Sample Test 
x_future = np.array(2015)
x_future = x_future.reshape(-1,1)
y_predict_future = lm.predict(x_future)
print(f"Predicted Honey Production in {x_future}: {y_predict_future}")

#training plot
plt.scatter(x_train,y_train,alpha=0.5)
plt.plot(x_train,y_predict_train)
plt.show()

#testing plot
plt.scatter(x_test,y_test,alpha=0.5)
plt.plot(x_test,y_predict_test)
plt.show()