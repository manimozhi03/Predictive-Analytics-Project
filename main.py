import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_excel("Sales_Data.xlsx")

print(data)

data["Date"]=pd.to_datetime(data["Date"])
data["Month"]=data["Date"].dt.month

X=data[["Month"]]
y=data["Sales"]

model=LinearRegression()
model.fit(X,y)

predictions=model.predict(X)

print("Model Accuracy:",model.score(X,y))

plt.plot(data["Month"],y,marker="o",label="Actual Sales")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.title("Predictive Analytics ")
plt.legend()
plt.show()