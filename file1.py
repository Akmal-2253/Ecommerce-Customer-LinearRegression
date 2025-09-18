import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv('Ecommerce Customers')
df=pd.DataFrame(dataset)
# print(df.head())


# sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=df,alpha=0.5)
# # plt.show()


# sns.pairplot(df,kind='scatter',plot_kws={'alpha':0.5})
# # plt.show()


# sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=df)
# plt.show()


x = dataset[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y=dataset['Yearly Amount Spent']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
   x , y, test_size=0.30, random_state=42
)
# print(X_train)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=(scaler.fit_transform(X_train))
X_test=scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
model = LinearRegression()
model.fit(X_train, y_train)
mse = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)

#Prediction
reg_pred=model.predict(X_test)
print(reg_pred)

# comparing perdiction with actual value
sns.kdeplot(y_test-reg_pred)
plt.show()


from sklearn.metrics import r2_score
score=r2_score(y_test,reg_pred)
print(score)





