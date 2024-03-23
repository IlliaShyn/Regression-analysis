import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LassoCV

wine_data = pd.read_csv("winequality.csv",delimiter=";") #read the file
wine_data.isnull().sum() #checking for 0's
correlation_matrix = wine_data.corr()
correlation_with_quality = correlation_matrix['quality'].sort_values(ascending=False)
print(correlation_with_quality)
# Plot correlations
plt.figure(figsize=(11,6))
sns.barplot(x=correlation_with_quality.index, y=correlation_with_quality.values)
plt.title("Correlation with Wine Quality")
plt.xticks(rotation=45)
plt.show()
# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix of wine quality")
plt.show()

X = wine_data.drop(columns=['quality'])
y = wine_data['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=None)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print("Linear Regression MSE:", mse_lr)
print("Linear Regression R^2:", r2_lr)
# Lasso
lasso_model = LassoCV(alphas=[0.001, 0.01, 0.1, 1.0, 10.0], cv=5)
lasso_model.fit(X_train, y_train)


y_pred_lasso = lasso_model.predict(X_test)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print("LASSO Regression MSE:", mse_lasso)
print("LASSO Regression R^2:", r2_lasso)

pred_data = pd.read_csv("pred_wine.csv",delimiter=";")
predicted_quality = lasso_model.predict(pred_data)
pred_data['predicted_quality'] = predicted_quality
pred_data.to_csv("predicted_wine.csv", index=False)

