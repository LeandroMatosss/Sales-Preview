import pandas as pd

tabela = pd.read_csv("advertising.csv")
print(tabela)

import seaborn as sns
import matplotlib.pyplot as plt

sns.pairplot(tabela)
plt.show()
sns.heatmap(tabela.corr(), cmap="Wistia", annot=True)
plt.show()

from sklearn.model_selection import train_test_split

x= tabela.drop('Vendas', axis=1)
y= tabela['Vendas']
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.3,random_state=1)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import numpy as np

#Treino AI
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train)

rf_reg = RandomForestRegressor()
rf_reg.fit(x_train,y_train)

# teste AI
test_pred_lin = lin_reg.predict(x_test)
test_pred_rf = rf_reg.predict(x_test)

r2_lin = metrics.r2_score(y_test, test_pred_lin)
mse_lin = metrics.mean_squared_error(y_test, test_pred_lin)
print(f"R² da Regressão Linear: {r2_lin}")
print(f"MSE da Regressão Linear: {mse_lin}")
r2_rf= metrics.r2_score(y_test, test_pred_rf)
mse_rf = metrics.mean_squared_error(y_test, test_pred_rf)
print(f"R² do Random Forest: {r2_rf}")
print(f"MSE do Random Forest: {mse_rf}")

tabela_resultado = pd.DataFrame()
tabela_resultado['y_teste'] = y_test
tabela_resultado['y_previsao_rf'] = test_pred_rf

tabela_resultado = tabela_resultado.reset_index(drop=True)
plt.figure(figsize=(15, 5))
sns.lineplot(data=tabela_resultado)
plt.show()
print(tabela_resultado)

# importancia_features = pd.DataFrame(rf_reg.feature_importances_, x_train.columns)
plt.figure(figsize=(15, 5))
sns.barplot(x=x_train.columns, y=rf_reg.feature_importances_)
plt.show()

print(tabela[["Radio", "Jornal"]].sum())