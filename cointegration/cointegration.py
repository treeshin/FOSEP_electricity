import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

df1 = pd.read_csv('data.csv',
                    sep = ",",
                    header = 0,
                    engine = 'python')

df1.columns = ['manufacture','total','gdp']
#df = np.array(df)


delman=np.zeros([len(df1.manufacture)-4,1])
delgdp=np.zeros([len(df1.manufacture)-4,1])

for i in range(len(delman)-4):
    delman[i]=(df1.manufacture[i+4]-df1.manufacture[i])/df1.manufacture[i]
    delgdp[i]=(df1.gdp[i+4]-df1.gdp[i])/df1.gdp[i]

delman = pd.DataFrame(delman.reshape(len(delman),1))
delgdp = pd.DataFrame(delgdp.reshape(len(delgdp),1))

df2 = pd.concat([delman, delgdp], axis=1)
df2.columns = ['delman', 'delgdp']

plt.scatter(df1.manufacture, df1.gdp)
plt.xlabel('man')
plt.ylabel('gdp')
#plt.show()

plt.scatter(df2.delman, df2.delgdp)
plt.xlabel('delman')
plt.ylabel('delgdp')
#plt.show()

df2.delman.plot(figsize=(8,4))
df2.delgdp.plot(figsize=(8,4))
#plt.show()

# Regression - Ordinary Least Squares
model = sm.OLS(df1.manufacture, df1.gdp)
model = model.fit()
print(model.params[0])

# Spread
df1['spread'] = df1.manufacture - model.params[0] * df1.gdp
# Plot the spread
df1.spread.plot(figsize=(8,4))
plt.ylabel("Spread")
#plt.show()

# Compute ADF test statistics
adf = adfuller(df1.spread, maxlag = 1)
print(adf[0])
print(adf[4])

# Regression - Ordinary Least Squares
model = sm.OLS(df2.delman, df2.delgdp)
model = model.fit()
print(model.params[0])

# Spread
df2['spread'] = df2.delman - model.params[0] * df2.delgdp
# Plot the spread
df2.spread.plot(figsize=(8,4))
plt.ylabel("Spread")
#plt.show()

# Compute ADF test statistics
adf = adfuller(df2.spread, maxlag = 1)
print(adf[0])
print(adf[4])
