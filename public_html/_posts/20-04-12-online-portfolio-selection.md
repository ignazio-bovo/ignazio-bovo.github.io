---
layout: post
title: Online portfolio selection
---
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

```python
import pandas as pd 
import numpy as np
import yfinance as yf
pd.options.display.precision = 4
```


```python
df = yf.download('IBM KO AAPL IR AA PG SLB',start = '2001-01-01',end='2020-01-01')['Adj Close']
```

    [*********************100%***********************]  7 of 7 completed



```python
# pandas .div() does round to 1.0
n,m = df.shape
X = df.iloc[1:,:].div(df.iloc[:-1,:])
for j in range(m):
    for i in range(1,n):
        X.iloc[i,j] = df.iloc[i,j] / df.iloc[i-1,j]
X.dropna(inplace=True)
```

See this paper: best survey on online machine learning for trading so far
[Online Portfolio Selection: A Survey](https://doi.org/10.1145/2512962)

## Buy and hold
### Uniform buy and hold
split the entire capital uniformly across all assets, never 
rebalance. Final wealth is:


```python
b = 1/m * np.ones(m)
np.sum(df.iloc[-1,:].div(df.iloc[0,:]).mul(b))
```




    49.2722347400404



### Best stock strategy 
Invest everything on best performing stock. Final wealth is 
the relative growth of the best performing stock:



```python
max_growth = df.iloc[-1,:].div(df.iloc[0,:])
idx = np.argmax(max_growth)
b = np.zeros(m)
b[idx] = 1
max_growth[idx]
```




    317.6371913379486



## Constant Rebalanced Portfolios 
Rebalances the portfolio to $b$ at every trading period 
$$b_t = \{ b ,b ,b ,\dots,b \}$$ the wealth is 
$$S_n = \prod_{t=1}^n b \cdot x_t$$
$$b$$ is chosen as the strategy 
that maximizes the growth rate:

$$ n W_n = \sum_{t=1}^n \log(b^T x_t) $$.

Such $$ b^* $$ is called the best CRP:

$$ \text{maximize} \sum_{t=1}^n \log(b^T x_t) $$ 

such that 

$$ \sum_{i=1}^m b_i = 1 \quad b_i \geq 0 $$


```python
import cvxpy as cp 
n,m = X.shape
x = X.values
b = cp.Variable(m)
cost = 0
for t in range(n):
    cost = cost + cp.log(b @ x[t,:])
problem = cp.Problem(cp.Maximize(cost), [cp.sum(b) == 1, b >= 0])
problem.solve()
```




    5.801321652343832




```python
# strategy b
np.round(b.value,3)
```




    array([-0.   ,  0.943, -0.   ,  0.057, -0.   , -0.   , -0.   ])



### Exponential gradient 
$$ b_{t+1} $$ is the solution to the problem: 
$$
\text{maximize}_b \,\, \eta \log(b \cdot x_t) - R(b,b_t)
$$
such that

$$ 
\sum_{i = 1}^m b_i = 1 ,\quad b_i \geq 0
$$
use linear approx of $\log(b^T x_t)$ about $\log(b_t^T x_t)$.
Coupled with the relative entropy regularization 
$$
R(b,b_t) = \sum_{i = 1}^m b_i \, \log \bigg(\frac{b_i}{b_{t,i}}\bigg)
$$
gives the update rule:
$$ 
b_{t+1,i} = b_{t,i} \exp \bigg( \eta \frac{x_{t,i}}{b_t \cdot x_t} \bigg) / Z
$$

$Z$ is normalization constant s.t. portfolio weight sums up to $1$


```python
x = X.values
eta = .01
n,m = x.shape
bb = np.zeros((n,m))
bb[0,:] = 1/m * np.ones(m)
for t in range(1,n):
    growth = np.dot(bb[t-1,:], x[t-1,:])
    bb[t,:] = bb[t-1,:] * np.exp(eta*x[t-1,:]/growth)
    bb[t,:] = bb[t,:]/sum(bb[t,:])

```


```python
S = 1
for t in range(n):
    S = S * np.dot(bb[t,:],x[t,:])
S
```




    12.223726572994318




```python
import matplotlib.pyplot as plt 
B = pd.DataFrame(bb)
B.columns = df.columns
ax = B.plot(figsize=(16,9))
ax.set_title('weights')
ax.set_ylabel('%')
ax.set_xlabel('time')
```




    Text(0.5, 0, 'time')



![png]({{ "/assets/images/online_strategies_15_1.png" }})


with regularization 

$$ R(b,b_t) = \frac{1}{2} \sum_{i=1}^m (b_{t,i} - b_i)^2 $$

the update rule becomes 

$$ b_{t+1,i} = b_{t,i} + \eta \bigg( \frac{x_{t,i}}{b_t \cdot x_t} - \frac{1}{m} \sum_{j=1}^m \frac{x_{t,j}}{b_t \cdot x_t}  \bigg)
$$

called gradient projection rule (GP)


```python
eta = .01
x = X.values 
n,m = x.shape 
bb = np.zeros((n,m))
bb[0,:] = 1/m * np.ones(m)
for t in range(1,n):
    growth = np.dot(x[t-1,:],bb[t-1,:])
    coeff = eta / growth
    bb[t,:] = bb[t-1,:] + coeff * (x[t-1,:] - np.mean(x[t-1,:]) * np.ones(m))
```

```python
S = 1
for t in range(n):
    S = S * np.dot(bb[t,:],x[t,:])
S
```




    13.310014683307498




```python
B = pd.DataFrame(bb)
B.columns = df.columns
ax = B.plot(figsize=(16,9))
ax.set_title('weights')
ax.set_ylabel('%')
ax.set_xlabel('time')
```




    Text(0.5, 0, 'time')




![png]({{ "/assets/images/online_strategies_19_1.png" }})


with regularization 

$$
 R(b,b_t) = \frac{1}{2} \sum_{i=1}^m \frac{(b_{t,i} - b_i)^2}{b_{t,i}} 
$$

the update rule becomes 

$$ 
b_{t+1,i} = b_{t,i} \bigg( \eta \bigg( \frac{x_{t,i}}{b_t \cdot x_t} - 1 \bigg) + 1 \bigg)
$$

called Expectation Maximization (EM)


```python
eta = .01
x = X.values 
n,m = x.shape 
bb = np.zeros((n,m))
bb[0,:] = 1/m * np.ones(m)
for t in range(1,n):
    growth = np.dot(x[t-1,:],bb[t-1,:])
    bb[t,:] = bb[t-1,:] * (eta * ( x[t-1,:]/growth  - 1) +1) 
```


```python
S = 1
for t in range(n):
    S = S * np.dot(bb[t,:],x[t,:])
S
```




    12.223540162913903




```python
B = pd.DataFrame(bb)
B.columns = df.columns
ax = B.plot(figsize=(16,9))
ax.set_title('weights')
ax.set_ylabel('%')
ax.set_xlabel('time')
```




    Text(0.5, 0, 'time')




![png]({{ "/assets/images/online_strategies_23_1.png" }})


## Follow the leader 

$$
 b_{t+1} = \text{argmax}_b \, \sum_{j = 1}^t \log(b \cdot x_t) 
$$

such that 

$$ \sum_{i=1}^m b_i = 1 ,\quad b_i \geq 0 $$


```python
import cvxpy as cp
eta = .01
x = X.values 
n,m = x.shape 
bb = np.zeros((n,m))
bb[0,:] = 1/m * np.ones(m)
for t in range(1,n):
    b = cp.Variable(m)
    cost = 0
    for t in range(n):
        cost = cost + cp.log(b @ x[t,:])
    problem = cp.Problem(cp.Maximize(cost), [cp.sum(b) == 1, b >= 0])
    problem.solve(solver=cp.SCS, max_iters=100)
    bb[t,:] = b.value
S = 1
for t in range(n):
    S = S * np.dot(bb[t,:],x[t,:])
S
```


```python
import cvxpy as cp
print(cp.installed_solvers())
```

    ['ECOS', 'ECOS_BB', 'OSQP', 'SCS']



```python

```
