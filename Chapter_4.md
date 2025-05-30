---
marp: true
theme: gaia
title: Anatomy of a learning algorithm
math: mathjax

---

# Learning Algorithms

## Key Components : -
- **Loss Function**: measures how far predictions are from actual values
- **Optimization Criterion** (Cost function): based on loss function
- **Opimization Routine** : uses training data to find solution to the optimization criterion.

Some models came first, optimization came later (Decision Trees, kNN). But now, most models are built with optimization at the core (eg. Logistic regression, SVM, Neural nets)

---

## Optimization Algorithms
- used for minimizing the cost function

Two most common algorithms that we use when the optimization criterion is differentiable are:

- **Gradient Descent (GD)**

- **Stochastic Gradient Descent (SGD)**

---

# Gradient Descent

- Iteratively improves parameters to reduce loss.
- Works by taking steps proportional to the negative of the gradient of the function at the current point.
- Useful when:
  - The cost function is smooth/differentiable.
  - The function is convex (single global minimum).

---

## How it works

1. **Start with initial parameters** (random or zero).
2. **Compute the gradient** of the loss with respect to parameters.
3. **Update parameters** in the direction opposite to the gradient:
   $$
   \theta = \theta - \alpha \cdot \nabla_\theta L
   $$
   where:
   - $\theta$ are the parameters
   - $\alpha$ is the learning rate
   - $\nabla_\theta L$ is the gradient of the loss w.r.t $\theta$

---

4. **Repeated** until loss stops decreasing.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![h:500](/Images/Gradient%20descent.png)

---

## Gradient Descent in Linear Regression

- Linear model:  $y = wx + b$

- We want to find **$w$** and **$b$** such that:

    $$
    l = \frac{1}{N} \sum_{i=1}^{N} (y_i - (w x_i + b))^2
    $$

    is minimum. 

- We compute gradients and update:

    $$
    w = w - \alpha \cdot \frac{\partial l}{\partial w} 
    $$

---

$$
b = b - \alpha \cdot \frac{\partial l}{\partial b}
$$

Partial derivatives:

$$
\frac{\partial l}{\partial w} = \frac{1}{N} \sum_{i=1}^N-2x_i(y_i - (wx_i + b))
$$
$$
\frac{\partial l}{\partial b} = \frac{1}{N} \sum_{i=1}^N -2(y_i - (wx_i + b))
$$


---

### Python Code: Gradient Computation

```python
def compute_gradients(x, y, w, b):
    dl_dw, dl_db = 0, 0
    n = len(x)
    for i in range(n):
        error = y[i] - (w * x[i] + b)
        dl_dw += -2 * x[i] * error
        dl_db += -2 * error
    return dl_dw/n , dl_db/n 

def update_parameters(w, b, dl_dw, dl_db, learning_rate):
    w -= learning_rate * dl_dw
    b -= learning_rate * dl_db
    return w, b

def compute_avg_loss(x, y, w, b):
    n = len(x)
    total_loss = 0
    for i in range(n):
        prediction = w * x[i] + b
        total_loss += (y[i] - prediction) ** 2
    return total_loss / n
```

---


```python
def train(x, y, w, b, learning_rate, epochs):
    for epo in range(epochs):
        dl_dw, dl_db = compute_gradients(x, y, w, b)
        w, b = update_parameters(w, b, dl_dw, dl_db, learning_rate)
        if epo % 400 == 0:
            loss = compute_avg_loss(x, y, w, b)
            print(f"Epoch {epo}, Loss: {loss}")
    return w, b

def predict(x, w, b):
    return w * x + b

```

---

- **Epoch** : one full pass through the entire training dataset.
- In each epoch:
  - Compute predictions for all data points.
  - Calculate loss.
  - Update parameters using gradient.
- Usually, multiple epochs are needed to reach convergence.

---

## Stochastic gradient descent (SGD)

- A varient of gradient descent which speeds up learning by using **subsets of data** (not the whole dataset).

It has many versions:

- **Adagrad**:  
  - Adapts learning rate ($\alpha$) for each parameter  
  - Large past gradients ⟶ smaller learning rate  
  - Good for sparse data  

---

- **Momentum**:  
  - Remembers previous gradients  
  - Accelerates in relevant direction  
  - Reduces oscillations

- **RMSprop and Adam**: more frequently used in neural network training


---

# How Machine Learning Engineers Work

- They usually **don’t implement** learning algorithms or solvers like gradient descent.
- Directly use **open-source libraries** like `scikit-learn`, `TensorFlow`, `PyTorch`, etc.
- These libraries provide **efficient** and **well-tested implementations**.

---

**Example: Linear Regression in `scikit-learn`**

```python
from sklearn.linear_model import LinearRegression

def train(x, y):
    model = LinearRegression().fit(x, y)
    return model

model = train(x, y)
x_new = [[21]]
y_new = model.predict(x_new)
print(y_new)
```

(similar to the code in previous chapter)