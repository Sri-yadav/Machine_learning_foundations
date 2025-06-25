---
marp: true
theme: gaia
title: Notations and Definations
math: mathjax

---
# Few Mathematical Notations

- **Scaler** - simple numerical values eg. 4,5,-10.

- **Vectors** - ordered list of scalar values (attributes)

- **Sets** - unordered collection of unique elements.

**Some basic codes related to it :-**
```python
a=4 # Scaler
b=-10 #scaler

import numpy as np 
w= np.array([3,4,5,6]) # 1-D array or Vectors

```
---  

```python
x = np.array([[2,3],[4,5]]) # 2-D array or Matrix
# Similarly there can be n-dimentional array

m= {1,3,5,7} #Set
```
**Visual Representation of 2-D vector as directions and points**

```python
import numpy as np
import matplotlib.pyplot as plt
m=np.array([[2,3],[1,4],[7,4]]) # a list of vector / a matrix / a 2D array
x= m[:,0] #First column (x-component)
y= m[:,1] #Second column (y-component)
x_org=np.zeros(m.shape[0]) # x-component origin of vector
y_org=np.zeros(m.shape[0]) # y-component origin of vector
colours=['red','green','yellow']
plt.quiver(x_org, y_org, x,y, angles='xy', scale_units='xy', scale=1, color=colours)
plt.xlim(0,8)
plt.ylim(0,8)
plt.show()
```

---

- **Capital Sigma Notation**  
- **Capital Pi Notation** 

- **Operations on vectors (shown with basic python codes) : -**

```python
import numpy as np
a= np.array([2,4]) #vector
b= np.array([1,3]) #vector
A= np.array([[1,2],[3,4]]) #matrix
print(a+b) #addition
print(a-b) #subtraction
print(np.dot(a,b)) #dot-product
print(A.dot(a)) #matrix-vector multiplication
print(A.T) #Transpose of a matrix
```

---

- **Functions** 
    - minima and maxima
        ```python
        import numpy as np
        import matplotlib.pyplot as plt

        def f(x) :     #defining a function (an example)
            return x**2 -4*x +2

        x= np.linspace(-5,5,20)
        y= f(x)
        x_min_index= np.argmin(y) # gives index of x-value that corresponds to min-f(x)
        x_min= x[x_min_index] # value of that x
        print(x_min)
        plt.plot(x,y) #its graphical representation
        plt.show()

        ```
---

- **Max and Arg Max**

- **Derivative & Gradient**

```python
import sympy as sp
x1, x2= sp.symbols('x1 x2')
f1= x1**2 -4*x1 +2 # defining a function
f1_diff= sp.diff(f1,x1) #derivative 
print(f1_diff)

f2= 4*x1**2 + 2*x2 +2 #defining another function with multiple variable
f2_grad= [sp.diff(f2,vr) for vr in (x1 , x2)] #gradient
print(f2_grad)

```

---

# Random Variable
A variable whose value is decided by a random outcome. 

**Two types : -**
- **Discrete random variable** 
     - takes only countable number of distinct values.
     - its probability distribution given by- pmf (probability mass function) which is basically list of probablities of each possible value
    - probability of each value $\ge$ 0, 
    sum of probabilities of each value $=1$

---

- **Continuous random variable**
    - takes infinite (uncountable) number of possible value
    - probability distribution given by- pdf (probability density function)
    - Codomain of pdf is non-negative & area under the curve = 1

---

## Expectation 

Weighted average of all possible values, weighted by their probablities. Also called **mean , average** or **expected value**. Often represented by $\mu$ .

- **For discrete random variable**
 Let X be a random variable , possible values - $x_1, x_2, x_3 . . . . . x_n$ and let their corresponding probablities be $p_1,p_2,p_3, . . . . p_n$ 

    $$
    E[x]=  \sum_{i=1}^{n} x_i p_i
    $$  

---

- **For continuous random variable**

    $$
    f_x - \text{pdf of random variable X}
    $$
    $$
    E[X] = \int_R x f_x(x) dx
    $$  

    For a pdf,  $\int_R f_x(x) dx = 1$
<br>
    - Most of the time we don't know $f_x$ .

---

## Standard deviation

- Measure of how much the value of a variable deviate from its mean (expected value)
- Quantifies the  spread of the distribution

Given by,  $\sigma$ = $\sqrt{E(X- \mu )^2}$ 

- For discrete random variable
    $\sigma$ = $\sqrt{\sum_{i=1}^n Pr(X=x_i) (x_i- \mu )^2}$ 

- For continuous random variable
    $\sigma$ = $\sqrt{\int_R (x-\mu)^2 . f(x)dx}$ 

---

# Unbiased estimators 

- an estimator is a rule or function that we apply to sample data to estimate an unknown parameter

- an estimator whose expected value is equal to the true value of the parameter it is expecting.

- $\hat{\theta} (S_X)$ is unbiased estimator if   $E[\hat{\theta} (S_X)] =\theta$
where $\hat{\theta}$ - sample estimator ;  $S_X$ - sample data ; $\theta$ -  true value 

- Unbiased estimator of an unknown $E[X]$ is given by $\frac{1}{N} \sum_{i=1}^N x_i$ (given the outcomes are independent and identically distributed) and is called as Sample Mean.

---

# Bayes' Rule

- **$Pr(X=x|Y=y) = \frac{Pr(Y=y|X=x) Pr(X=x)}{Pr(Y=y)}$**

    - $Pr(X=x|Y=y)$ is probability of the random variable X to have a specific value $x$ given that another random variable has a specific value $y$.

        $Pr(X=x|Y=y) = \frac{Pr( X=x\;\cap\;Y=y )}{Pr(Y=y)}$

---

- Eg. Disease  $- 0.1\% \implies Pr(Disease) = 0.001$ , 
    Test accuracy $-95\% \implies Pr(Positive|Disease) = 0.95$ ,
    $$
    Pr(Disease|Positive) = \frac {Pr(Positive|Disease) Pr(Disease)} {Pr(Positive)} =0.0187 =1.87\% 
    $$

**Few terms : -**
- Prior
- Posterior
- Likelihood
- Evidence


---

# Parameter Estimation 

- We often assume that data X follows some probability distribution.
- The distribution is defined by a function $f_{\theta}$ that has parameters in vector form $\theta$.
- We want to estimate the best value of $\theta$ from the observed data.


- Eg, Gaussian function, defined as $f_{\theta}(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{- \frac{(x-\mu)^2}{2 \sigma^2}}$   
    where $\theta = [\mu,\sigma]$ . It has all the properties of a pdf.

---

**Using Bayes' Rule**

$Pr(\theta =\hat{\theta} |X=x) = \frac{Pr( X=x |\theta = \hat{\theta}) Pr(\theta = \hat{\theta})}{Pr(X=x)} = \frac{Pr( X=x |\theta = \hat{\theta}) Pr(\theta = \hat{\theta})}{\sum_{\tilde{\theta}}Pr(X=x | \theta = \tilde{\theta})}$

where $Pr(X=x | \theta = \hat{\theta}) = f_{\theta}$ 

**Estimation with a Dataset**

Given a dataset $S=\{x_1, x_2, x_3,........x_N\}$

- Start with a **Prior** (distribution)  guess such that $\sum_{\hat{\theta}} Pr(\theta = \hat{\theta}) = 1$

---

- For each example $x \in S$, apply Bayes' rule but  before  updating $x$ replace the prior by the average of posteriors ($Pr(\theta =\hat{\theta} |X=x))$ from earlier data each time. 

    Average of posteriors given by, $\frac{1}{N} \sum_{x\in S}Pr(\theta =\hat{\theta} |X=x)$

---

**Principle of Maximum-likelihood**  
(gives the best value of the parameter $\theta^*$ )

$\theta^* = arg\;max \prod_{i=1}^N Pr(\theta = \hat{\theta} | X=x)$

Out of all possible values of $\hat{\theta}$ choosing the one that maximaizes the product of posteriors over all data points. 



---

# Classification vs. Regression

### Classification 
- Assigning label to an unlabelled example. Eg, Spam detection.
- Solved by a classification learning algorithm
    - takes labelled examples as input
    - produces a model
    - the model takes unlabelled example as input , outputs the label or a number (like probability) that can be used to deduce the label

---

- Labels - (a member of a finite set of classes)

- **Binary classification** (size of set of classes=2)
**Multiclass classification** (size of set of classes>=3)

### Regression

- Predicting a real-valued label (also called **Target**). Eg, estimating house prices based in house feature.

- solved by a regression learning algorithm
    - takes a collection of labelled examples as input
    - produces a model 
    - model takes an unlabelled example as input & outputs a target.

---

## Model-based vs. Instance-based learning : -

- **Model based learning**
    -  creates a model that has **parameters** (using training data),  
    - training data can be discarded after the model is created,
    - eg. **SVM** 
- **Instance-based learning**
    - uses the **entire dataset** as the **model**
    - therefore, keeps the data
    - eg. **k-Nearest Neighbors (kNN)** 

---

# Shallow vs. Deep Learning

- **Shallow Learning**
    - learns directly from feraures
    - most supervised learnings are shallow
    - eg. Logistic regression , SVM
- **Deep Learning**
    - uses neural networks with many layers

(discussed later)
























