---
marp: true
theme: gaia
title: Fundamental Algorithms
math: mathjax

---

# Fundamental Algorithms
(Five most commonly used **supervised learning algorithms** will be explained here). These are :-

**1. Linear Regression  
2. Logistic Regression 
3. Decision Tree Learning
4. Support Vector Machine
5. k Nearest Neighbors**

---

# 1. Linear Regression 

A learning algorithm that learns a model which is a linear combination of features of the input examples and uses the model to output real-valued target when an input in given.

- Commanly used for regression task
- Simple and effective

---

## Problem Statement


- **Given** : *Dataset* of N labelled examples - $\{x_i,y_i\}_{i=1}^N$ 

    **$x_i$** - D-dimensional feature vector , $y_i$ - real-valued target

- **Goal** : Build a model ($f_{\boldsymbol{w},b}$) given by,

    $f_{\boldsymbol{w},b}(x)= \boldsymbol{w^*x} + b^*$

    where $w^*$ and $b^*$  the optimal values that makes the most accurate predictions


    **Predict**: Unknown $y$ using : **$y=f_{\boldsymbol{w},b}(x_{new})$** ; 

---

- **Note**:-
    - It predicts real-values not classes.
    - Unlike SVM it tries to decrease the error.

**Visualization**

![w:500](/Images/Linear%20regression%201D.png)  

---

Regression model for :-
- $1D$ - a line
- $2D$ - a plane
- $D$ - dimensions - a hyperplane of $D$-dimensions

**Note:** It is different from SVM (which has hyperplane of $D - 1$ dimensions).



---

## Solution

- **Objective function** - function that we minimize or maximize.

- We want to find the optimal values of $w$ and $b$ by minimising the prediction error. It is given by the **cost function** (or empirical risk).

    $$
    \frac{1}{N}\sum_{i=1}^N(f{w,b}(x_i)-y_1)^2
    $$

    where $(f{w,b}(x_i)-y_1)^2$ is called the loss function (or squared error loss)

- **Loss function** is a measure of how wrong the predicted value is from the real value (i.e penalty for misclassification).



---

- **Why Linear** ?
    - It's simple.
    - Rarely **overfits**.

- **Why use squared loss? Why not absolute or cubed?**

    - Absoulte error? Cubed error? All are valid. But we use squared because 
        - It exaggerates larger errors - so penalize big mistakes more

        - Has smooth , continuous derivative - Optimization becomes easier. Simpler to compute closed form solution using linear algebra.

---

- **Gradient descent** : a complex numerical optimization method (discussed later).

- **Overfitting** (high accuracy on training data, poor performance on unseen data)

![h:400](/Images/Overfitting.png) ![w:500, h:400](/Images/Linear%20regression%201D.png)

---

## Optimization 

- Minimizing cost by setting the gradient of the cost function to zero
- This gives optimal values - $w$* , $b$*

    $$
    C(w,b) = \frac{1}{N}\sum_{i=1}^N(f_{w,b}(x_i)-y_1)^2  
    $$

    <br>


    $$
    \frac{\partial{C}}{\partial{w}}= 0 \;\;\;\\;\;\;\;\; \frac{\partial{C}}{\partial{b}}= 0
    $$

---

# 2. Logistic Regression

- Not really a regression, but classification. 
- Named so - because its mathematical form is similar to linear regression.

---

## Problem Statement 

For binary classification (can also be extended to multiclass)

- **Given** : *Dataset* of N labelled examples - $\{x_i,y_i\}_{i=1}^N$ where $y_i\in\{0,1\}$ 

- **Goal** : Build a linear model that predicts whether an input belongs to class 0 or 1.
    - Problem: Can't use linear expression like **$wx_i + b$** as it spans from -$\infty$ to +$\infty$ while $y_i$ only has two values.

    - Solution: Use a simple continuous function whose codomain is (0,1)
    
---

## Logistic Regression Model

Defined as, 

$$

f_{w,b}(x)= \frac{1}{1 + e^{-(wx + b)}}

$$

- a logistic function that give values between 0 and 1 
- can be interpreted as $Pr(y=1\mid x)$
- if $f(x) >= 0.5$, predict class 1
- if $f(x) < 0.5$, predict class 0

---

## Solution

(Again we want to find the optimal values $w^*$ & $b^*$)

- Unlike linear regression ( which minimizes **MSE**), 
 Logistic regression uses **maximum likelihood** estimation (which maximize the likelihood of the training data according to our model). Given by, 

 $$
 L_{w,b}= \prod_{i=1}^N f_{w,b}(x_i)^{y_i} (1-f_{w,b}(x_i))^{1-y_i} 
 $$
 $$
 f_{w,b}(x)\;when\;y_i=1 , (1-f_{w,b}(x))\;otherwise
 $$

---

- In practice, we take *log-likelihood* instead because :-
    - presence $exp$ function, taking $log$ makes it easier to compute
    - turns products into sum
    - leads to same optimization

$$
LogL_{w,b}= \sum_{i=1}^N y_i\;lnf_{w,b}(x) + (1-y_i)\;ln(1-f_{w,b}(x))
$$

---

## Optimization

- Unlike linear regression, it doesn't have a closed forms solution

- **Gradient descent** is used to optimise such problems

---

# 3. Decision Tree Learning

- A learning algorithm that makes a **non-parametric model** given by a decision tree to predict which class the input feature vector belongs to.
- **Decision Tree**: 
    - an acyclic graph
    - where at each node of the graph a specific feature $j$ of the feature vector is examined.
    - if the value of the feature is below threshold, then left branch is followed. Else, the right branch is followed. 
    - Leaf node represents class predictions.

---

## Problem statement

**Given**: *Dataset* of N labelled examples - $\{x_i,y_i\}_{i=1}^N$ where $y_i\in\{0,1\}$ 

**Goal**: To build a decision tree (model) that can predict the label of the the class of the input feature vector belongs to.

- The tree is built from the data
- Initially, all the training data at root node.
- Repeatedly split the data by choosing the best feature & threshold.

---

## Model function

- Initially, start with a constant model $f_{ID3}^S$ given by,
$$
f_{ID3}^S = \frac{1}{\mid S \mid} \sum_{(x,y)\in S}y
$$

- Predicts same value for any input $x$.

---

### Splitting the Data

- Try all features $j = 1, \dots, D$ and all thresholds $t$.
- Split into:
  - $S^- = \{(x, y) \mid x^{(j)} < t\}$
  - $S^+ = \{(x, y) \mid x^{(j)} \geq t\}$
- Choose split that gives **lowest entropy**.

#### Entropy
- measure of uncertainty about a random variable. Given by,
$$
H(S) = -f_{ID3} \ln f_{ID3} - (1 - f_{ID3}) \ln(1 - f_{ID3})
$$

---

- Minimum Entropy = 0 ; all labels are same (no uncertainty).
- Maximum Entropy = 1 ; labels are split 50-50 (maximum uncertainty).

- Entropy after a Split :
    $$
    H(S^-, S^+) = \frac{|S^-|}{|S|} H(S^-) + \frac{|S^+|}{|S|} H(S^+)
    $$

    - Weighted average of entropies of both subsets.
    - Best split minimizes this value.

---

### Stop Splitting

Tree growth stops at a leaf node, if :

- All labels in node are same.
- No good split is found.
- Entropy reduction < $\varepsilon$ (found experimentally).
- Tree reaches maximum depth $d$ (found experimentally).

---

## Optimization Criterion

- The algorithm implicitly connects to optimizing a  **log-likelihood** :
$$
\frac{1}{N} \sum_{i=1}^N y_i \ln f_{ID3}(x_i) + (1 - y_i) \ln(1 - f_{ID3}(x_i))
$$




---


 ## Limitations of ID3

- Makes local decisions, doesn’t guarantee globally optimal tree.
- Can be improved using Backtracking (but can possibly increase the time taken to build the model) .

---

# 4. Support Vector Machines (SVMs)

- Finds the best hyperplane that separates classes with the widest margin. 

- We’ve already seen the basic idea. Now we explore how SVM handles :  
  1. **Noisy data** (imperfect separation)  
  2. **Non-linear boundaries** (inherently complex patterns)  

---

## Hard-Margin SVM

- Constraints for perfectly separable data:  
  $$
  \begin{cases}
  w x_i - b \ge +1 & \text{if } y_i = +1 \\[6pt]
  w x_i - b \le -1 & \text{if } y_i = -1
  \end{cases}
  $$

- **Objective:** maximise margin  → minimise $\tfrac12 \|w\|^2$.  
  $$
  \min_{w,b}\; \tfrac12 \|w\|^2
  \quad\text{s.t.}\quad
  y_i\bigl(w x_i - b\bigr) \ge 1\;\; \forall i
  $$

---

## 1. Dealing with Noise 

(Linearly *Non-Separable* by Outliers)

- Some points cannot be separated perfectly due to noise or mislabeling.  
- Hard-margin SVM fails to find a perfect hyperplane.

---

### Soft-Margin & Hinge Loss

- **Hinge loss:**  
  $$
  \max\!\bigl(0,\; 1 - y_i(w^\top x_i - b)\bigr)
  $$

  It is zero when the constraints are satisfied (i.e $wx_i$ lies on the correct side)

- **New cost function** :  
  $$
  \min_{w,b}\; C\|w\|^2 + \frac1N \sum_{i=1}^{N}
           \max\!\bigl(0,\; 1 - y_i(w x_i - b)\bigr)
  $$

---

- **Hyper-parameter \(C\):**

    Chosen experimentally
    - High \(C\) → penalize errors, smaller margin  
  - Low \(C\) → allow some errors, larger margin → better generalisation

---

## 2. Dealing with Inherent Non-Linearity

- Even noise-free data may need a **curved** boundary 
- cannot be seperated by a hyperplane in the original space.

### Solution : Map to Higher Dimension

- Use a mapping $\phi : x \to \phi(x)$ where $\phi(x)$ is a vector in higher dimension so that data becomes linearly separable. 

---

- Example : for 2D point $(q,p)$:  
  $$
  \Phi(q,p) = \bigl(q^2,\; \sqrt2\,qp,\; p^2\bigr)
  $$

    ![w:420 h:380](/Images/SVM(3D).png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![w:420 h:380](/Images/SVM%20(2D).png) 
- In new space, a plane can separate the classes.


---

## SVM Optimization

- Optimization uses **Lagrange multipliers**  
- Dual form (simplified):

$$
\max_{\alpha} \sum_{i=1}^{N} \alpha_i - \frac{1}{2} \sum_{i=1,j=1}^N \alpha_i \alpha_j y_i y_j (x_i.x_j)
$$

Subject to:
$$
\sum_{i=1}^{N} \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$

---

## Kernel Trick

- For data in higher dimensional form , we nedd to replace $x.x'$ by $\Phi(x_i) \Phi(x_k)$. It would it costly to do.
- To avoid computation of $\Phi(x)$ explicitly.  
    - Replace dot-product $\Phi(x_i) \Phi(x_k)$ by kernel $k(x_i,x_k)$ where $k(x_i,x_k)$ gives the same result as $\Phi(x_i) \Phi(x_k)$ 

---

- **Common kernels**: 

  - Polynomial: $k(x,x') = (x.x')^n$  
  - RBF: $k(x,x') = \exp \left(\frac{-\|x-x'\|^2} {2\sigma^2}\right)$

    where $\|x-x'\|^2$ is the squared **Euclidean distance** between two feature vectors.

---

## Final Decision Function

$$
f(x) = \operatorname{sign}\Big(\sum_{i \in SV} \alpha_i y_i k(x_i, x) - b\Big)
$$

- Only **support vectors** (with $\alpha_i > 0$) influence the decision.

---

# 5. k-Nearest Neighbors (kNN)

- **Non-parametric** and **instance-based** algorithm  
- Does **not discard** training data instead uses it as a model  
- For a new input $x$:  
  - Finds $k$ closest training examples  
  - **Returns:**  
    - Majority label (classification)  
    - Average label (regression)

---

## How Distance Is Measured

- Requires a ***distance metric***
- Common choices:
  - **Euclidean distance**
  - **Negative cosine similarity**
  - **Chebyshev distance**
  - **Mahalanobis distance**
  - **Hamming distance**
- *Distance metric* and $k$ are **hyperparameters**

---


## Cosine Similarity

$$
\text{s}(x_i, x_k) = \cos(\theta) = 
\frac{\sum_{j=1}^{D} x_i^{(j)} x_k^{(j)}}{
\sqrt{\sum_{j=1}^{D} (x_i^{(j)})^2} \cdot
\sqrt{\sum_{j=1}^{D} (x_k^{(j)})^2}
}
$$

- 1 → same direction  
- 0 → orthogonal  
- –1 → opposite direction  
- Multiply by –1 to use as **distance**

---

## Prediction as Local Linear Classifier

Assume:
- Binary classification: $y \in \{0, 1\}$  
- Normalized vectors

$$
w_x = \sum_{(x', y') \in R_k(x)} y' x'
$$

- Sum of feature vectors of **positive-labeled** neighbors  
- Decision based on cosine similarity:
$$
\text{predict} = \text{sign}(x w_x)
$$

---

## Cost Function
**Given by Li & Yang, 2003**

$$
L = -\sum_{(x', y') \in R_k(x)} y' x' w_x + \frac{1}{2} \|w_x\|^2
$$

- Optimizing this gives same $w_x$ as before  
- kNN approximates a **local linear model**

*(Using all these five learning models for predicting - refer to next file)*












