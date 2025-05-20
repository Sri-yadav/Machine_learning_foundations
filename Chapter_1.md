---
marp: true
theme: gaia
title: Machine Learning Presentation

---
# Machine Learning  

A subset of artificial intelligence which is concerned with developing algorithms and models that can help system to find patterns in the given data and make predications without giving detailed instructions or rules for every possible situation.

---

## What is needed ?
1. Dataset 
2. Algorithm

<br>

### What the machine does? 
Uses the **algorithm** to make a **statistical model** based on the given dataset. Then uses the model to predict and solve a problem.

---

<br>
<br>
<br>

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Types of Machine Learning



---

## 1. Supervised Learning

**Dataset** - collection of labeled examples {(x<sub>i</sub>,y<sub>i</sub> )}<sup>N</sup><sub>i=1</sub>

**x<sub>i</sub>** - feature vector
**y<sub>i</sub>** - label

**Goal :** Produce a model that takes a feature vector as input and output the label for it.

---

#### Support Vector Machine (SVM) 

- a supervised learning algorithm

- labels - classes

- requires positive label has +1 value & negative label has -1 value

- creates a decision boundary to seperate different categories of data

-  prefer that the hyperplane separates positive examples from negative ones with the largest margin (for better generalization).

---

- **Decision boundary** (the hyperplane)  given by,  **wx** - b = 0
            
- A label, y = sign(**wx**-b)

- **Model:** f(x) = sign(**w\*x** - b\*)
    
    **w\*** - optimised value of w
    b\* - optimised value of b

- Constrains : y<sub>i</sub> (**wx** - b) >= 1
- Minimize ||**w**|| (so that the margin is large)
- Eg. , checking if a message is spam or not_spam.

---


**Few terms to know :-**

- Kernels
- Bag of words
- Outliers
- Accuracy

---

**Note:** Every classification algorithm distinguish between classes by creating a decision boundary explicitly or implicitly.

- These algorithms are majorly distinuished by :-
    - Form of the decision boundary (determines the accuracy of the model)
    - Speed of model building 
    - Prediction processing time



---

# 2. Unsupervised Learning

**Dataset** - collection of unlabelled examples {x<sub>i</sub>}<sup>N</sup><sub>i=1</sub>

**x<sub>i</sub>** - feature vector

**Goal :** Create a model that takes an input vector and returns something that give information about the structure/pattern of the data. Eg - in clustering, it returns cluster-id.

---
# 3. Semi-Supervised Learning

**Dataset** - collection of both labelled and unlabeled data

**Goal:** Same as Supervised learning. The unlabelled data is used to improve the model's understanding of the structure of the data distribution.

---

# 4. Reinforcement Learning

In this the machine lives in an **environment**, precieves the **state** of the environment and takes **action** in every states. The different actions taken leads to different rewards. Over time it learns what are the **optimal actions** in a particular state.

**Goal :** To learn a **Policy** (a function that takes feature vector of a state as input and give the optimal action that can be taken in that state as output).









