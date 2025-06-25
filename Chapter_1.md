---
marp: true
theme: gaia
title: Machine Learning Presentation
math: mathjax

style: |
    .img-center{
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100vh;
        margin: 0;
        padding-top: 20px;
    }
    
    .img-center img { 
        max-width: 90%;
        height: auto;
        margin: 0;
        padding-top: 20px; 
    }

---
# Can we teach machines to learn ?

#### Problem : Detecting Spam Emails

**Traditional Programming**

- writing explicit rules for recognizing every possible spam message

- Eg. If the message contains - *"win money", "free offer"*, etc.

**Issues**
- not feasiable if too many and complex rules, easy to bypass, no learning, poor generalization
---

# Machine Learning  

Machine learning is a subfield of **artificial intelligence** that focuses on developing **algorithms** that enable systems to **learn patterns** and make decisions from data **without being explicitly programmed**.

---

## What is needed ?
1. **Dataset** 
2. **Algorithm**

<br>

### What the machine does? 
Uses the **algorithm** to make a **model** based on the given dataset. Then uses the model to predict and solve a problem.

---

<br>
<br>
<br>

# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Types of Machine Learning

$$
\text{(based on the availability and nature of the data)}
$$



---

## 1. Supervised Learning

**Dataset** - collection of labeled examples {(x<sub>i</sub>,y<sub>i</sub> )}<sup>N</sup><sub>i=1</sub>

**x<sub>i</sub>** - **feature vector** 
$$
x_i=[x^{(1)}, x^{(2)}, x^{(3)},....x^{(D)}]
$$

**y<sub>i</sub>** - **label**

**Goal :** Produce a model that takes a feature vector as input and output the label for it.

---

#### Support Vector Machine (SVM) 

- a supervised learning algorithm

- labels - classes

- requires positive examples - labelled as +1 (class of interest) & negative examples - labelled as -1 (the other class)

- creates a decision boundary to seperate the two categories of data

-  prefer that the hyperplane separates positive examples from negative ones with the **largest margin** (for better generalization and making the model more roboust to noise).

---

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![h:600](/Images/SVM.png)

---

- **Decision boundary** (the hyperplane)  given by,
$$
\mathbf{wx} - b = 0
$$
            
- A label, $y = sign(\mathbf{wx}-b)$

- **Model:** $f(x) = sign(\mathbf{w^*x}-b^*)$
    
    $w^*$ - optimised value of w
    $b^*$ - optimised value of b

- Constrains : $y_i (\mathbf{wx} - b) \ge 1$
- Minimize $||\mathbf{w}||$ (so that the margin is large)
- Eg. , checking if a message is spam or not_spam.

---

# 2. Unsupervised Learning

**Dataset** - collection of unlabelled examples {x<sub>i</sub>}<sup>N</sup><sub>i=1</sub>

**x<sub>i</sub>** - feature vector

**Goal :** Create a model that takes an input vector and returns something that give information about the structure/pattern of the data. Eg - emails without spam and not spam labels.

---
# 3. Semi-Supervised Learning

**Dataset** - collection of both labelled and unlabeled data

**Goal:** Same as Supervised learning. The unlabelled data is used to improve the model's understanding of the structure of the data distribution.

---

# 4. Reinforcement Learning

In this the machine lives in an **environment (E)**, precieves the **state(S)** of the environment and takes **action (A)** in every states. The different actions taken leads to different **rewards (R)**. Over time it learns what are the **optimal actions** in a particular state. $E=(S,A,P,R,γ)$
Eg. Solving a maze problem, 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;![w:300](/Images/Maze_1.png) 
**Goal :** To learn a **Policy** (a function that takes feature vector of a state as input and give the optimal action that can be taken in that state as output).

---

# Why the Model Work on New Data?

Because of **statistical generalizations**: 

- If the data is random and representative then we can make statements or predictions about the whole population or future data.

PAC learning theory explores this in more detail. 





