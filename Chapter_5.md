---
marp: true
theme: gaia
title: Basic Practice
math: mathjax

---

# Basic Practice
- Before training a model, there are some essential steps that needs to be done :
  - Feature engineering 
  - Learning Algorithm Selection 
  - Splitting of dataset
- Handling Underfitting and Overfitting
- Model preformance assessment
- Hyperparameter tuning 


---


# 1. Feature Engineering

- Transforming **raw data** into **datasets** that contain feature vectors which are usable by ML algorithms.
- **Dataset** = labeled examples $(x_i, y_i)$ where each $x_i$ is a vector of features $(x^{(1)}, x^{(2)}, ..., x^{(D)})$
- Requires:
  - Creativity
  - Domain knowledge

- **Informative features** = high **predictive power**


---

## One-Hot Encoding

- Used to convert **Categorical** features into **Binary vectors**

- Used when the values of the a particular features don't have an order among themselves.

- Example: Feature = “Color”  
  - red → [1, 0, 0]  
  - yellow → [0, 1, 0]  
  - green → [0, 0, 1]

- Don't use red=1, yellow=2, green=3 → introduces **false ordering**

---

## Binning (Bucketing)

- Converts **numerical** features into **categorical** ones (bins)

- Helps when **exact values** aren't important, only the **range** matters

- Example: Age   
  - 0–5 → bin1  
  - 6–10 → bin2  
  - 11–15 → bin3



---

## Normalization

- Rescales numerical features to a standard range [0, 1] or [−1, 1]
- Formula:  
  $$
  \bar{x}^{(j)} = \frac{x^{(j)} - \min^{(j)}}{\max^{(j)} - \min^{(j)}}
  $$

- Benefits:
  - Increased speed of learning. Eg. - Faster gradient descent convergence.
  - Removes scale bias  
  - Avoids **numerical overflow**

---

## Standardization

- Rescales features to have:
  - Mean (μ) = 0  
  - Standard deviation (σ) = 1  

- Formula:  
  $$
  \hat{x}^{(j)} = \frac{x^{(j)} - \mu^{(j)}}{\sigma^{(j)}}
  $$

- Good for:
  - Outliers  
  - Normally distributed data

---

## Normalization vs. Standardization

**Use Standardization When:**
- Feature has outliers  
- Follows a normal distribution  
  

**Use Normalization When:**
- Feature has known fixed range  
- No extreme values (may cause distortion of min and max) 
- Faster training required

---

## Dealing with Missing Features

- Missing values are common in real-world datasets.
- **Causes:**
  - Manual data entry errors  
  - Unavailable measurements  

- **Solutions:**
    - Drop incomplete rows  
    - Use robust ML algorithms that can deal with missing feature values 
    - Apply **data imputation techniques**

---

## Data Imputation Techniques

1. **Mean Imputation**  
   Replace missing value with the **average** value of that feature.

2. **Out-of-Range Value**  
   It can learn what is it better to do when the feature has a value significantly different from other values.
   E.g., if range is [0, 1], use 2 or −1. 

3. **Mid-Range Value**  
   Use neutral value like 0 in range [−1, 1]. Such values won't affect predictions much.

--- 

4. **Regression Imputation**  
   - Predict missing value using other features  
   - Train model using complete rows

5. **Binary Indicator Method**  
   - Add a binary feature:  
     `1` = value present, `0` = missing  
   - Replace missing value with 0 or other value

---

### Choosing the Right Imputation Technique

- No universal best method  
- Try multiple approaches and compare results

 Use the **same technique** for training and prediction

---

# 2. Learning Algorithm Selection 

- Choosing a machine learning algorithm is hard.
- We could try them all and choose the one that predicts the best — but time is limited.
- But some criterias can narrow our options.

---

## Explainability

- Does the model need to be easy to explain?
    - **Black-box models**: neural networks, ensembles  
        - High accuracy  
        - Hard to explain
    - **Transparent models**: kNN, linear regression, decision trees  
        - Easy to interpret  
        - May lose some accuracy

---

## In-memory vs. Out-of-memory

- Can the dataset fit in **RAM** ?
  - Yes → Use any algorithm
  - No → Use **incremental learning** (updates model gradually)

---

## Number of Features and Examples

- How **large** is the dataset?
    - Handles large data:
        - Neural networks (discussed later)  
        - Gradient boosting (discussed later)
    - Struggles with scale:
        - SVM (especially with many features)

---

## Data Types: Categorical vs Numerical

- What kind of features does the data have?
  - Categorical  
  - Numerical  
  - Mixed
- Some algorithms need categorical data converted into numerical data (e.g., one-hot encoding)

---

## Linearity of the Data

- Is the data **linearly separable**?
  -  Yes:
        - Logistic regression  
        - Linear regression  
        - Linear SVM
  -  No:
        - Neural networks  
        - Ensemble models (e.g., Random Forests)

---

## Training Speed

- How fast should training be?
    -  Fast:
        - Logistic/linear regression  
        - Decision trees
    -  Slow:
        - Neural networks
-  Can be improved by using:
    - Optimized libraries  
    - Multi-core CPUs for faster training

---

## Prediction Speed

- How fast does the model need to **predict**?
    -  Fast:
        - Linear/logistic regression  
        - SVM  
        - Small neural networks
    -  Slower:
        - kNN  
        - Deep/recurrent nets  
        - Ensembles


---

# 3. Splitting of dataset

- In practice, ML datasets are split into:

    1. **Training set**  
    2. **Validation set**  
    3. **Test set**

- **Training set**: largest, used to build the model  
- **Validation + Test sets**: smaller, used to evaluate  
- These two are called **hold-out sets**

---

## Why Three Sets

- **Training**: fit the model  
- **Validation**:  
  - Select best algorithm  
  - Tune hyperparameters  
- **Test**:  
  - Final evaluation before deployment

---

## Typical Data Splits

- Old rule of thumb:
    - 70% training  
    - 15% validation  
    - 15% test

- Big data setups:
    - 95% training  
    - 2.5% validation  
    - 2.5% test

---

# 4. Underfitting & Overfitting

## Underfitting  

A model **underfits** if it performs poorly on training data.  
    This means it has **high bias**.

- **Causes:**
    - Model too simple (e.g. linear regression)
    - Weak or irrelevant features

---

### Example

![h:300](/Images/Underfitting_5.png) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![h:300](/Images/Good%20fit_5.png)

### Fixes:
- Use more complex model  
- Engineer better features

---

## Overfitting

A model **overfits** if it performs well on training data but poorly on validation/test data.

- **Causes:**
    - Model too complex (deep net, large tree)
    - Too many features, too little data

- **Solutions:**

    1. Try simpler models &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3. Add more training data 
    2. Reduce feature dimensions &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. Apply **regularization**  
    
---

![w:500](/Images/Overfitting_5.png)&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; ![w:495](/Images/Good%20fit_5.png)

---

# Regularization

Used to avoid making too complex model that leads to overfitting.
- Penalizes model complexity  
- Reduces **variance**, might increase **bias**

This is the **bias-variance trade-off**

---

## L1 and L2 Regularization

### L1 Regularization (Lasso):
- Promotes **sparsity**
- Helps in feature selection

Eg. - **Linear regression (without Regularization):**
$$ \frac{1}{N} \sum (f_{w,b}(x_i) - y_i)^2 $$

---

**With L1 Regularization:**
$$ C \cdot 
|w| + \frac{1}{N} \sum (f_{w,b}(x_i) - y_i)^2 $$

### L2 Regularization (Ridge):
- Keeps weights **small**
- Smooths the model
- Differentiable

**Linear Regression (with L2 Regularization)**:
$$ C \cdot \|w\|^2 + \frac{1}{N} \sum (f_{w,b}(x_i) - y_i)^2 $$

---

## Elastic net regularization
- **Elastic Net** = L1 + L2  

(Discussed later) ; -
## Dropout & Batch normalization
Neural networks regularize using this.

## Other techniques:
- Data augmentation  
- Early stopping

---

# Model Performance Assessment

Once the learning algorithm builds a model using the training set, how do we know if it's good?

Use the **test set** to assess generalization performance.

---

# Model Assessment - for Regression model

- A good regression model produces predicted values close to actual values.
- Compare it to a **mean model**, which always predicts the average label. The fit of model should be better than fit of mean model. 
- Use **Mean Squared Error (MSE)** on:
  - Training data
  - Test data

    **If test MSE >> training MSE → Overfitting!**

    Try **regularization** or **better hyperparameter tuning**.

---

# Model Assessment - for Classification model

More complex than regression. Common metrics include:

- Confusion Matrix
- Accuracy
- Cost-sensitive Accuracy
- Precision / Recall
- Area under ROC Curve (AUC)

---

## Confusion Matrix

A table comparing **predicted** vs. **actual class labels**.

Eg., 
![](/Images/Confusion%20matrix_5.png)

where :
 **TP** = True Positives, **FN** =  False Negative, 
**FP** =  False Positives, **TN** = True Negatives



---

**Used for:**
- Analyzing error patterns
- Computing metrics: **Precision**, **Recall**

---

### Precision / Recall

- **Precision** = $\frac{TP}{(TP + FP)}$ 

    How many predicted positives are actually correct?

- **Recall** = $\frac{TP}{(TP + FN)}$

    How many actual positives were correctly predicted?

**Trade-off:**
- High precision → fewer false positives → May lead to lower recall
- High recall → fewer false negatives → May lead to lower precision

---

### Adjusting Precision vs Recall

Ways to shift the balance:

- **Weighting classes** (e.g., SVM supports class weights)
- **Tuning hyperparameters**
- **Changing decision threshold**  
  (e.g., predict positive if probability > 0.9)

---

### Precision/Recall in Multiclass

- Pick a class → treat it as **positive**
- Others → treated as **negative**
- Now use binary precision/recall formulas

---

## Accuracy

**Accuracy** = $\frac{TP}{(TP + TN + FP + FN)}$

- Useful when **all class errors are equally important** . 

 Eg, - In spam detection, false positives are worse than false negatives!

---

## Cost-Sensitive Accuracy

Account for **importance of different mistakes**:

- Assign **costs** to FP and FN
- Multiply FP and FN counts by respective costs
- Plug into accuracy formula

Useful when misclassification costs differ (e.g., in healthcare or fraud detection).

---

## Area under ROC Curve 

**ROC = Receiver Operating Characteristic**

Measures performance using:

- TPR = $\frac{TP}{(TP + FN)}$  
- FPR = $\frac{FP}{(FP + TN)}$


---

### Drawing ROC Curve

**Steps:**

1. Discretize confidence scores (e.g., 0.0 to 1.0)
2. For each threshold:
   - Predict positive if score ≥ threshold
   - Compute TPR and FPR
3. Plot **TPR vs FPR** for different thresholds.

---

## AUC (Area Under Curve)

- AUC = 1 → Perfect classifier
- AUC = 0.5 → Random guessing
- AUC < 0.5 → Model is flawed

Goal: Maximize TPR while keeping FPR low

---



![h:600](/Images/ROC%20Curve%20_5.png)


---

# Hyperparameter Tuning

**Hyperparameters** : Parameters **not learned** by the model but given by the user.  

Examples:  
- Depth of tree (ID3)
- C in SVM
- Learning rate in gradient descent

Needs to be **tuned** manually (or with automation).

---
## Ways of tuning :-

### Grid Search

Simple brute-force tuning:

1. Define a **set of possible values** for each hyperparameter.
2. Try **all combinations**.
3. Train models and evaluate performance on **validation set**.
4. Choose the best one.

---

**For example :-**

For SVM with:

- C ∈ [0.001, 0.01, ..., 1000]
- Kernel ∈ {linear, rbf}

You try all 14 combinations like:

(0.01, "linear"), (0.01, "rbf"), ...

Train each → evaluate → pick the best.

---

### Smarter Alternatives

- **Random Search**: Randomly sample from distributions of hyperparameters.
- **Bayesian Optimization**: Use past results to inform next combinations.
- **Gradient-Based** or **Evolutionary Algorithms**

There are also hyperparameter tuning libraries that can help in tuning. 

---

# Cross-Validation

Used when we **don’t have a separate validation set**. But 

- Still want to tune hyperparameters  
- Still want to estimate model performance reliably

**Steps:**

1. Split training set into k subsets of same size. 
Each subset is called ***fold***
2. Train on (k-1) *folds*, validate on 1 *fold*
3. Repeat k times, each time using a different *fold* for validation

---

4. **Average metric** over all *folds*

## Cross-Validation + Grid Search

- Combine **grid/random/Bayesian search** with **cross-validation**
- Choose best hyperparameters
- Retrain using entire training set with selected hyperparameters
- Evaluate on **test set**





