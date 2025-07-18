---
marp: true
theme: gaia
title: Problems and Solutions
math: mathjax

---

# Advanced Practice in Machine Learning  
**(Handling Imbalanced Datasets & Combining Models)** 

### Why "Advanced Practice"?

- These techniques are **not harder**, but used in **less frequent, specific situations**

---

## Problem with Imbalanced Datasets

- Real-world data often has **unequal class distribution**
  - Eg. Fraud detection → 99.9% transactions are genuine
- Classifiers tend to **favor the majority class**
  - **Why?** 
    - Minimizing overall error -  **favors  class that dominates the data**
    - If 99.9% are genuine, a model accuracy - 99.9% **just by predicting “genuine” every time**

---

## SVM and Imbalanced Data

- SVM tries to **minimize a loss function** based on classification margin
- When majority class dominates:
  - **Misclassifying majority points contributes more** to overall cost
  - So the algorithm shifts decision boundary **to favor the majority**
- Result: **Minority class gets ignored**


---

## Adjusting Class Weights in SVM

- You can assign **higher penalty** to misclassifying minority class
  - In `scikit-learn`, use: `class_weight='balanced'`
- The model will **shift the hyperplane** to reduce errors on rare class
- This can improve **recall (sensitivity)** for the minority class  
  → important in fraud, disease detection, etc.

---

## Data-Level Solutions

Balance dataset **before training**, without modifying the algorithm

### 1. Oversampling:
- Duplicate existing examples of minority class
- Helps the model see the class more often during training

### 2. Undersampling:
- Randomly remove examples from the majority class
- Risk: Potentially discard useful data

---

## Synthetic Sampling (SMOTE & ADASYN)

 **SMOTE :-**
- For each minority point $x_i$, pick a neighbor $x_z$
- Create:  
  $$
  x_{new} = x_i + \lambda (x_z - x_i)
  $$  
  $$
  \text{where } λ ∈ [0,1]
  $$
- Result: New point **on the line** between $x_i$ and $x_z$

---

 **ADASYN:**
- Like SMOTE, but **focuses on areas with fewer minority samples**
- Generates **more synthetic points** where the class is **harder to learn**

## Why Use SMOTE or ADASYN?

- Helps **generalize** decision boundaries
- Prevents **overfitting** that simple oversampling (copying) may cause
- Makes training dataset **more representative**


---

## Models Robust to Imbalance

Some algorithms handle imbalance **better by default**:

- Decision Trees  
- Random Forest  
- Gradient Boosting

**Why?**
- They split based on **information gain** or **Gini index**, not class frequency
- Learn **local patterns**, even from a few minority examples

---

# Combining Models

Also called **Ensembling**

- Combine **multiple models** to improve accuracy
- Boosts performance by **leveraging diversity**
- Example:
  - One model detects fraud well
  - Another is better at detecting outliers
  - Together, they’re more powerful

---

## Ways to Combine Models

### 1. Averaging  
 For regression or classifiers that return **probabilities**

### 2. Majority Voting  
For classification (each model votes on output)

### 3. Stacking  
Combine model outputs using a **meta-model**

---

## Why Does Ensembling Work?

 “Several uncorrelated models agreeing = more likely to be correct”

- Each model makes **different errors**
- If errors are **uncorrelated**, they can **cancel each other out**
- Works best with models trained on:
  - **Different features**
  - **Different algorithms** (e.g., SVM + Random Forest)

---

## Averaging & Majority Voting

### Averaging:
- Models return **score/probability**
- Combine by **taking the mean**
- Smooths out individual model biases

### Majority Vote:
- Each model picks a class
- Final class is the **most common prediction**
- For ties: Random tie-break Or reject prediction (if error is costly)

---

## Stacking (Meta-Learning)

- Combine predictions from multiple models using another model
- Base models: $f_1(x), f_2(x), ...$
- For each input $x$, collect:
  $$
  x̂ = [f_1(x), f_2(x), ...]
  $$  
  This becomes the input to a **meta-model**
- The true label $y$ is used as target → train on $(x̂, y)$

---

## What Kind of Meta-Model?

- Often a **simple, fast** model like:
  - **Logistic Regression** (most common)
  - **Decision Tree**
  - **Random Forest** (if base models are very different)
- Purpose: Learn how to **trust or weight** each base model

---

### Example: Stacking for Fraud Detection

**Base models:**
1. $f_1$ – Logistic Regression on transaction amount
2. $f_2$ – Random Forest on user behavior
3. $f_3$ – SVM on device/IP history

**Meta-model:**
- **Logistic Regression** trained on outputs $[f_1(x), f_2(x), f_3(x)]$

The meta-model learns:

---

- When $f_1$ is unreliable (e.g., small transactions)
- When $f_2$ is more confident (e.g., behavioral anomalies)

**# Use Class Probabilities (Not Just Labels)**

- If base models output **probabilities or scores**, use them:
  - Instead of $[0, 1, 1]$, use $[0.2, 0.9, 0.8]$
- Gives meta-model **more information** about certainty
- Helps capture **confidence patterns** from base models

---

## Training Neural Networks

Neural networks expect **structured, uniform input**

- **Images** → Resize to same dimensions, normalize to [0, 1]. Ensures pixel values have consistent scale  
- **Text** → Tokenize (words, punctuation, symbols)

Each data type needs careful **preprocessing** before feeding into a network

---

## Encoding Text for Neural Networks

### For CNNs & RNNs:
- Use **one-hot encoding**:  
  - Each word/token becomes a binary vector  
- Better: Use **word embeddings** (e.g., Word2Vec, GloVe)  
  - Capture semantic relationships between words

### For MLPs (Multilayer Perceptrons):
- Use **Bag-of-Words**, especially for **longer texts** like reviews, emails.  
  - Works poorly for **short texts** like tweets or SMS

---

## Choosing a Neural Network Architecture

- Many tasks (like sequence-to-sequence) have **multiple architectures**
- New models emerge **every year** (e.g., Transformers, ConvNeXt, etc.)

**How to choose?**
- Look for **state-of-the-art** solutions using:
  - Google Scholar  
  - Microsoft Academic

---

- If you want simplicity:
  - Search GitHub for existing implementations  
  - Start from something **close to use case**

## Should You Use the Latest Model?

Fancy/Modern ≠ Better (for your case)

- Modern models often:
  - Require **huge datasets**
  - Need **expensive hardware**
  - Are **complex to implement**

---

**Often better to:**
- Use a **simpler, well-tested model**
- Focus effort on:
  - **Preprocessing**
  - **Getting more data**
  - **Tuning your training process**

---

## Deciding Network Depth and Size

**Start simple → grow as needed**
1. Begin with **1 or 2 layers**
2. Train and check:
   - Does it fit training data well? ( **Low bias**?)
3. If **underfitting**:
   - Increase **layer size**
   - Add **more layers**
4. If **overfitting**:
   - Apply **regularization**

---
Iterate until your model performs well on **both training and validation** sets

---

## Advanced Regularization Techniques

Neural networks overfit easily → Regularization is essential

Beyond L1/L2, use:

- **Dropout**  
- **Batch Normalization**  
- **Early Stopping**  
- **Data Augmentation**

Some methods are **specific to neural nets**; others are **universal**

---

## Dropout: -

**Dropout** randomly disables a percentage of neurons during each training pass

- Forces the network to **not rely too much on any one neuron**
- Reduces **co-dependency** between neurons

Control it via a **dropout rate** (0 to 1)
- Example: `Dropout(0.5)` drops half the units per batch  
- Tune the rate using **validation data**

Add dropout layers in frameworks like Keras, PyTorch

---

## Batch Normalization : -

Standardizes outputs of each layer → keeps training **fast and stable**

- Normalizes intermediate activations:
  - Mean ≈ 0  
  - Std Dev ≈ 1
- Reduces **internal covariate shift**

- Though not designed for it, it acts like a **regularizer**

- Often placed **between linear (Dense/Conv) layers and activations**

---

## Early Stopping : -

Training loss always goes down — but validation loss doesn’t!

- After some point, the model **overfits**
- Early stopping **pauses training** before that happens

**How it works :**
1. After each epoch, save the model
2. Monitor validation performance
3. Stop when it **starts to degrade**

Keep **best-performing checkpoint**, prevent waste of compute

---

## Data Augmentation: Create More from Less

- Used mostly in **image tasks**

- Apply transformations to training images:
    - Rotate, Flip, Zoom, Change brightness, etc.

- Label stays the same  
- Great for boosting model **generalization**
- Also used in: Audio (noise addition), Text (e.g., synonym replacement)

---


## Handling Multiple Inputs

- In many real-world problems, data is **multimodal**  
  → comes from different sources/types

**Examples**:
- Product info = **image + description**
- Medical diagnosis = **X-ray + patient record**
- Visual question answering = **image + text**

---

## Are Shallow Models Enough?

Traditional ML models struggle with multimodal data because:
- They expect **flat, single-type** input (just vectors)
- They can't **specialize** for each data modality



---

## SOLUTION : Train Models Separately

### Step-by-step:
1. Train one model on **image**
2. Train another on **text**
3. Combine predictions  
   → use **ensembling** (e.g., stacking, averaging)

Works well if:
- Modalities are **somewhat independent**
- Each model is good at capturing its part

---

## Alternative: Concatenate Feature Vectors

If you can **vectorize** both inputs:

- Image features: $[i_1, i_2, i_3]$  
- Text features: $[t_1, t_2, t_3, t_4]$

Combine to: $[i_1, i_2, i_3, t_1, t_2, t_3, t_4]$

Train a single model on this **combined feature vector**

- Useful with shallow models like logistic regression, SVM, or tree-based methods

---

## Deep Learning: More Flexible and Powerful

Neural networks allow:
- **Separate subnetworks** for each input type
- Special layers suited for each:
  - **CNN** for images
  - **RNN or Transformer** for text

Then:
- Combine learned **embeddings**
- Add **classification layers** on top

---

## Example: Neural Model for Image + Text

**Task:** Check if caption matches the image

### Network Structure:
- CNN → Embedding vector for image  
- RNN → Embedding vector for text

- **Concatenate embeddings**  
- Feed to a **classifier layer** (e.g., softmax)

End-to-end training: model learns **joint representation**  

---

## Handling Multiple Outputs

Some tasks need to predict more than one thing at a time  **Multi-output problems**

Example :
- Input: Image of a cat  
- Outputs:  
  - Coordinates of the cat
  - Label: “cat”

---

## Strategy 1: Multi-label Classification

- When outputs are **independent** and **discrete** (e.g., tags)
- Model predicts all labels at once using:
  - One-hot or multi-hot vectors
  - Sigmoid activation + binary cross-entropy loss

Works well when outputs are **similar in nature**

---

## Strategy 2: Separate Outputs from Shared Encoder

Complex case: Outputs are **different in nature**

- Example:
    - Input: Image
    - Output 1: Coordinates (real numbers)
    - Output 2: Label (categorical class)

Solution :  
- Build an **encoder subnetwork** (e.g., CNN)

---

- Add two **heads** on top:
  - One predicts coordinates
  - One predicts label

### Architecture Breakdown

- **Input** : Image (e.g., 128×128×3)


- **Convolutional Encoder**
  - Extracts features from the image
- **Embedding Layer**
  - Dense representation used by both heads

---

**Two Output Heads**
-  **Coordinates Head** *(Regression Task)*
    - Activation: **ReLU**
    - Loss: **Mean Squared Error (MSE)**
    - Predicts: Real-valued coordinates (e.g., object position)

- **Label Head** *(Classification Task)*
    - Activation: **Softmax**
    - Loss: **Cross-Entropy**
    - Predicts: Class probabilities (e.g., Cat vs. Dog)

---

## Problem: Two Loss Functions

We now have:
- `C₁`: Loss for coordinates (e.g., MSE)
- `C₂`: Loss for label (e.g., cross-entropy)

Optimizing both **at once** is hard:
- Improving one may affect the other

---

## Solution: Weighted Combined Loss

Define combined loss :
$$
\text{Total Loss} = \alpha C_1 + (1 - \alpha) C_2
$$

- $\alpha \in (0, 1)$ is a **hyperparameter**

- Controls **tradeoff** between tasks

- If labels are more important, choose lower $\alpha$  
- If location matters more, use higher $\alpha$

---

# Transfer Learning



- Adapt a pre-trained model to a **new but related task**  
- Useful when:
  - You have a **large labeled dataset** for Task A (e.g., wild animals)  
  - You have **limited data** for Task B (e.g., domestic animals)  

---

### Why Does Transfer Learning Work?

- Early neural network layers learn **general features** (edges, shapes, textures)  
- These features are **reusable** across different but related problems  
- Saves time and data labeling effort  

---

## How Does It Work?

1. Train a deep model on a **large original dataset**  
2. Remove the last one or more layers (usually task-specific layers)  
3. Replace with new layers suited for your new problem  
4. **Freeze** the remaining pre-trained layers  
5. Train only the new layers on the **smaller new dataset**  

---

## Example: Wild Animals → Domestic Animals

- Pre-trained CNN on wild animal images (large dataset)  
- Remove classifier layers, add new layers for domestic animal classes  
- Freeze convolutional layers (feature extractor)  
- Train new classifier layers on small domestic animal dataset  

---

# Algorithmic Efficiency

## Why Efficiency Matters

- Some algorithms solve problems but are too slow or memory-heavy to be practical  
- Efficiency depends on how time or space grows with input size  

---

## Big-O Notation: Measuring Algorithm Complexity

- Describes **worst-case growth** of time or space with input size $N$  
- Example:  
  - Naive max-distance algorithm: $O(N^2)$
  - Optimized max-distance algorithm: $O(N)$ 

---

## Naive Algorithm Example (O(N²))

```python
def find_max_distance(S):
    result = None
    max_distance = 0
    for x1 in S:
        for x2 in S:
            if abs(x1 - x2) >= max_distance:
                max_distance = abs(x1 - x2)
                result = (x1, x2)
    return result
```

Checks every pair → slow for large datasets



---

## Efficient Algorithm Example (O(N))
```python

def find_max_distance(S):
    min_x = float("inf")
    max_x = float("-inf")
    for x in S:
        if x < min_x:
            min_x = x
        elif x > max_x:
            max_x = x
    return (max_x, min_x)
```
One pass through data → much faster

---

## Efficient Data Structures Matter

- If **order doesn’t matter**, use a **set** instead of a list  
- Why?  
  - Membership check `x in S` is **fast** $(O(1))$ for sets  
  - But **slow** $(O(N))$ for lists, especially large ones  

- Use Dictionaries for Fast Lookups

    - Python **dict** = key-value pairs (aka hashmap)  
    - Allows **fast retrieval** of values by key $(O(1))$ 
    - Useful for counting, caching, indexing data  

---

## Prefer Popular Scientific Libraries

- Libraries like **numpy**, **scipy**, **scikit-learn** are optimized for speed  
- Implemented in **C** under the hood → faster than pure Python  

## Use Generators for Large Data

- Generators yield one item at a time instead of loading all data at once  
- Saves memory and improves performance when handling big datasets  

