---
marp: true
theme: gaia
title: Problems and Solutions
math: mathjax

---

## Linear Regression  

- Linear regression fits a straight line to data.
- But what if the data is not linear?
- Polynomial regression is one option:
  $$
  y = w_1x_i + w_2x_i^2 + b
  $$

---

## Polynomial Regression

- Fit parameters $w_1, w_2, b$ using:
  - Mean squared error
  - Gradient descent
- Works well for low-dimensional data
- But for high-dimensional input ($D > 3$):
  - Hard to choose the right polynomial
  - Difficult to visualize

---

# Kernel Regression

- **Non-parametric method**
  - No parameters to learn
- Model depends entirely on data
- Similar to **kNN (k-Nearest Neighbors)**

**Prediction function**: $f(x) = \frac{1}{N} \sum_{i=1}^{N} w_i y_i$

$$ 
\text{where } w_i = \frac{k(x_i - x, b)}{\sum_{k=1}^{N} k(x_k - x, b)}
$$


---

## What is a Kernel?

- A kernel measures **similarity**
- Commonly used: **Gaussian kernel**
$$
k(z) = \frac{1}{\sqrt{2\pi}} \exp\left(-\frac{z^2}{2}\right)
$$
- Weighs data points based on their distance from $x$

---

## Hyperparameter $b$

- $b$ controls the **bandwidth** of the kernel
- Affects how local or global the model is
- Tuned using a **validation set**
  - Try different $b$ values
  - Pick the one that minimizes validation error

---

## Effect of $b$ on Fit

![w:1150](/Images/Effect_of_b_7.png)
- Small $b$ : Overfit (wiggly curve)
- Large $b$: Underfit (too smooth)
- Kernel regression works with high-dimensional features also

---

# Multiclass Classification

<small>

- In multiclass classification:
  $$
  y \in \{1, 2, ..., C\}
  $$
- The label can belong to one of **C classes**.
- Many algorithms are binary (e.g., SVM).
- But some algorithms **can naturally handle** multiple classes.

**Algorithms That Naturally Extend**

-  Decision Trees (e.g., ID3)
    - Estimate class probability: 
    $f_{ID3}(x) = Pr(y_i = c|x) = \frac{1}{|S|} \sum_{\{y \mid (x, y) \in S, y = c\}} y$
    - Just count how many times each class appears in dataset $S$.

</small>

---

- Logistic Regression
    - Replace **sigmoid** with **softmax** for multiple classes.
    - Softmax gives probabilities across multiple classes:
      $$
      \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}
      $$

- k-Nearest Neighbors (kNN)
    - Look at **k closest points**
    - Predict the **most frequent class** among them

---

**Algorithms That Don't Extend Naturally**

- **Support Vector Machines (SVM)**
    - Designed for **binary classification**
    - No direct support for multiple classes
    - **Strategy: One-vs-Rest (OvR)**
        - Convert a **C-class** problem into **C binary problems**
        - For each class:
            - Label it as 1 (positive)
            - Label all others as 0 (negative)

---

**One-vs-Rest: Example (3 Classes)**

- If $y \in \{1, 2, 3\}$, make 3 datasets:
  - Model 1: 1 vs not-1 (2,3)
  - Model 2: 2 vs not-2 (1,3)
  - Model 3: 3 vs not-3 (1,2)
- Train **3 binary classifiers**

**Making Predictions**
- Input: a new feature vector $x$
- Apply all 3 models → get 3 scores
- Pick the class with the **highest score** (or most certain prediction)

---

## Interpreting Certainty

### Logistic Regression:
- Returns probability $\in (0, 1)$
- Higher value → more certain

### SVM:
- Returns a **distance to decision boundary**:
$$
d = \frac{w \cdot x + b}{\|w\|}
$$
- Greater distance → higher certainty

---

# One-Class Classification

- Also called:
  - Unary Classification
  - Class Modeling
- Learns from **only one class** of data
- **Goal:** Detect if a new input **belongs to this class** or not
- Used in:
  - Outlier detection
  - Anomaly detection
  - Novelty detection

---

## How is it Different?

- Traditional classification:  
  - Learns from **multiple classes**
- One-class classification:  
  - Learns from **only one class**

**Example:**  
Detecting normal traffic in a secure network  
- Normal traffic: many examples  
- Attack traffic: rare or unknown

---

## Common One-Class Algorithms

- One-Class Gaussian
- One-Class k-Means
- One-Class k-Nearest Neighbors (kNN)
- One-Class Support Vector Machine (SVM)

## One-Class Gaussian

- Assumes data comes from a **Multivariate Normal Distribution (MND)**  

---

- Probability Density Function:
    $$
    f_{\mu,\Sigma}(x) = \frac{1}{(2\pi)^{D/2}|\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x - \mu)^T \Sigma^{-1}(x - \mu)\right)
    $$
    where:
    - $\mu$: Mean vector (center)
    - $\Sigma$: Covariance matrix (shape)
    - $|\Sigma|$: Determinant of the covariance matrix
    - $\Sigma^{-1}$: Inverse of the covariance matrix
    - $(x - \mu)^T$: Transpose of the difference vector*

---

- **Interpretation of Parameters**

    - **$\mu$** → Where the Gaussian is centered  
    - **$\Sigma$** → Shape and spread of the distribution  
    - Use **maximum likelihood** to learn both from data

- **Making Predictions**

    - Once model is trained, compute:
      $$
      f_{\mu,\Sigma}(x)
      $$
    - If this **likelihood > threshold** → belongs to the class  
    - Otherwise → **classified as an outlier**
    - Threshold is chosen based on: Experimental tuning & Domain knowledge

---

## Mixture of Gaussians 

- For complex data shapes:
  - Combine several Gaussians
- Learn:
  - One $\mu$ and $\Sigma$ for each component
  - Extra parameters to combine them

(Covered later)

---

## One-Class k-Means

- Use **k-means clustering** on training data
- For a new input $x$ :
  - Compute:
    $$
    d(x) = \min_{i} \text{distance}(x, \text{cluster center}_i)
    $$
  - If $d(x)$ < threshold → accept as in-class

---

## One-Class SVM

- Two formulations:
  1. **Separate data from origin** in feature space  
     - Maximize margin from origin
  2. **Find minimal hypersphere** around data  
     - Enclose training data with smallest possible volume

---

# Multi-Label Classification

- Each input can have **multiple labels**.
- Example: An image might be labeled:
  - "people", "concert", "nature" — all at once unlike multiclass, where one example has one label

---

## Transforming Multi-Label to Multi-Class

- If labels are like **tags** (many, same nature), you can:
  - Split each example into multiple ones — one per label
  - Each has the same feature vector but only one label
- Now treat it as **multiclass**
- Use **One-vs-Rest strategy**

**Threshold Hyperparameter**

- Prediction produces **scores for each label**

---

- Apply a **threshold**:
  - If score > threshold → assign that label
- Multiple labels can be assigned if multiple scores exceed threshold
- Threshold is chosen using **validation set**

## Algorithms That Work for Multi-Label

- Algorithms that naturally work for **multiclass** can be reused:
  - Decision Trees
  - Logistic Regression
  - Neural Networks
- These return a score per class → apply thresholding

---

## Neural Networks for Multi-Label

- Output layer:
  - One unit per label
  - Each unit uses **sigmoid** activation
- Labels are binary:
  $$
  y_{i,l} \in \{0, 1\}
  $$
- Use **Binary Cross-Entropy Loss**:
  $$
  -\left( y_{i,l} \log(\hat{y}_{i,l}) + (1 - y_{i,l}) \log(1 - \hat{y}_{i,l}) \right)
  $$

---

## Total Loss for Neural Networks

- Average the binary cross-entropy:
  - Over all **labels $l$** and all **examples $i$** in the training set
- Optimized using gradient descent or other optimizers

---

## Alternate Approach :
### Flatten Labels

- Used when number of combinations is **small**
- Example:
  - Label 1: {photo, painting}
  - Label 2: {portrait, paysage, other}
- Make **6 fake classes** combining both

---

### Fake Class Mapping Table

| Fake Class | Label 1 | Label 2   |
|------------|---------|-----------|
| 1          | photo   | portrait  |
| 2          | photo   | paysage   |
| 3          | photo   | other     |
| 4          | painting| portrait  |
| 5          | painting| paysage   |
| 6          | painting| other     |

- Now treat as **standard multiclass problem**

---

## Pros and Cons of Flattening

### Advantage:
- Keeps **label correlations** intact  
  (labels depend on each other)

### Disadvantage:
- If label combinations are many:
  - Need a **lot more data**
  - Class space grows exponentially

---

## Why Label Correlation Matters?

- Predicting multiple **related labels**
- Example: Email classification  
  - Labels: [spam, not_spam], [ordinary, priority]
- Avoid invalid combos like:
  - [spam, priority] — doesn’t make sense

---

# Ensemble Learning

- Train **many weak models**, not one strong model
- Combine their predictions → build a **meta-model**
- Goal: Improve **accuracy** through diversity and voting

### Weak Learners

- Usually fast and simple models (e.g., **shallow decision trees**)
- Individually: not very accurate
- Together: can form a strong predictor

---

## How Does Ensemble Work?

- Each weak model gives a prediction
- Predictions are **combined** (average or vote)
- Example: If most say “spam,” we label input as spam

---

# Two Popular Algorithms

- **Random Forest** (Bagging)
- **Gradient Boosting** (Boosting)

## Random Forest:- 


#### Bagging

- Bagging = Bootstrap Aggregating
- Make **B random samples** (with replacement)
- Train **B decision trees**
- Combine predictions

---

**Sampling in Bagging**

- For each sample $S_b$, draw $N$ examples **with replacement**
- Train one decision tree on each $S_b$
- Predict on new example $x$ :
  - Regression: average of predictions
  - Classification: majority vote

#### Feature Randomization

- At each split → pick **random subset of features**
- Prevents correlation between trees
- Why? Correlated trees -> less useful diversity

---

## Gradient Boosting

- Another ensemble method
- Trains models **sequentially**, not in parallel
- Each new model fixes **errors** made by previous ones

## Gradient Boosting for Regression

1. Start with a constant model:  
$f_0(x) = \frac{1}{N} \sum_{i=1}^{N} y_i$
2. Compute **residuals**:  $\hat{y}_i = y_i - f(x_i)$

---

3. Train tree on residuals
4. Update model:  
   $f = f_0 + \eta f_1$
5. Repeat until M trees trained

### Why "Gradient" Boosting?

- We don't compute gradients directly
- Residuals are **proxies for gradients**
- Just like small steps in gradient descent

---

### Key Hyperparameters

- Number of trees (M)
- Learning rate ($\eta$)
- Tree depth

Deeper trees = slower training, better accuracy

---

## Bias vs Variance

- **Bagging** reduces **variance** (avoids overfitting)
- **Boosting** reduces **bias** (avoids underfitting)
- Boosting can overfit → tune depth, $M$, and $\eta$

---

## Gradient Boosting for Classification (Binary)

- Prediction uses **sigmoid function**:  
  $$
  \Pr(y=1|x) = \frac{1}{1 + e^{-f(x)}}
  $$  
  $$
  \text{where } f(x) = \sum_{m=1}^{M} f_m(x)
  $$
---

## Maximizing Likelihood

- Maximize:  
  $$
  L_f = \sum_{i=1}^{N} \ln(Pr(y_i = 1 | x_i, f))
  $$
- Start with:  
  $$
  f_0 = \ln\left(\frac{p}{1 - p}\right), p = \frac{1}{N} \sum y_i
  $$

---

## Training Steps per Iteration

1. Compute gradients $g_i$ for each example
2. Replace $y_i$ with $g_i$ in dataset
3. Train new tree $f_m$
4. Find optimal step $\lambda_m$
5. Update model:  
   $f \leftarrow f + \eta \lambda_m f_m$

Repeat until $m = M$

---

# Learning to Label Sequences  

## What is Sequence Labeling?

- Sequence 
  - Language: words/sentences
  - Biology: DNA/proteins
  - Finance: stock prices
- Sequence labeling = assign labels to each element of sequence

---

## Input and Output Format

- Each example is a pair of lists (X, Y)
  - X = list of feature vectors (per time step)
  - Y = list of labels (same length as X)
- Example:  
  X = ["big", "beautiful", "car"]  
  Y = ["adjective", "adjective", "noun"]

---

## Formal Notation

- Example $i$:  
  $$
  X_i = [x^1_i, x^2_i, ..., x^{n}_i]
  $$
  $$ 
  Y_i = [y^1_i, y^2_i, ..., y^{n}_i]
  $$  
- Each $y^t_i \in \{1, 2, ..., C\}$  
- Length of each sequence = $\text{size}_i$

---

## Using RNNs for Sequence Labeling

- At time $t$: input $x^t_i$, output $y^t$
- Types of labels:
  - Binary
  - Multiclass
  - Multilabel
- Uses recurrent structure to capture **context**

---

## Conditional Random Fields (CRF)

- CRF - popular alternative to RNNs
- Useful when **feature vectors are rich**
- Example task: Named Entity Recognition (NER)
  - Sentence: “I go to San Francisco”
  - Labels: \{location, name, company_name, other\}

---

## Feature Engineering in CRFs

- CRFs depend on informative features like:
  - Does the word start with a capital letter?
  - Is the word found in a location list?
- Requires **handcrafted features** and **domain expertise**

---

## CRFs vs RNNs

| Criteria         | CRF          | RNN               |
|------------------|--------------|-------------------|
| Feature design   | Requires handcrafted features | Learns features automatically |
| Speed            | Slower training | Faster with large data|
| Accuracy         | Good with strong features | Usually better with deep models|
| Scalability      | Limited      | High              |

---

# Sequence-to-Sequence Learning

- Seq2seq learning generalizes sequence labeling  
- Input sequence $X_i$ and output sequence $Y_i$ can have **different lengths**  
- Applications:  
  - Machine translation (e.g. English → French)  
  - Conversational interfaces (chatbots)  
  - Text summarization  
  - Spelling correction  
  - And many others  

---

## Neural Networks for Seq2Seq

- Many seq2seq problems are solved by neural networks  
- Multiple architectures exist depending on the task  
- All share a common structure:  
  - **Encoder**  
  - **Decoder**  
- Also called **encoder-decoder neural networks**

---

## Encoder

- Neural network that reads sequential input  
- Can be:  
  - RNN  
  - CNN  
  - Other architectures  
- Generates a **state** (numerical representation of meaning)  
- Output is an **embedding** (vector or matrix of real numbers)

---

## Decoder

- Neural network that takes the embedding as input  
- Generates a sequence of outputs  
- Starts with a start-of-sequence vector $x^{(0)}$ (often zeros)  
- Produces output $y^{(1)}$ and updates its state combining embedding + $x^{(0)}$  
- Output $y^{(1)}$ used as next input $x^{(1)}$  
- Dimensionality of $y^{(t)}$ can be same or different as $x^{(t)}$  
- Both encoder and decoder are trained together using backpropagation

---

## RNN Output 

- Each RNN layer can produce multiple outputs simultaneously  
- One output generates label $y^{(t)}$  
- Another output can be used as next input $x^{(t)}$

---

## Traditional Seq2Seq Architecture

![Traditional Seq2Seq Architecture]()  
*(Replace with your figure 4 image)*

---

## Attention Mechanism

- Improves prediction accuracy  
- Adds parameters combining:  
  - Encoder outputs (all time step states)  
  - Current decoder state  
- Helps capture long-term dependencies better than gated or bidirectional RNNs

---

## Seq2Seq with Attention Architecture

![Seq2Seq with Attention](fig5.png)  
*(Replace with your figure 5 image)*

---

# Active Learning

- Active learning is a supervised learning paradigm  
- Useful when **labeling data is costly**  
- Common in medical, financial domains where expert annotation is needed  
- Start with few labeled examples + many unlabeled  
- Iteratively label only the most **informative** examples  

---

## Strategies of Active Learning

Two main strategies discussed here:  
1. Data density + uncertainty based  
2. Support vector machine (SVM) based

---

## Data Density & Uncertainty Based Strategy

- Train model $f$ on labeled data  
- For each unlabeled example $x$, compute importance score:  
  $$
  \text{importance}(x) = \text{density}(x) \times \text{uncertainty}_f(x)
  $$
- **Density**: How many examples surround $x$  
- **Uncertainty**: How unsure model $f$ is about $x$

---

## Uncertainty in Binary Classification

- Using sigmoid output:  
- Prediction close to 0.5 → high uncertainty  
- In SVM:  
  - Closer to decision boundary → more uncertain

---

## Uncertainty in Multiclass Classification

- Use **entropy** as uncertainty measure:  
  $$
  H_f(x) = - \sum_{c=1}^{C} Pr(y^{(c)}; f(x)) \ln Pr(y^{(c)}; f(x))
  $$
- Max entropy $= 1$ when all classes equally likely  
- Min entropy $= 0$ when model is certain about one class

---

## Density Computation

- Average distance from $x$ to its $k$ nearest neighbors  
- $k$ is a hyperparameter  
- High density → example lies in a well-populated region

---

## Active Learning Loop

1. Compute importance scores for unlabeled data  
2. Select example with highest score  
3. Ask expert to annotate it  
4. Add new labeled example to training set  
5. Retrain model  
6. Repeat until stopping criterion is met  

---

## Stopping Criteria

- Fixed budget (max number of expert queries)  
- Model performance threshold on a metric

---

## SVM-based Active Learning

- Train SVM on labeled data  
- Select unlabeled example closest to SVM hyperplane  
- Closest examples are most uncertain → highest potential to improve model  

---

## Other Active Learning Strategies

- **Cost-sensitive learning**: Consider cost of querying expert  
- **Query by Committee**:  
  - Train multiple models  
  - Query examples where models disagree the most  
- Select examples that reduce model **variance** or **bias** the most

---

# Semi-Supervised Learning (SSL)

- Small fraction of dataset is labeled  
- Majority of examples are unlabeled  
- Goal: Use unlabeled data to improve model without extra labeling effort

---

## Early SSL Methods: Self-Learning

- Train initial model on labeled data  
- Label unlabeled examples with model predictions  
- Add examples with confidence > threshold to training set  
- Retrain and repeat until stopping criterion met  
- Improvement often small; can sometimes degrade model

---

## Use of Neural Networks & SSL 

- Recent advances brought impressive results  
- Example: MNIST dataset  
  - 10 labeled examples per class → nearly perfect accuracy  
  - Total 70,000 labeled examples in MNIST  
- Key architecture: **Ladder Network**

---

## What is an Autoencoder?

- Feed-forward neural network with encoder-decoder  
- Trained to reconstruct input: pairs $(x, x)$  
- Bottleneck layer in middle compresses input to embedding  
- Decoder reconstructs input from embedding  
- Bottleneck usually smaller dimension than input  

---

- Cost function:  
  - Mean Squared Error (MSE) for continuous features  
  $$
  \frac{1}{N} \sum_{i=1}^N \|x_i - f(x_i)\|^2
  $$
  - Negative Log-Likelihood for binary features  

---

## Denoising Autoencoder

- Corrupt input $x$ by adding noise during training  
- Noise sampled from Gaussian distribution:  
  $$
  n^{(j)} \sim \mathcal{N}(\mu, \sigma^2)
  $$
- Train network to reconstruct original input from corrupted input

---

## Ladder Networks

- Autoencoder having same number of encoder and decoder layers  
- Bottleneck layer predicts label (softmax activation)  
- Multiple cost functions:  
  - Reconstruction cost for each layer $C_d^l$  
  - Classification cost $C_c$ for labeled examples  
- Optimize combined cost:  
  $$
  C_c + \sum_{l=1}^L \lambda^l C_d^l
  $$
- Noise added to input as well as encoder layers during training  

---

## Other SSL Approaches

- Clustering based:  
  - Build model on labeled data  
  - Cluster labeled + unlabeled examples together  
  - Predict label by majority vote in cluster  

- S3VM (Semi-Supervised SVM):  
  - Train SVMs on possible labelings of unlabeled data  
  - Choose model with largest margin  
  - Efficient algorithms avoid exhaustive search

----

# One-Shot Learning

- Important supervised learning paradigm 
- Commonly applied in **face recognition** 
- Eg: Recognize if two photos represent the **same person** or **different people**  


---

## Siamese Neural Network (SNN)

- Neural network architecture for one-shot learning  
- Can be CNN, RNN, or MLP  
- Key: training procedure, not just architecture

---

## Triplet Loss Function

- Training data: triplets $(A, P, N)$ 
  - $A$ : Anchor image  
  - $P$ : Positive image (same person as $A$)  
  - $N$ : Negative image (different person)  
- Model $f$ outputs embedding vectors of images  

---

## Triplet Loss Definition

$$
\text{loss} = \max \left( \|f(A_i) - f(P_i)\|^2 - \|f(A_i) - f(N_i)\|^2 + m, 0 \right)
$$

- $m$ : margin hyperparameter (> 0)  
- Intuition:  
  - Embeddings of $A$ and $P$ should be **close**  
  - Embeddings of $A$ and $N$ should be **far apart**

---

## Triplet Loss Objective

- Average loss over $N$ triplets:

$$
\frac{1}{N} \sum_{i=1}^N \max \left( \|f(A_i) - f(P_i)\|^2 - \|f(A_i) - f(N_i)\|^2 + m, 0 \right)
$$

- Optimize via backpropagation and gradient descent  

---

## Triplet Selection Strategy

- Random $N$ slows training (easy negatives)  
- Better: select $N$ close to $A$ and $P$ based on current model embeddings  
- Hard negatives encourage faster learning and better margins

---

## Training the SNN

- Decide on architecture (commonly CNN for images)  
- For each triplet in batch:  
  - Compute embeddings for $A, P, N$  
  - Calculate triplet loss  
- Update model parameters by backpropagation

---

## One-Shot Learning Misconception

- Not literally just one example per person needed for training  
- Called one-shot because:  
  - After training, only one example needed to identify a person (e.g., phone unlock)  
- Identification:  
  - Compare embeddings $f(A)$ and $f(\hat{A})$ 
  - If $\|f(A) - f(\hat{A})\|^2 < \tau$ (threshold), same person 

--- 

# Zero-Shot Learning

- Relatively new research area  
- No widely practical algorithms yet  
- Goal: Predict labels **not seen during training**  
- Common use: Labeling images with unseen classes

---

## Key Idea : 
### Embeddings for Inputs and Outputs

- Represent both input $x$ and label $y$ as embeddings  
- Word embeddings represent labels (e.g. English words)  
- Similar words have similar embeddings (e.g. Paris & Rome)  
- Dissimilar words have distant embeddings (e.g. Paris & potato)

---

## Word Embeddings

- Each dimension captures a semantic feature  
- Example with 4 dimensions (animalness, abstractness, sourness, yellowness) :
  - bee → $[1, 0, 0, 1]$  
  - yellow → $[0, 1, 0, 1]$  
  - unicorn → $[1, 1, 0, 0]$ 
- Usually, embeddings have 50–300 dimensions

---

## Training and Prediction in ZSL

- Replace label $y_i$ with its embedding during training  
- Train model $f$ to predict word embeddings from input $x$  
- For new input, predict embedding $\hat{y}$ 
- Find closest label by comparing $\hat{y}$ to embeddings of all words (cosine similarity)

---

## Why Does This Work?

- Example: zebra, clownfish, tiger  
  - Zebra: white, mammal, stripes  
  - Clownfish: orange, not mammal, stripes  
  - Tiger: orange, mammal, stripes  
- Model learns features (mammalness, color, stripes)  
- Can recognize tiger even if unseen in training, by matching learned features




