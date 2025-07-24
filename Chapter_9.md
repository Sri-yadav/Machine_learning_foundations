---
marp: true
theme: gaia
titile: 
math: mathjax

---

# Unsupervised Learning

- No labels in the dataset  
- Difficult to evaluate model quality  
- We'll focus on methods that allow **data-driven evaluation**

---

## Density Estimation?

- **Aim:** Estimate the **Probability Density Function (PDF)** of the data source  
- **Useful for:**
  - Novelty detection
  - Intrusion detection
- Example: One-class classification using **Multivariate Normal Distribution (MVN)**

---

## Parametric vs Non-Parametric

- **Parametric models**:  
  Assume specific distributions (e.g., MVN)  
  - Poor performance if assumption is wrong
- **Non-parametric models**:  
  Make fewer assumptions  
  - Flexible, more adaptable  
  - Used in kernel regression and density estimation

---

## Kernel Density Estimation (KDE)

Given a dataset $\{x_i\}_{i=1}^N$, KDE estimates the PDF as:

$$
\hat{f}_b(x) = \frac{1}{Nb} \sum_{i=1}^{N} k\left(\frac{x - x_i}{b}\right)
$$

- $b$ : bandwidth (bias-variance control)
- $k(z)$: kernel (e.g., Gaussian)

---

## Gaussian Kernel

The Gaussian kernel is defined as:

$$
k(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2 / 2}
$$

- Smooth and commonly used  
- Bandwidth $b$ adjusts smoothing level

---

## Bias–Variance Trade-off : MISE

To find best $b$, minimize **Mean Integrated Squared Error (MISE)**:

$$
\text{MISE}(b) = \mathbb{E} \left[ \int_{\mathbb{R}} \left( \hat{f}_b(x) - f(x) \right)^2 dx \right]
$$

- Measures expected squared error across all datasets  
- Generalization of mean squared error to continuous functions

---

## MISE Expansion

$$
\text{MISE}(b) = \mathbb{E} \left[ \int \hat{f}_b^2(x) dx - 2 \int \hat{f}_b(x)f(x) dx + \int f^2(x) dx \right]
$$

- Last term $\int f^2(x) dx$ is **independent of $b$**  
- So we minimize:

$$
\int \hat{f}_b^2(x) dx - \frac{2}{N} \sum_{i=1}^{N} \hat{f}_b^{(i)}(x_i)
$$

---

## Leave-One-Out Estimate

- $\hat{f}_b^{(i)}(x_i)$ : density estimated **excluding $x_i$**  
- Called **Leave-One-Out Cross-Validation**  
- Gives an **unbiased estimate** of the expected value

---

## Finding Optimal $b^*$

Minimize cost:

$$
\text{Cost}(b) = \int \hat{f}_b^2(x) dx - \frac{2}{N} \sum_{i=1}^{N} \hat{f}_b^{(i)}(x_i)
$$

- Use **grid search** to test multiple $b$ values  
- Choose $b^*$ that minimizes the cost

---

## KDE in Higher Dimensions

For $\mathbf{x} \in \mathbb{R}^D$:

$$
\hat{f}_b(\mathbf{x}) = \frac{1}{Nb^D} \sum_{i=1}^{N} k\left( \frac{\lVert \mathbf{x} - \mathbf{x}_i \rVert}{b} \right)
$$

- Use **Euclidean distance**
- Same concept extends to multidimensional data

---

## Visualizing Bandwidth Effect

- Dataset with 100 points  
- KDE with:
  - Small $b$ : too spiky (overfitting)  
  - Large $b$ : too smooth (underfitting)  
  - Optimal $b^*$ : balanced and best fit

# FIXME: ADD PHOTOS

---

# Clustering 

- **Goal :** **Assign labels** to examples without labeled data  
- Challenge: No ground truth → Hard to evaluate model quality  
- Performance depends on unknown properties of the underlying data distribution

---

## Common Clustering Algorithms

- Many algorithms exist:
  - Centroid-based (e.g., K-Means)
  - Density-based (e.g., DBSCAN, HDBSCAN)
- No one-size-fits-all:  
  - Each algorithm behaves differently on different datasets

---

# K-Means Clustering

## Algorithm Overview

1. Choose number of clusters $k$
2. Randomly initialize $k$ **centroids** in feature space  
3. Assign each example to its **nearest centroid**  
4. Update each centroid to be the **mean of its assigned examples**  
5. Repeat steps 3–4 until assignments no longer change

---

- Final model = **Cluster assignments** (centroid IDs)
- Sensitive to **initial centroid positions**  
  - Different runs may give different results  
- Clusters formed are **spherical in shape**

# FIXME: are u sure ?

---

## Tuning $k$

- $k$ is a **hyperparameter**: must be chosen manually
- No optimal method exists  
  - Use heuristics, visual inspection, or evaluation metrics  
- Later in the chapter: technique to choose $k$ **without looking at the data**

# FIXME: check it again.
---

# DBSCAN and HDBSCAN

## DBSCAN: Density-Based Clustering

- No need to specify $k$ !
- Instead, define two hyperparameters:
  - $\epsilon$
  - $n$ (min. neighbors)

---

## How It Works

1. Pick a random example $x$
2. If $x$'s $\epsilon$ -neighborhood has ≥ $n$ points → start cluster
3. Expand cluster by checking neighbors' neighborhoods recursively
4. Repeat for unvisited points
5. Points with < $n$ neighbors → **outliers**

---

## DBSCAN: Pros and Cons

**Advantages**:
- Can form **arbitrarily shaped** clusters  
- Automatically detects **outliers**

**Limitations**:
- Needs **good choice** of $\epsilon$ and $n$  
- Cannot handle **varying density** across clusters

---

## HDBSCAN

- Extension of DBSCAN  
- Removes need to tune $\epsilon$ 
- Can handle **clusters with different densities**

- Only one key hyperparameter:  
  → $n$ : **Minimum samples per cluster**
- Intuitive to choose  
- Works well on **large datasets**  
- Modern **K-Means is faster**, but HDBSCAN offers better flexibility and robustness.

---

 For most practical tasks, **try HDBSCAN first**  
 - Robust  
 - Scales to millions of examples  
 - Minimal tuning required

---

# Number of Clusters?

- Key question: **What is the correct number of clusters $k$ ?**
- For **1D–3D** data:  
  - Can visually inspect “clouds” or groupings  
- For **D > 3**:  
  - Visual inspection becomes difficult → need formal methods

---

## Prediction Strength : Core Idea

- Inspired by **supervised learning**
- Steps:
  1. Split data into **training** $S_{\text{tr}}$ and **test** $S_{\text{te}}$
  2. Choose a value for $k$
  3. Run clustering algorithm $C$ on both sets:
     - $C(S_{\text{tr}}, k)$
     - $C(S_{\text{te}}, k)$

---

## Building the Co-Membership Matrix

Let:

- $A = C(S_{\text{tr}}, k)$ : training clustering
- Define matrix $D[A, S_{\text{te}}]$ of size $N_{\text{te}} \times N_{\text{te}}$

$$
D[A, S_{\text{te}}](i, i') =
\begin{cases}
1 & \text{if } x_i, x_{i'} \text{ from test set are in same cluster under } A \\
0 & \text{otherwise}
\end{cases}
$$

---

- Compare clusterings from test set and training set
- If **cluster structure is stable**, test set examples that cluster together will map to same training clusters
- If not, many zeros in $D[A, S_{\text{te}}]$

---

## Prediction Strength Formula

$$
ps(k) = \min_{j = 1, ..., k} \frac{1}{|A_j|(|A_j| - 1)} \sum_{i, i' \in A_j} D[A, S_{\text{te}}](i, i')
$$

Where:
- $A = C(S_{\text{tr}}, k)$
- $A_j$ : $j^{th}$ cluster from test set clustering
- $|A_j|$ : number of examples in cluster $A_j$

---

## Intuition Behind $ps(k)$

- For each cluster $A_j$ in test set:
  - Measure fraction of point pairs that are **co-assigned** in training clustering
- Take **minimum value** across all clusters
- High prediction strength → stable clustering at that $k$

---

## Selecting the Best $k$

- Choose **largest** $k$ such that:

$$
ps(k) > 0.8
$$

- If $ps(k)$ drops below threshold → clustering becomes unreliable

---

## Multiple Runs for K-Means

- K-means is **non-deterministic**  
  - Different runs may give different results
- Solution:  
  - Run multiple times for same $k$
  - Compute **average prediction strength** :

$$
\bar{ps}(k) = \text{mean of } ps(k) \text{ over runs}
$$

---

## Other Methods to Estimate $k$

- **Gap Statistic**:
  - Compare within-cluster dispersion to random distribution
- **Elbow Method**:
  - Plot error vs. $k$ → look for "elbow"
- **Average Silhouette Score**:
  - Measures how similar a point is to its own cluster vs others

---

## Best Practices

- Use **Prediction Strength** or **Gap Statistic** for reliability  
- Run clustering **multiple times** if algorithm is non-deterministic  
- Avoid relying solely on visual inspection for **high-dimensional data**

---

# Other Clustering Algorithms

- So far : K-means and DBSCAN → **hard clustering**
  - Each point assigned to **only one cluster**
- Now: Methods that support **soft clustering**
  - Points can belong to **multiple clusters** with **probabilities**
- Example: **Gaussian Mixture Models (GMMs)**  
  (Also supported by **HDBSCAN**)

---

## Gaussian Mixture Models (GMMs)

- **Idea :** Data is generated from a mixture of **multiple Gaussian distributions**
- Each distribution represents a **cluster**
- Probability model:
  
$$
f_X(x) = \sum_{j=1}^{k} \pi_j \cdot f_{\mu_j, \sigma_j}(x)
$$

- $\pi_j$: weight (mixing coefficient) for cluster $j$  
- $f_{\mu_j, \sigma_j}$: Gaussian PDF with mean $\mu_j$ and variance $\sigma_j^2$

---

## Estimating GMM Parameters

- Use **Expectation-Maximization (EM)** algorithm to estimate:
  - Means $\mu_j$
  - Variances $\sigma_j^2$
  - Weights $\pi_j$

## EM Algorithm $(1D, k = 2)$ – Initialization

- Start with:
  - Initial guesses for $\mu_1, \sigma_1^2, \mu_2, \sigma_2^2$
  - Set $\pi_1 = \pi_2 = \frac{1}{2}$

---

## EM Algorithm Steps

### Step 1: Likelihood Calculation

$$
f(x_i | \mu_j, \sigma_j^2) = \frac{1}{\sqrt{2\pi\sigma_j^2}} \exp\left( -\frac{(x_i - \mu_j)^2}{2\sigma_j^2} \right)
$$

---

### Step 2: Posterior (Soft Assignment)

- For each $x_i$, compute:

$$
\beta_i^{(j)} = \frac{f(x_i | \mu_j, \sigma_j^2) \cdot \pi_j}{\sum_{l=1}^{k} f(x_i | \mu_l, \sigma_l^2) \cdot \pi_l}
$$

- $\beta_i^{(j)}$ : probability that point $x_i$ belongs to cluster $j$

---

### Step 3: Update Cluster Means & Variances

$$
\mu_j = \frac{\sum_{i=1}^{N} \beta_i^{(j)} x_i}{\sum_{i=1}^{N} \beta_i^{(j)}}, \quad
\sigma_j^2 = \frac{\sum_{i=1}^{N} \beta_i^{(j)} (x_i - \mu_j)^2}{\sum_{i=1}^{N} \beta_i^{(j)}}
$$

---

### Step 4: Update Cluster Weights

$$
\pi_j = \frac{1}{N} \sum_{i=1}^{N} \beta_i^{(j)}
$$

- Repeat steps **1–4** until convergence (e.g., small change in parameters)    |

---

## GMM : Multidimensional Case

- For $D > 1$ : use **multivariate normal distributions (MNDs)**
- Variance $\sigma^2$ → becomes **covariance matrix**
  - Controls **shape**, **elongation**, **orientation** of clusters
- Advantage: Clusters can be **elliptical**, not just circular

---

## Choosing Number of Clusters in GMM

- No universal method — typical approach:
  1. Split into training and test sets
  2. For each $k$, train GMM on training data → $f_k^{\text{tr}}$
  3. Compute **likelihood** of test data under the model:

$$
\text{Choose } k \text{ that maximizes } \prod_{i=1}^{N_{\text{te}}} f_k^{\text{tr}}(x_i)
$$

---

## Other Notable Clustering Algorithms

- **HDBSCAN**  
  - Like DBSCAN, but handles variable density  
  - Supports **soft assignments**

- **Spectral Clustering**  
  - Based on graph Laplacian  
  - Great for **non-convex** clusters

- **Hierarchical Clustering**  
  - Builds a **tree of clusters**  
  - Agglomerative (bottom-up) or divisive (top-down)

---

# Dimensionality Reduction 

## Why we need it?

- Modern ML algorithms (e.g., neural networks, ensembles) can handle millions of features.
- Still useful in:
  - **Visualization**: Humans can only visualize up to 3D.
  - **Model Interpretability**: Needed when using simple models like linear regression or decision trees.
  - **Noise Reduction**: Helps remove redundancy and highly correlated features.

---

## Common Dimensionality Reduction Techniques

- **Principal Component Analysis (PCA)**
- **Uniform Manifold Approximation and Projection (UMAP)**
- **Autoencoders** (already covered)

---

## PCA – Principal Component Analysis

- Projects data into new coordinate axes called **principal components**.
- First axis: direction of **highest variance** in data.
- Second axis: **orthogonal** to the first, second-highest variance, and so on.

- Reduces dimensionality while preserving most variation.
- Commonly, 2 or 3 components capture most information.


# FIXME: Add photos

---

## UMAP – Intuition

- Designed for visualization.
- Like t-SNE, it preserves **local structure** of data.
- Defines a similarity metric combining:
  - Euclidean distance
  - Local density (distance to nearest neighbors)

---

## UMAP – Similarity Metric

Similarity between two points:

$$
w(x_i,x_j) = w_i(x_i,x_j) + w_j(x_j,x_i) - w_i(x_i,x_j) \times w_j(x_j,x_i)
$$

Where:
$$
w_i(x_i,x_j) = exp\left(\frac{ - d(xi,xj) - ρi} {σi }\right)
$$
- $ρ_i$ : distance to closest neighbor
- $σ_i$ : distance to $k^{th}$ neighbor (hyperparameter)

---

## UMAP – Optimization

- Let **w** be similarities in high-dim space, and **w'** in low-dim.
- Objective: minimize **cross-entropy** between w and w′:

$$
C(w, w′) = \sum_{i=1}^{N} \sum_{j=1}^N w(x_i, x_j) ln \left(\frac {w(x_i, x_j)}{w′(x'_i, x'_j)} \right) + \left(1 - w(x_i, x_j)\right)  ln\left(\frac {1- w(x_i, x_j)}{1 - w′(x'_i, x'_j)} \right)
$$

- Minimized using **gradient descent**.
- Outputs low-dimensional representation x′ for each input x.

---

## UMAP – Example

- Applied to **MNIST** handwritten digits dataset.
- 70,000 labeled examples (10 digits).
- Each point colored by digit class.
- UMAP separates clusters visually **without using labels**.

# FIXME: Comparision photos

---

# Outlier Detection 
## Introduction

- **Goal:** Identify examples very **different** from typical ones.
- **Useful in:**
  - Fraud detection
  - Anomaly monitoring
  - Quality control

---

## Outlier Detection Techniques

1. **Autoencoder-Based**
   - Train on normal data.
   - Outliers are **poorly reconstructed**.
2. **One-Class Classification**
   - Model learns to recognize “normal”.
   - Any deviation = outlier.

---

## Autoencoder for Outliers

- Autoencoder is trained on all examples.
- Compresses and reconstructs.
- For **normal** examples: low reconstruction error.
- For **outliers**: high reconstruction error.

---

## One-Class Classifier

- Learns a decision boundary around **normal** data.
- At test time:
  - If input is inside boundary → accepted.
  - Else → flagged as outlier.

---



