---
marp: true
theme: gaia
title: Neural Networks and Deep Learning
math: mathjax

---

## Neural Networks

- Neural networks generalize logistic regression.
- Logistic regression and softmax regression are standard building blocks of NNs.
- Concepts like:
  - Linear regression
  - Logistic regression
  - Gradient descent  
  can help in understanding NNs

---

## What is a Neural Network ? 

- Just like regression and SVM, it is also a **function**:
  $$
  y = f_{NN}(x)
  $$
  (**a nested function**)

- Eg. 3-layer NN:
  $$
  y = f_3(f_2(f_1(x)))
  $$


---
## What is a layer?

- Each layer is also a function,
  $$
  f_l(z) = g_l(W_l z + b_l)
  $$
- Where:
  - $W_l$: weight matrix
  - $b_l$: bias vector
  - $g_l$: activation function (non-linear)

---

## Why Use a Matrix $W_l$?

- $g_l$ is a **vector function**
- $W_l$ must support multiple output units
- Each row $w_{l,u}$ in $W_l$ is a weight vector for one unit
- Each unit computes:
  $$
  a_{l,u} = w_{l,u}z + b_{l,u}
  $$
  $$
  \text{Output: } g_l(a_{l,u})
  $$

- A unit is called a neuron.

---

## Layer Output Vector

- For all units in a layer:
  $$
  [g_l(a_{l,1}), g_l(a_{l,2}), ..., g_l(a_{l,\text{size}_l})]
  $$
- This becomes the input to the next layer.

---

## Multilayer Perceptron (MLP)

- A type of Feed-Forward Neural Network (FFNN)
- Input: 2D feature vector
- Output: A number
- **Fully-connected**
- Also called: **Vanilla Neural Network**

---

## How Units Work

- Input → Vector
- Apply:
  1. **Linear transformation**: $w \cdot x + b$
  2. **Activation**: $g(w \cdot x + b)$
- Output sent to next layer’s all units (fully connected)

---

![w:1160](/Images/MLP_6.png)

- Each unit = circle or rectangle


---

## FFNN Output Layer

- Last layer decides task:
  - **Regression** → Linear activation
  - **Binary Classification** → Logistic activation

---

## Why Activation Functions Matter

- Without them → entire NN becomes **linear**
- Linear function of a linear function = linear
- Nonlinear activations let NNs learn complex patterns


## Common Activation Functions

### Logistic (Sigmoid)
$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

---

### TanH
$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

### ReLU
$$
\text{ReLU}(z) =
\begin{cases}
0 & \text{if } z < 0 \\
z & \text{otherwise}
\end{cases}
$$

---

## Layer Computation Recap

- Compute:
  $$
  a_l = W_l z
  $$
- Add bias:
  $$
  c_l = a_l + b_l
  $$
- Apply activation:
  $$
  y_l = g_l(c_l)
  $$

---
## What is Deep Learning?

- Training neural networks with **more than two non-output layers**.
- Earlier, training deep networks was difficult due to:
  - **Exploding gradient problem**
  - **Vanishing gradient problem**

---

## Exploding vs Vanishing Gradient

- **Exploding Gradient:**
  - Easier to handle
  - Techniques: Gradient clipping, L1/L2 regularization

- **Vanishing Gradient:**
  - More challenging
  - Causes very small gradients → parameters stop updating → training halts

---

## Why Vanishing Gradient Happens

- Neural networks are trained using **backpropagation**:
  - Uses the **chain rule** to compute gradients layer by layer.
- Traditional activations (e.g., hyperbolic tangent) have gradients in (0,1).
- Multiplying many small gradients across layers → gradient shrinks **exponentially** with depth.
- Result: Early layers train very slowly or not at all.

---

## Modern Solutions to Vanishing Gradient

- Use of **ReLU** activation, which reduces vanishing gradient effect.
- Architectures like **LSTM** and **Residual Networks** (with skip connections).
- These allow training of very deep networks (hundreds or thousands of layers).

- "Deep learning" now means training neural networks using modern techniques regardless of depth.

- **Hidden layers** = layers that are neither input nor output.

---

# Convolutional Neural Network (CNN)

### Why CNNs?

- MLPs grow **very fast** in parameters with more layers.
- Adding a 1000-unit layer adds **over 1 million parameters**.
- Image inputs are **high-dimensional**, making MLPs **hard to optimize**.
- CNNs reduce parameters drastically **without losing much accuracy**.
- Especially useful in **image and text processing**.

---

## Intuition Behind CNNs

- Nearby pixels in images often represent the **same type of info** (eg, sky, water, etc).
- Exceptions are edges where different objects meet.
- CNNs learn to detect **regions and edges** to recognize objects.
- Example: detecting skin regions + edges (blue colour) → likely a face on sky background.

---

## How CNNs Work: Moving Window Approach

- Split image into **small square patches**.
- Train multiple small regression models on patches.
- Each model detects a **specific pattern** (sky, grass, edges).
- Each model learns parameters of a **filter matrix F** (e.g., 3×3).

---

## CNN Layer Structure

- One CNN layer = multiple filters + biases.
- Filters **slide (convolve)** across the image.
- Convolution + bias → passed through **non-linearity (ReLU)**.
- Output: one matrix per filter → stacked as a **volume**.

## Multiple CNN Layers

- Next layer convolves the **volume** output of previous layer.
- Convolution on volume = sum of convolutions on individual matrices.

---

- Input images are often 3-channel volumes: **R, G, B**.

![h:550 w:800](/Images/CNN_6.png)

---

## Other CNN Features 
(Not Covered Here)
- **Strides** and **padding**: control filter sliding and image size.
- **Pooling**: reduces parameters by downsampling feature maps.

---

# Recurrent Neural Networks (RNNs)

## What is an RNN?
- A type of neural network used for **sequential data**.
- Handles **labeling**, **classification**, and **generation** of sequences.
- Commonly used in:
  - **Text processing**
  - **Speech recognition**
  - **Language modeling**

---

## Sequence Types
- **Labeling**: Predict a class for each time step.
- **Classification**: Predict a single class for the full sequence.
- **Generation**: Output a related sequence of possibly different length.

---

## How RNNs Work
- Not feed-forward: contains **loops**.
- Each unit has a **state** (memory) $h_{l,u}$.
- Each unit receives:
  - Output from previous layer $l - 1$
  - State from **same layer**, previous time step $t - 1$

---

![w:1100](/Images/RNN_6.png)

---
##  Example
Let input sequence be:
$$
X = [x^1, x^2, ..., x^t, ..., x^{length(x)}]
$$
- $x^t$ is a feature vector at time $t$
- Input is processed **one timestep at a time**
- If $X$ is a text sentence, then each feature vector $x^t$ represent a word in the sentence at position t.

---

## State Update Formula

For unit $u$ in layer $l$ :
$$
h_{l,u}^t = g_1(w_{l,u} · x_t + u_{l,u} · h_{l,u}^{t-1} + b_{l,u})
$$
- $g_1$ is usually $tanh$

Output:
$$
y_1^t = g_2(V_1 · h_1^t + c_{l,u})
$$
- $g_2$ is typically $softmax$

---

## Softmax Function
$$
\sigma(z) = [\sigma^{(1)}, ..., \sigma^{(D)}]
$$
$$
\sigma(j) = \frac {exp(z^{(j)})}{\sum_{k=1}^D exp(z^{(k)})} 
$$
- Generalization of sigmoid
- Produces probability distribution

---

## RNN Training
- Parameters: $w, u, b, V, C$
- Trained via **gradient descent**
- Use **Backpropagation Through Time (BPTT)**


## Problems with Vanilla RNNs
1. **Vanishing gradient** (especially with long sequences)
2. **Long-term dependencies** are hard to remember

---

## Solution - Gated RNNs 

- Two common types:
  - **LSTM (Long Short-Term Memory)**
  - **GRU (Gated Recurrent Unit)**
- Use **gates** to control memory

---

## GRU: Key Idea
- Store, read, and forget info using gates
- A GRU unit uses:
  - Input $x^t$
  - Memory from previous timestep $h_l^{t-1}$

---

## GRU Equations (Minimal Gated Unit)
$$
h_{l,u}^t = g_1(w_{l,u} · x^t + u_{l,u} · h_l^{t-1} + b_{l,u})
$$
$$
\tau_{l,u}^t = g_2(m_{l,u} · x^t + o_{l,u} · h^{t-1} + a_{l,u})
$$
$$
h_{l,u}^t = \tau_{l,u}^t h_{l}^t  + (1- \tau_{l,u}^t) h_{l}^{t-1}
$$

**GRU Output :-**
$$
h_l^t = [h_{l,1}^t, h_{l,2}^t, ..., h_{l,n}^t]
$$
$$
y_l^t = g_3(V_l · h_l^t + c_{l,u}) 
$$ 

---

($g_3$ is usually softmax)

## Why GRUs Work
- **Store info** for many timesteps
- **Control** read/write via sigmoid gates (values between 0 and 1)
- Avoid vanishing gradients (identity function is part of the design)

---

## Other RNN Variants
- **Bi-directional RNNs**
- **Attention-based RNNs**
- **Sequence-to-sequence (seq2seq)** models
- **Recursive** neural networks

---

## Applications of Sequence Models
- Language translation
- Chatbots
- Speech recognition
- Text summarization
