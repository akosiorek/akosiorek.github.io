---
layout: post
title:  "Attention in Neural Networks and How to Use It"
date:   2017-10-14 12:00:00 +0100
comments: True
categories: ML
---

Attention mechanisms in neural networks, otherwise known as *neural attention* or just *attention*, have recently attracted a lot of attention (pun intended). In this post, I will try to find a common denominator for different mechanisms and use-cases and I will describe two mechanisms of soft visual attention.

# What is Attention?
Informally, a neural attention mechanism equips a neural network with the ability to focus on a subset of its inputs (or features): it selects specific inputs. Let $$\mathbf{x} \in \mathcal{R}^d$$ an input, $$\mathbf{z} \in \mathcal{R}^k$$ a feature vector, $$\mathbf{a} \in [0, 1]^k$$ an attention vector and $$f_\phi(\mathbf{x})$$ an attention network. Typically, attention is implemented as

  $$
  \begin{align}
  \mathbf{a} &= f_\phi(\mathbf{x}), \tag{1} \label{att}\\
  \mathbf{z}_a &= \mathbf{a} \odot \mathbf{z},
  \end{align}
  $$

where $$\odot$$ is the Hadamard product.
We can talk about *soft attention*, which multiplies features with a soft mask of values between zero and one, or *hard attention*, when those values are constrained to be exactly zero or one, namely $$\mathbf{a} \in \{0, 1\}^k$$. In the latter case, we can use the hard attention mask to directly index the feature vector: $$\mathbf{z}_a = \mathbf{z}[\mathbf{a}]$$ (in Matlab notation), which changes its dimensionality.

To understand why attention is important, we have to think about what a neural network really is: a function approximator. Its ability to approximate different classes of functions depends on its architecture. A typical neural net is implemented as a chain of matrix multiplications and element-wise non-linearities, where elements of the input or feature vectors interact with each other only by addition.

Attention mechanisms compute a mask which is used to multiply features. This seemingly innocent extension has profound implications: suddenly, the space of functions that can be well approximated by a neural net is vastly expanded, making entirely new use-cases possible.

The above definition of attention as multiplicative interactions allow us to consider a broader class of models if we relax the constrains on the values of the attention mask and let $$\mathbf{a} \in \mathcal{R}^k$$. For example,  [Dynamic Filter Networks (DFN)](https://arxiv.org/abs/1605.09673) use a filter-generating network, which computes filters (or weights of arbitrary magnitudes) based on inputs, and applies them to features, which effectively is a multiplicative interaction. The only difference with soft-attention mechanisms is that the attention weights are not constrained to lie between zero and one. Going further in that direction, it would be very interesting to learn which interactions should be additive and which multiplicative, a concept explored in [A Differentiable Transition Between Additive and Multiplicative Neurons](https://arxiv.org/abs/1604.03736). The excellent [distill blog](https://distill.pub/2016/augmented-rnns/) provides a great overview of soft-attention mechanisms.

# Visual Attention
Attention can be applied to any kind of inputs, regardless of their shape. In the case of matrix-valued inputs, such as images, we can talk about *visual attention*. Let $$\mathbf{I} \in \mathcal{R}^{H \times W}$$ be an image and $$\mathbf{g} \in \mathcal{R}^{h \times w}$$ an attention glimpse *i.e.* the result of applying an attention mechanism to the image $$\mathbf{I}$$.

**Hard attention** for images has been known for a very long time: image cropping. It is very easy conceptually, as it only requires indexing. Hard-attention can be implemented in Python (or Tensorflow) as

```python
g = I[y:y+h, x:x+w]
```

The only problem with the above is that it is non-differentiable; to learn the parameters of the model, one must resort to *e.g.* the score-function estimator, briefly mentioned in my [previous post](http://localhost:4000/ml/2017/09/03/implementing-air.html).

**Soft attention**, in its simplest variant, is no different for images than for vector-valued features and is implemented exactly as in equation \ref{att}. One of the early uses of this types of attention comes from the paper called [Show, Attend and Tell](https://arxiv.org/abs/1502.03044): ![aa](https://distill.pub/2016/augmented-rnns/assets/show-attend-tell.png)
The model learns to *attend* to specific parts of the image while generating the word describing that part.

This type of soft attention is computationally wasteful, however. The blacked-out parts of the input do not contribute to the results but still need to be processed. It is also over-parametrised: sigmoid activations that implement the attention are independent of each other; it can select multiple objects at once. In practice, however, we would like the attention mechanism to be selective. The two following mechanisms, introduced by [DRAW](https://arxiv.org/abs/1502.04623) and [Spatial Transformer Networks](https://arxiv.org/abs/1506.02025), respectively, solve this issue. They can also resize the input, leading to further potential gains in performance.

**Gaussian Attention**

**Spatial Transformer**
