---
layout: draft
title:  "IAF"
date:   2018-03-18 09:43:0 +0000
categories: ml
---

# Autoregressive Flows
Enhancing expressivity of normalising flows is not easy, since we are constrained by functions, whose Jacobians are easy to compute.
It turns out, though, that we can introduce dependencies between different dimensions of the latent variable, and still end up with a tractable Jacobian.
Namely, if after a transformation, the dimension $$i$$ of the resulting variable depends only on dimensions $$1:i$$ of the input variable, then the Jacobian of this transformation is triangular.
As we know, a Jacobian of a triangular matrix is equal to the product of the terms on the diagonal.
More formally, let $$J \in \mathcal{R}^{d \times d}$$ be the Jacobian of the mapping $$f$$, then

$$
  \mathbf{y}_i = f(\mathbf{z}_{1:i}),
  \qquad J = \frac{\partial \mathbf{y}}{\partial \mathbf{z}}, \tag{7}
$$

$$
  \det{J} = \prod_{i=1}^d J_{ii}. \tag{8}
$$

There are three interesting flows that use the above observation, albeit in different ways, and arrive at mappings with very different properties.

## [Real Non-Volume Preserving Flows (R-NVP)](https://arxiv.org/abs/1605.08803)
R-NVPs are arguably the least expressive but the most generally applicable of the three.
Let $$1 < k < d$$, $$\circ$$ element-wise multiplication and $$\mu$$ and $$\sigma$$ two mappings $$\mathcal{R}^k \mapsto \mathcal{R}^{d-k}$$. R-NVPs are defined as:

$$
  \mathbf{y}_{1:k} = \mathbf{z}_{1:k},\\
  \mathbf{y}_{k+1:d} = \mathbf{z}_{k+1:d} \circ \sigma(\mathbf{z}_{1:k}) + \mu(\mathbf{z}_{1:k}). \tag{9}
$$

It is an autoregressive transformation, although not as general as equation (7) allows.
It copies the first $$k$$ dimensions, while shifting and scaling all the remaining ones.
The first part of the Jacobian (up to dimension $$k$$) is just an identity matrix, while the second part is lower-triangular with $$\sigma(\mathbf{z}_{1:k})$$ on the diagonal.
Hence, the determinant of the Jacobian is

$$
  \frac{\partial \mathbf{y}}{\partial \mathbf{z}} = \prod_{i=1}^{d-k} \sigma_i(\mathbf{z}_{1:k}). \tag{10}
$$

R-NVPs are particularly attractive, because both sampling and evaluating probability of some external sample are very efficient.
Computational complexity of both operations is, in fact, exactly the same.
This allows to use R-NVPs as a parametrisation of an approximate posterior $$q$$ in VAEs, but also as the output likelihood (in VAEs or general regression models).
To see this, first note that we can compute all elements of $$\mu$$ and $$\sigma$$ in parallel, since all inputs ($$\mathbf{z}$$) are available.
We can therefore compute $$\mathbf{y}$$ in a single forward pass.
Next, note that the inverse transformation has the following form, with all divisions done element-wise,

$$
\mathbf{z}_{1:k} = \mathbf{y}_{1:k},\\
\mathbf{z}_{k+1:d} = (\mathbf{y}_{k+1:d} - \mu(\mathbf{y}_{1:k}))~/~\sigma(\mathbf{y}_{1:k}). \tag{11}
$$

The original paper applies several layers of this mapping.
The authors also reverse the ordering of dimensions after every step.
This way, variables that are just copied in one step, are transformed in the following step.

## Autoregressive Transformation
We can be even more expressive than R-NVPs, but we pay a price.
Here's why.

Let $$\mathbf{\mu} \in \mathbb{R}^d$$ and $$\mathbf{\sigma} \in \mathbb{R}^d_+$$.
Let $$\mathbf{\epsilon} \sim \mathcal{N} (\mathbf{0}, \mathbf{I}), \mathbf{\epsilon} \in \mathbb{R}^d$$ be a noise vector.
We can introduce complex dependencies between dimensions of a random variable $$\mathbf{z} \in \mathbb{R}^d$$ by specifying it in the following way.

$$ z_1 = \mu_1 + \sigma_1 \epsilon_1 \tag{12}$$

$$ z_i = \mu (\mathbf{z}_{1:i-1}) + \sigma (\mathbf{z}_{1:i-1}) \epsilon_i \tag{13}$$

Since each dimension depends only on the previous dimensions, the Jacobian of this transformation is a lower-triangular matrix with $$\sigma (\mathbf{z}_{1:i-1})$$ on the diagonal (to be derived later); the determinant is just a product of the terms on the diagonal.
We can sample $$\mathbf{\epsilon}$$ in parallel (since different dimensions are *i.i.d.*), but the transformation is inherently sequential. We need to compute all $$\mathbf{z}_{1:i-1}$$ before computing $$\mathbf{z}_i$$, which can be time consuming, and is therefore expensive to use as a parametrisation for the approximate posterior in VAEs.

This is an invertible transformation, and the inverse has the following form.

$$
  \epsilon_i = \frac{
    z_i - \mu (\mathbf{z}_{1:i-1})
  }{
    \sigma (\mathbf{z}_{1:i-1})
  } \tag{14}
$$

Given vectors $$\mathbf{\mu}$$ and $$\mathbf{\sigma}$$, we can vectorise the inverse transformations as

$$
  \mathbf{\epsilon} = \frac{
    \mathbf{z} - \mathbf{\mu} (\mathbf{z})
  }{
    \mathbf{\sigma} (\mathbf{z})
}. \tag{15}
$$

The Jacobian is again lower-triangular, with $$\mathbf{\sigma}^{-1}$$ on the diagonal.

The difference between the forward and the inverse transofrmations is that in the forward transformation, statistics used to transform every dimension depend on all the previously transformed dimensions. In the inverse transformation, the statistics used to invert $$\mathbf{z}$$ (which is the input), depend only on that input, and not on any result of the inversion.


## [Inverse Autoregressive Flow (IAF)](https://arxiv.org/abs/1606.04934)
IAF builds on equation (14). Let $$\mathbf{x} \in \mathbb{R}^D$$ be an observation and $$\mathbf{h} \in \mathbb{R}^n$$ a hidden state. We use a neural network $$h^\mathrm{enc}_\phi$$ to produce initial statistics and we sample a noise vector from a standard normal.

$$
\begin{align}
  \mathbf{\mu}_0, \mathbf{\sigma}_0, \mathbf{h}_0 = &h^\mathrm{enc}_\phi ( \mathbf{x} )\\
  \mathbf{z}_0 = \mathbf{\mu}_0 + \mathbf{\sigma}_0 \mathbf{\epsilon},
  \quad &\mathbf{\epsilon} \sim \mathcal{N} (\mathbf{0}, \mathbf{I})
\end{align} \tag{10}
$$

We then use another neural network $$R^k_\phi$$ (or a series of them), to parametrise subsequent transformations,

$$
\begin{align}
  \mathbf{\mu}_k, \mathbf{\sigma}_k, \mathbf{h}_k &= R^k_\phi ( \mathbf{z}_{k-1}, \mathbf{h}_{k-1} ),\\
  \mathbf{z}_k &= \mathbf{\mu}_k + \mathbf{\sigma}_k \mathbf{z}_{k-1}.
\end{align} \tag{11}
$$

Even though the second line of equations (10) and (11) looks more like equation (7), they are really reparamtrised versions of equation (9) - the inverse autoregressive transformation.
To see this, set $$\mathbf{z}_k = \epsilon$$, $$\mathbf{z}_{k-1} = \mathbf{z}$$, $$\mu_k = -\frac{\mu(\mathbf{z})}{\sigma(\mathbf{z})}$$ and $$\sigma_k = \frac{1}{\sigma(\mathbf{z})}$$. Substitution gives us:

$$
  \mathbf{z}_k = \mathbf{\mu}_k + \mathbf{\sigma}_k \mathbf{z}_{k-1}
  \implies
  \epsilon = -\frac{\mu(\mathbf{z})}{\sigma(\mathbf{z})} + \frac{1}{\sigma(\mathbf{z})}\mathbf{z} =
  \frac{\mathbf{z} - \mu(\mathbf{z})}{\sigma(\mathbf{z})} = (7).
$$

This reparametrisation is useful, because it avoids divisions, which can be numerically unstable. This type of autoregressive functions can be efficiently implemented using [MADE](https://arxiv.org/abs/1502.03509)-type neural networks, which is nicely explained in [this blog post](s).

To understand how does this transformation affect the distribution of $$\mathbf{z}$$, we can compute the resulting probability density function.

$$ \log q( \mathbf{\epsilon} ) = \log \mathcal{N} (\mathbf{\epsilon} \mid \mathbf{0}, \mathbf{I}) = - \sum_{i=1}^d \left(
  \log \epsilon_i + \frac{1}{2} \log 2 \pi \right)
  = - \frac{d}{2} \log 2 \pi - \sum_{i=1}^d \log \epsilon_i
$$

To factor in subsequent transformations, we need to compute all the Jacobians.

$$
  \frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}}
  = \underbrace{
    \frac{\partial \mu_k}{\partial \mathbf{z}_{k-1}}
  + \frac{\partial \sigma_k}{\partial \mathbf{z}_{k-1}} \mathrm{diag} ( \mathbf{z}_{k-1} )
}_\text{lower triangular with zeros on the diagonal}
  + \mathrm{diag}( \sigma_k )
  \underbrace{
    \frac{\partial \mathbf{z}_{k-1}}{\partial \mathbf{z}_{k-1}}
  }_{= \mathbf{I}}
$$

If $$\mu_k = \mu_k ( \mathbf{z}_{k-1})$$ and $$\sigma_k = \sigma_k ( \mathbf{z}_{k-1})$$ are implemented as autoregressive transofrmations (with respect to $$\mathbf{z}_{k-1}$$), then the first two terms in the Jacobian above are lower triangular matrices with zeros on the diagonal.
The last term is a diagonal matrix, with $$\sigma_k$$ on the diagonal.
Thus, the determinant of the Jacobian is just

$$
  \mathrm{det} \left( \frac{\partial \mathbf{z}_k}{\partial \mathbf{z}_{k-1}} \right) = \prod_{i=1}^d \sigma_{k, i}.
$$

Therefore, the final log-probability can be written as

$$
  \log q_K (\mathbf{z}_K) = \log q(\epsilon) - \sum_{k=0}^K \sum_{i=1}^d \log \sigma_{k, i}.
$$

### Sampling and Density Evaluation
Sampling is easy, since we just sample $$\epsilon \sim q(\epsilon)$$ and then forward-transform it into $$\mathbf{z}_K$$. Each of the transformations gives us the vector $$\sigma_k$$, so that we can readily evaluate the probability of the sample $$q_K(\mathbf{z}_K)$$.

To evaluate the density of a sample not taken from $$q_K$$, we need to compute the chain of inverse transformations $$f^{-1}_k$$, $$k = K, \dots, 0$$. To do so, we have to sequentially compute

$$
  \mathbf{z}_{k-1, 1} = \frac{\mathbf{z}_{k, 1} - \mu_{k, 1}}{\sigma_{k, 1}},\\
  \mathbf{z}_{k-1, i} = \frac{\mathbf{z}_{k, i} - \mu_{k, i} (\mathbf{z}_{k-1, 1:i-1})}{\sigma_{k, i} (\mathbf{z}_{k-1, 1:i-1})}.
$$

This can be expensive, but as long as $$\mu$$ and $$\sigma$$ are implemented as autoregressive transformations, it is possible.
