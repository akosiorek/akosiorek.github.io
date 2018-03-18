---
layout: draft
title:  "Normalizing Flows"
date:   2018-03-18 09:43:0 +0000
categories: notes
---
# Change of Variables in Probability Distributions
We can transform a probability distribution using an invertible mapping.
Let $$\mathbf{z} \in \mathbb{R}^d$$ be a random variable and $$f: \mathbb{R}^d \mapsto \mathbb{R}^d$$ an invertible smooth mapping. We cab use $$f$$ to transform $$\mathbf{z} \sim q(\mathbf{z})$$. The resulting random variable $$\mathbf{z}' = f(\mathbf{z})$$ has the following probability distribution:

$$
  q(\mathbf{z}') = q(\mathbf{z}) \left|
    \mathrm{det} \frac{
      \partial f^{-1}
    }{
      \partial \mathbf{z}\
    }
  \right|
  = q(\mathbf{z}) \left|
    \mathrm{det} \frac{
      \partial f
    }{
      \partial \mathbf{z}\
    }
  \right| ^{-1}. \tag{1}
$$

We can apply a series of mappings $$f_k$$, $$k \in {1, \dots, K}$$, with $$K \in \mathbb{N}_+$$ and obtain a normalizing flow, first introduced in [Variational Inference with Normalizing Flows](https://arxiv.org/abs/1505.05770),

$$
  \mathbf{z}_K = f_K \circ \dots \circ f_1 (\mathbf{z}_0), \quad \mathbf{z}_0 \sim q_0(\mathbf{z}_0), \tag{2}
$$

$$
  \mathbf{z}_K \sim q_K(\mathbf{z}_K) = q_0(\mathbf{z}_0) \prod_{k=1}^K
  \left|
    \mathrm{det} \frac{
      \partial f_k
    }{
      \partial \mathbf{z}_{k-1}\
    }
  \right| ^{-1}. \tag{3}
$$

This series of transformations can transform a simple probability distribution (*e.g.* Gaussian) into a complicated multi-modal one.
 To be of practical use, however, we can consider only transformations, whose determinants of Jacobians are easy to compute.
The original paper considered two simple family of transformations, named planar and radial flows.

# Simple Flows
## Planar Flow
$$
  f(\mathbf{z}) = \mathbf{z} + \mathbf{u} h(\mathbf{w}^T \mathbf{z} + b), \tag{4}
$$

with $$\mathbf{u}, \mathbf{w} \in \mathbb{R}^d$$ and $$b \in \mathbb{R}$$ and $$h$$ an element-wise non-linearity.
Let $$\psi (\mathbf{z}) = h' (\mathbf{w}^T \mathbf{z} + b) \mathbf{w}$$. The determinant can be easily computed as

$$
  \left| \mathrm{det} \frac{\partial f}{\partial \mathbf{z}} \right| =
  \left| 1 + \mathbf{u}^T \psi( \mathbf{z} ) \right|. \tag{5}
$$

We can think of it as slicing the $$\mathbf{z}$$-space with straight lines (or hyperlines), where each line contracts or expands the space around it.

## Radial Flow

$$
  f(\mathbf{z}) = \mathbf{z} + \beta h(\alpha, r)(\mathbf{z} - \mathbf{z}_0),
$$

with $$r = \Vert\mathbf{z} - \mathbf{z}_0\Vert_2$$, $$h(\alpha, r) = \frac{1}{\alpha + r}$$
and parameters $$\mathbf{z}_0 \in \mathbb{R}^d, \alpha \in \mathbb{R}_+$$ and $$\beta \in \mathbb{R}$$.

Similarly to planar flows, radial flows introduce spheres in the $$\mathbf{z}$$ space, which either contract or expand the space inside the sphere.

These simple flows are useful only for low dimensional spaces, since each transformation affects only a small volume in the original space. As the volume of the space grows exponentially with the number of dimensions $$d$$, we need a lot of layers in a high-dimensional space.

Another way to understand the need for many layers is to look at the form of the mappings. Each mapping behaves as a hidden layer of a neural network with one hidden unit and a skip connection. Since a single hidden unit is not very expresive, we need a lot of transformations.

# Autoregressive Flows
## Autoregressive Transformation
Let $$\mathbf{\mu} \in \mathbb{R}^d$$ and $$\mathbf{\sigma} \in \mathbb{R}^d_+$$.
Let $$\mathbf{\epsilon} \sim \mathcal{N} (\mathbf{0}, \mathbf{I}), \mathbf{\epsilon} \in \mathbb{R}^d$$ be a noise vector.
We can introduce complex dependencies between dimensions of a random variable $$\mathbf{z} \in \mathbb{R}^d$$ by specifying it in the following way.

$$ z_1 = \mu_1 + \sigma_1 \epsilon_1 \tag{6}$$

$$ z_i = \mu (\mathbf{z}_{1:i-1}) + \sigma (\mathbf{z}_{1:i-1}) \epsilon_i \tag{7}$$

Since each dimension depends only on the previous dimensions, the Jacobian of this transformation is a lower-triangular matrix with $$\sigma (\mathbf{z}_{1:i-1})$$ on the diagonal (to be derived later); the determinant is just a product of the terms on the diagonal.
We can sample $$\mathbf{\epsilon}$$ in parallel (since different dimensions are *i.i.d.*), but the transformation is inherently sequential. We need to compute all $$\mathbf{z}_{1:i-1}$$ before computing $$\mathbf{z}_i$$, which can be time consuming, and is therefore expensive to use as a parametrisation for the approximate posterior in VAEs.

This is an invertible transformation, and the inverse has the following form.

$$
  \epsilon_i = \frac{
    z_i - \mu (\mathbf{z}_{1:i-1})
  }{
    \sigma (\mathbf{z}_{1:i-1})
  } \tag{8}
$$

Given vectors $$\mathbf{\mu}$$ and $$\mathbf{\sigma}$$, we can vectorise the inverse transformations as

$$
  \mathbf{\epsilon} = \frac{
    \mathbf{z} - \mathbf{\mu} (\mathbf{z})
  }{
    \mathbf{\sigma} (\mathbf{z})
}. \tag{9}
$$

The Jacobian is again lower-triangular, with $$\mathbf{\sigma}^{-1}$$ on the diagonal.

The difference between the forward and the inverse transofrmations is that in the forward transformation, statistics used to transform every dimension depend on all the previously transformed dimensions. In the inverse transformation, the statistics used to invert $$\mathbf{z}$$ (which is the input), depend only on that input, and not on any result of the inversion.

## [Inverse Autoregressive Flow (IAF)](https://arxiv.org/abs/1606.04934)
IAF builds on equation (9). Let $$\mathbf{x} \in \mathbb{R}^D$$ be an observation and $$\mathbb{h} \in \mathbb{R}^n$$ a hidden state. We use a neural network $$h^\mathrm{enc}_\phi$$ to produce initial statistics and we sample a noise vector from a standard normal.

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

This reparametrisation is useful, because it avoids divisions, which can be numerically unstable. This type of autoregressive functions can be efficiently implemented using [MADE](https://arxiv.org/abs/1502.03509)-type neural networks.

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
  \log q_K (\mathbf{z}_K) = \log q(\epsilon) - \sum_{k=0}^K \sum_{i=1}^d \sigma_{k, i}.
$$

### Sampling and Density Evaluation
Sampling is easy, since we just sample $$\epsilon \sim q(\epsilon)$$ and then forward-transform it into $$\mathbf{z}_K$$. Each of the transformations gives us the vector $$\sigma_k$$, so that we can readily evaluate the probability of the sample $$q_K(\mathbf{z}_K)$$.

To evaluate the density of a sample not taken from $$q_K$$, we need to compute the chain of inverse transformations $$f^{-1}_k$$, $$k = K, \dots, 0$$. To do so, we have to sequentially compute

$$
  \mathbf{z}_{k-1, 1} = \frac{\mathbf{z}_{k, 1} - \mu_{k, 1}}{\sigma_{k, 1}},\\
  \mathbf{z}_{k-1, i} = \frac{\mathbf{z}_{k, i} - \mu_{k, i} (\mathbf{z}_{k-1, 1:i-1})}{\sigma_{k, i} (\mathbf{z}_{k-1, 1:i-1})}.
$$

This can be expensive, but as long as $$\mu$$ and $$\sigma$$ are implemented as autoregressive transformations, it is possible.

## [Masked Autoregressive Flow (MAF)](https://arxiv.org/abs/1705.07057)
# [Distribution distillation: MAF + IAF](https://arxiv.org/abs/1711.10433)
