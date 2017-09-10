---
layout: post
title:  "Conditional KL-divergence and VAEs"
date:   2017-09-09 14:57:00 +0100
comments: True
categories: ML
---

Inference is hard and often computationally expensive. Variational Autoencoders (VAE) lead to an efficient amortised inference scheme. What amortised means is that once the model is trained (which can take a long time), we can deploy it and do constant-time inference.
Variational Autoencoders (VAE) learn the approximate posterior distribution $$q(z\mid x)$$ over some latent variables $$z$$ by maximising a lower bound on the true data likelihood $$p(x)$$. This is useful, because the latent variables explain what we see ($$x$$), and often in a concise form.

One problem with VAEs is that we have to assume some functional form for $$q$$, and most works just take the Gaussian distribution with a diagonal covariance matrix. While it is often ok, several papers have shown that we can improve the quality of the model by making the approximate posterior more complex: *e.g.* multi-modal, see [the normalizing flows paper](https://arxiv.org/abs/1505.05770).

Normalizing flows take a simple probability distribution (here: a Gaussian) and apply a series of invertible transformations to get a more complicated distribution. Another way of achieving the same goal is to split the latent variables into two groups $$z = \{u, v\}$$, say, and express the joint distribution as $$q(z) = q(u, v) = q(u \mid v) q(v)$$ by using the product rule of probability. The conditional distribution $$q(u \mid v)$$ can depend on $$v$$ in a highly non-linear fashion. Even though both the marginal $$q(v)$$ and the conditional $$q(u \mid v)$$ can be Gaussians, their joint might be highly non-Gaussian. Consider the below example and the resulting density plot (in the plot $$x=v$$ and $$y=u$$).

$$\begin{aligned}
q(v) &= \mathcal{N} (v \mid 0, I)\\
q(u \mid v) &= \mathcal{N} (u \mid Fv, Fvv^TF^T + \beta I),
\end{aligned}$$

<img src="true_distrib.png" style="width: 500px; display: block; margin: auto;">

This simple scheme can lead to relatively complicated posterior distributions, but it leads to a slightly more complicated optimisation procedures. Let me elaborate.

As part of the variational objective, we are optimising the
Kullback-Leibler divergence $$KL[q \mid p]$$ between the approximate posterior $$q$$ and a prior over the latent variables $$p$$. KL is an asymmetric measure of similarity between two probability distributions $$q$$ and $$p$$ that is often used in machine learning. It can be interpreted as the information gain from using $$q$$ instead of $$p$$, or in the context of coding theory, the extra number of bits to code samples from $$q$$ by using $$p$$. You can read more about information measures [here](http://threeplusone.com/on_information.pdf). It is defined as

   $$
   KL[q(z) \mid \mid p(z)] = \int q(z) \log \frac{q(z)}{p(z)} \mathrm{d}z.
   $$

If we split the random variable $$z$$ into two disjoint sets $$z = \{u, v\}$$ as  above, the KL factorises as

   $$
   \begin{aligned}
   KL[q(u, v) \mid \mid p(u, v)] &= \iint q(u, v) \log \frac{q(u, v)}{p(u, v)} \mathrm{d}u \mathrm{d}v\\
   % sum of integrals
   &= \int q(v) \log \frac{q(v)}{p(v)} \mathrm{d}v
   + \int q(v) \int q(u \mid v) \log \frac{q(u \mid v)}{p(u \mid v))} \mathrm{d}u \mathrm{d}v\\
   % sum of KLs
   &= KL[q(v) \mid \mid p(v)] + KL[q(u \mid \mid v) \mid p(u \mid v)].
   \end{aligned}
   $$

The second term in the last line is known as the conditional KL-divergence; it amounts to the expected value of the KL-divergence between conditional distributions $$q(u \mid v)$$ and $$p(u \mid v)$$, where the expectation is taken with respect to $$q(v)$$. Since KL-divergence is non-negative, both terms are non-negative. KL is equal to zero only when both probability distributions are exactly equal. The conditional KL is equal to zero when both conditional distributions are exactly equal on the whole support defined by $$q(v)$$. This last bit can be problematic when we're attempting to minimise KL with respect to the parameters of both distributions.

Let $$q(z) = q_\psi(u, v) = q_\phi (u \mid v) q_\theta(v)$$, such that the posterior is parametrised by $$\psi = \{\phi, \theta\}$$. If we look at the gradient of the KL divergence, we have that

$$
\begin{aligned}
\nabla_\psi KL[q_\psi(u, v) \mid \mid p(u, v)] &=\\
&= \nabla_\theta KL[q_\theta(v) \mid \mid p(v)] + \nabla_\psi KL[q_\phi(u \mid v) \mid \mid p(u \mid v)],
\end{aligned}
$$

where the last term involves gradients with respect to the parameters of the conditional distribution $$\phi$$ as well as to the parameters of its support distribution $$\theta$$. If we minimise the last term with respect to both sets of parameters, it might well be easier to change the support $$q_\phi(v)$$ to a volume where both conditionals have very small values instead of optimising $$q_\theta(u \mid v)$$. From my experience, it happens especially when the value of the conditional KL is much bigger than the value of the first KL term.

An alternative approach would be to perform the optimisation of the second term only with respect to the parameters of the distribution inside the expectation: $$\phi$$. That would result in the following gradient equation:  

$$
\begin{aligned}
\nabla_\psi KL[q_\psi(u, v) \mid \mid p(u, v)] &=\\
&= \nabla_\theta KL[q_\theta(v) \mid \mid p(v)] + \nabla_\phi KL[q_\phi(u \mid v) \mid \mid p(u \mid v)],
\end{aligned}
$$

where

$$
\nabla_\phi KL[q_\phi(u \mid v) \mid \mid p(u \mid v)] = \mathbb{E}_{q_\theta(v)} \left[ \nabla_\phi KL[q_\phi(u \mid v) \mid \mid p(u \mid v)] \right].
$$

I overloaded notation here, so that the KL inside the expectation does not involve integrating with respect to $$v$$. This optimisation scheme resembles the [Expectation-Maximisation (EM) algorithm](https://en.wikipedia.org/wiki/Expectation%E2%80%93maximization_algorithm). In the E step, we compute the expectations, while in the M step we fix the parameters with respect to which the expectations were computed and we maximise with respect to the functions inside the expectation. In EM we do this, because maximum-likelihood with latent variables often does not have closed-form solutions. The motivation here is to make the optimisation more stable.

I wrote this blog post, because I have no idea whether this *changed* optimisation procedure is justified in any way. What do you think? I would appreciate any comments.
