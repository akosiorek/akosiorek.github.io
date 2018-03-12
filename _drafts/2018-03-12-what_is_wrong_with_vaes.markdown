---
layout: draft
title:  "What is wrong with VAEs?"
date:   2018-03-01 15:15:0 +0000
comments: True
categories: ML
---

Suppose you would like to model the world in terms of the probability distribution over its possible states $$p(\mathbf{x})$$ with $$\mathbf{x} \in \mathcal{R}^D$$.
The world may be complicated and we do not know what form $$p(\mathbf{x})$$ should have.
To account for it, we introduce another variable $$\mathbf{z} \in \mathcal{R}^d$$, which describes, or explains the content of $$\mathbf{x}$$.
If $$\mathbf{x}$$ is an image, $$\mathbf{z}$$ can contain information about the number, type and appearance of objects visible in the scene as well as the background and lighting conditions.
This new variable allows us to express $$p(\mathbf{x})$$ as an infinite mixture model

$$
  p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z}) p(\mathbf{z})~d \mathbf{z}. \tag{1}
$$
It is a mixture model, because for every possible value of $$\mathbf{z}$$, we add another conditional distribution to $$p(\mathbf{x})$$, weighted by its probability.

Having a setup like that, it is interesting to ask what the latent variables $$\mathbf{z}$$ are, given an observation $$\mathbf{x}$$. Namely, we would like to know the posterior distribution $$p(\mathbf{z} \mid \mathbf{x})$$. However, the relationship between $$\mathbf{z}$$ and $$\mathbf{x}$$ can be highly non-linear (e.g. implemented by a multi-layer neural network) and both $$D$ and $$d$$ can be quite large. Since both marginal and posterior probability distributions require evaluation of eq. 1, they are intractable.

To train a probabilistic model, we can use parametric distributions - parametrised by neural networks with parameters $$\theta \in \Theta$$.
We can now learn the parameters by maximum likelihood estimation, namely

$$
  \theta^\star = \arg \max_{\theta \in \Theta} p_\theta(\mathbf{x}) \tag{2}
$$

The problem is, we can't evaluate this expression due to the non-linearities.
We could approximate it by Monte-Carlo sampling, but since the volume of z-space is potentially large, we would need millions of samples to get a reliable estimate.
To improve things, we can resort to [importance sampling (IS)](https://en.wikipedia.org/wiki/Importance_sampling).
IS allows us to sample from a different probability distribution and use weighted samples, where weighting is with respect to the original pdf, to evaluate expectations.
We can write

$$
  p(\mathbf{x}) = \mathbb{E}_{p(\mathbf{z})} \left[ p_\theta (\mathbf{x} \mid \mathbf{z} )\right] =
  \mathbb{E}_{p(\mathbf{z})} \left[ \frac{q_\phi ( \mathbf{z} \mid \mathbf{x})}{q_\phi ( \mathbf{z} \mid \mathbf{x})} p_\theta (\mathbf{x} \mid \mathbf{z} )\right] =
  \mathbb{E}_{q_\phi ( \mathbf{z} \mid \mathbf{x})} \left[ \frac{p_\theta (\mathbf{x} \mid \mathbf{z} ) p(\mathbf{z})}{q_\phi ( \mathbf{z} \mid \mathbf{x})} )\right]. \tag{3}
$$

From [importance sampling literature](http://statweb.stanford.edu/~owen/mc/) we know that the optimal sampling distribution (proposal) is proportional to the original probability distribution times the function, whose expectation we are trying to approximate. In our setting, that function is just $$p_\theta (\mathbf{x} \mid \mathbf{z} )$$.
From Bayes theorm, we see that the optimal proposal is proportional to the posterior distribution, which is of course intractable.
It turns out that we can kill birds with one stone.
By trying to approximate the proposal, we can efficiently approximate the marginal probability $$p_\theta(\mathbf{x})$$

To approximate the posterior, we can use the KL divergence (think of it as a distance between probability distributions) between the approximation of the posterior and the posterior itself; and we can minimise it.

$$
  KL \left( q_\phi (\mathbf{z} \mid \mathbf{x}) || p(\mathbf{z} \mid \mathbf{x}) \right) = \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi (\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z} \mid \mathbf{x})} \right] \tag{4}
$$

Our new problem, of course, is that to evaluate the KL we need to know the posterior distribution.
Not all is lost, for doing a little algebra can give us an objective function that is possible to compute.

$$
  \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log \frac{q_\phi (\mathbf{z} \mid \mathbf{x})}{p(\mathbf{z} \mid \mathbf{x})} \right] =
  \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi (\mathbf{z} \mid \mathbf{x}) - \log p(\mathbf{z} \mid \mathbf{x}) \right] = \\
  \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})} \left[ \log q_\phi (\mathbf{z} \mid \mathbf{x}) - \log p_\theta(\mathbf{z}, \mathbf{x}) \right] - \log p_\theta(\mathbf{x}) =
  \mathcal{L} (\mathbf{x}; \theta, \phi) - \log p_\theta(\mathbf{x})
  \tag{5}
$$

Where $$\mathcal{L} (\mathbf{x}; \theta, \phi)$$ is the evidence-lower bound (ELBO). We can rewrite it as

$$
  \log p_\theta(\mathbf{x}) = \mathcal{L} (\mathbf{x}; \theta, \phi) - KL \left( q_\phi (\mathbf{z} \mid \mathbf{x}) || p(\mathbf{z} \mid \mathbf{x}) \right), \tag{6}
$$
$$
  \mathcal{L} (\mathbf{x}; \theta, \phi) = \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})}
    \left[
      \log \frac{
        p_\theta (\mathbf{x}, \mathbf{z})
      }{
        q_\phi (\mathbf{z} \mid \mathbf{x})
      }
    \right]. \tag{7}
$$

By maximising the ELBO, we (1) maximise the marginal probability, and/or (2) minimise the KL divergence.
It is worth noting that the ELBO has the form of importance-sampled expectation of $$f(\mathbf{x}) = 1$$, with importance weights $$w(\mathbf{x}) = \frac{ p_\theta (\mathbf{x}, \mathbf{z}) }{ q_\phi (\mathbf{z} \mid \mathbf{x})

# What is wrong with this estimate?
If you look long enough at importance sampling, it becomes apparent that the proposal distribution should be wider than the original probability distribution - that is both to avoid infinite variance of the estimator and numerical instabilities.
In that case, it would be better to optimise the reverse KL(p || q), which has mode-averaging behaviour instead of mode-matching one.
It would require taking samples from the true posterior distribution, which is hard.
Instead, we can use IS estimate of the ELBO, introduced in the excellent [IWAE paper](https://arxiv.org/abs/1509.00519) with a higher number of particles

$$
  \mathcal{L}_K (\mathbf{x}; \theta, \phi) = \mathbb{E}_{q_\phi (\mathbf{z} \mid \mathbf{x})}
    \left[ \frac{1}{K} \sum_{k=1}^{K}
      \log \frac{
        p_\theta (\mathbf{x}, \mathbf{z^{(k)}})
      }{
        q_\phi (\mathbf{z^{(k)}} \mid \mathbf{x})
      }
    \right]. \tag{8}
$$
This estimator [has been shown](https://arxiv.org/abs/1509.00519) to optimise modified KL divergence KL(Q||P), with Q and P defined as
$$
  Q = ... \tag{9}
$$
$$
  P = ... \tag{10}
$$

While similar to the original distributions, Q and P allow in small variations in q and p that we would not have expected.
Optimising this lower bounds leads to better generative models as shown in the original paper.
It also leads to higher-entropy estimates of the approximate posterior $q$, effectively breaking the mode-seeking behaviour of the original KL divergence.
A curious consequence of this fact is that as we increase the number of particles $$K$$ to infinity, we no longer need the inference model $$q$$.

# What is wrong with IWAE?
The importance-weighted ELBO, or the IWAE changes with the number of particles $$K$$ used to estimate it.
The higher the number of particles, the tighter (closer to the true log likelihood) the bound is.
This means that the gradient estimator derived by differentiating the IWAE points us in a better direction.
Additionally, the variance of that gradient estimator shrinks.
However, the behaviour is slightly different from the generative and the inference model.
For the latter, it turns out, the norm of the gradient goes to zero.
It is not a problem itself; but it is problematic that it does so faster than its variance.
If we define signal-to-noise ratio (SNR) as...

It turns out that SNR converges with $$K$$ as ... for q and ... for p.
The conclusion here is simple: the more particles we use, the worse the inference model becomes.
If we care about representation learning, we have a problem.


# Better solution
We can do better than IWAE, as we've shown in [our recent paper](https://arxiv.org/abs/1802.04537).
The idea is to use separate objectives for the inference and the generative models.
By doing so, we can ensure that both get non-zero low-variance gradients, which leads to better models.
As a perhaps surprising side effect, our new estimators achieve higher IWAE-64 bounds than the IWAE itself.
Why?
By looking at the effective sample-size (ESS), it looks like the VAE produces best proposals, but the worst generative models.
If we combine a good proposal with an objective that leads to good generative models, we should be able to provide lower-variance estimate of this objective and thus learn even better models.
