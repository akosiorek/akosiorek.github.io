---
layout: draft
title:  "Why you should use a latent variable model (LVM) for regression problems"
date:   2021-02-08 16:23:0 +0000
categories: ml
---

deterministic: a function y = f(x) which always gives the same result for a given x
stochastic: a function y = f(x) which gives different result with every invocation of f even though x does not change
generative (unconditional) model: a model that learns the distribution of data p(x)
discriminative (conditional) model: a model that estimates the distribution of a property y of x when conditioned on a data point x

- [Masegosa, "Learning under Model Misspecification: Applications to Variational and Ensemble methods", NeurIPS 2020.](https://arxiv.org/abs/1912.08335)
- [Bellemere etl. al, "A Distributional Perspective on Reinforcement Learning", ICML 2017](https://arxiv.org/abs/1707.06887)

these two use categorical output distributions
- [van den Oord et. al., "WaveNet: A Generative Model for Raw Audio"](https://arxiv.org/abs/1609.03499)
- [van den Oord et. al, "Pixel Recurrent Neural Networks"](https://arxiv.org/abs/1601.06759)

in wavenet they write That
"One approach [...]  would be to use a mixture model such as a mixture density network (Bishop, 1994)
or mixture of conditional Gaussian scale mixtures (MCGSM) (Theis & Bethge, 2015). However,
van den Oord et al. (2016a) showed that a softmax distribution tends to work better, even when the
data is implicitly continuous (as is the case for image pixel intensities or audio sample values). One
of the reasons is that a categorical distribution is more flexible and can more easily model arbitrary
distributions because it makes no assumptions about their shape."




- deep learning for classification works very well using deterministic discriminative Models
- regression is usually handled by similar deterministic discriminative models, with the difference that the output distribution is gaussian instead of categorical
- interestingly, many papers note that quantizing the problem and treating as classification usually works better. People offer various explanations: multimodal output distribution, not penalizing errors too much, making the problem simpler by choosing one of a finite number of answers. This holds both in regression and in continuous RL.
- I haven't seen any discussion of uncertainty in this context, but I think uncertainty is the key to understanding why discretization helps, and how to solve it.
- Generally we want to predict the correct solution. However, if we have not enough information, there are usually several possible answers. The problem with deterministic models is that they can output only one answer. In this case, the optimal solution in terms of the error is usually the mean answer.
- Returning the mean answer in a discrite problem is usually not a problem. Since the space of answers is limited to specific answers, the "mean" can be only one of those answers. When using a categorical output distribution parametrized by logits, the model does not learn the mean of the distributions, but the mean of the distributions' parameters: in this case, the mean of the logits. This is ok, because when we sample (or take arg-max), we still get a valid answer.
- Now, the situation is very different for continuous distributions. We limit our discussion to the Gaussian distribution, but the logic applies to other distributions as well. When a model is uncertain in the output, it learns to predict the mean of Gaussians. In this case, the "correct" solution would be to predict a mixture of Gaussians, with a varying number of Gaussian components depending on the particular data example. Since this is impossible, the model will predict the mean and the variance equal to the mean and variance of the mixture. In most cases, this will not overlap with any of the distributons' modes leading to incorrect answers. Why are those answers incorrect? Well, imagine that the correct answer is a point on a circle. If we average multiple points on that circle, we will get a point inside of the circle, which is wrong.
- A simple solution is to use a model with multiple output heads and predict several Gaussian components. That should give a better answer. The problem is, we never know how many components we need. In fact, if Gaussian distribution is incorrect for the particular problem, we might need an infinite number of such components. A better way is a latent-variable model (LVM). Such a model introduces a latent variable z, which can be also Gaussian. The point is that there will be a nonlinear transform between the z and the output x, and the distribution on x will have the form of $$\int p(x|z)p(z) dz$$. You can think of it as an infinite Gaussian mixture, with a weight of each Gaussian component $$p(x|z)$$ given by $$p(z)$$. Such a distribution can be modelled as a VAE or a BiGAN.
- An alternative choice is to model $$p(x)$$ directly as a flow, see [this post])(a) for an introduction to flows.


#### Acknowledgements
Sandy Huang for literature
