---
layout: post
title:  "Implementing Attend, Infer, Repeat"
date:   2017-09-03 15:44:17 +0100
categories: ML
---

Variational Autoencoders (VAEs) are getting quite popular in the Machine Learning community. 
While the formulation is more involved then that of a typiical feed-forward neural network, VAEs have a lot of added benefits. 
I've been recently playing with one of the more complicated frameworks: [Attend, Infer, Repeat (AIR)](https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models) by [Ali Eslami et. al.](http://arkitus.com/) from [DeepMind](https://deepmind.com/), and I must say it's really cool. 
In this blog post, I will describe the model and part of its implementation, which is available [here](https://github.com/akosiorek/attend_infer_repeat).

**Disclaimer**: This post might be not enough to actually understand the paper, and definitely not if you don't know how VAEs work. It assumes that you have a pretty good understanding and would like to know how to implement it, or to experiment with my implementation.

# What does it do?
Take a look at the image below. 
Given an image with some objects (here 2 digits, x) AIR takes a look at it, figures out how many interesting parts there are, and reconstruct the image by painting one-part-at-a-time onto a blank canvas. 
Sounds easy enough? 
It's not, and for two reasons: 
(i) it's completely unsupervised and 
(ii) it creates a variable-length representation of an image. 
The first one is tricky, because we don't really know how to define an object or an interesting part (more on this later).
The second leads to discrete latent variables, which are not-that-easy to deal with when computing gradients.

![AIR](air_flow.png){: style="display: block; margin: auto; max-width: 400px;"}
Let's go back to the figure. AIR does the following:
* Infer: figures out if an object is present in the image ($$z_{pres}$$) and where it is ($$z_{where}$$)
* Attend: takes a look at this location ($$z_{what}$$)
* reconstructs the crop and places it into a blank canvas
* Repeat: repeats the procedure

What is beautiful about it, is that we get a variable-length representation of the image: the more interesting stuff there is, the longer the description.
Even better is the fact that we know that each bit describes a particular location or an object, and it allows us to explicitly reason about objects and relations between them.

# What do we need?
We will start by defining a few core components. AIR is an autoencoder, and we will need an encoder and a decoder, but there's more than that, namely:
* Input encoder: creates some hidden representation of the input image
* Transition: Uses the input encoding and latent variables $$z_{what}$$, $$z_{where}$$ and $$z_{pres}$$ to aggregate information; it's responsible for knowing where we have already looked in the input image.
* Presence & Location models: Given the hidden state (from the Transition), their role is to predict if (another) object is present in the image and if yes, where it is.
* Spatial Transformers: given the location parameters, it takes a crop of the original input image; it will later place a reconstructed crop into a canvas.
* Glimpse encoder: Given a crop (or a glimpse), it encodes it some low-dimensional representation, typically a vector.
* Glimpse decoder: Given an output from the glimpse encoder, it's role is to reconstruct the glimpse.

I defined all these components in a [single file](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/modules.py) as [Sonnet](https://github.com/deepmind/sonnet) modules. 
Since we don't want to dwell on complicated architectures, I used a small fully-connected neural networks with 2 hidden layers of 256 units each and ELU nonlinearities for every component. 
My transition is a 256-dimensional LSTM core from Sonnet with a trainable initial state.
I put together all the modules into a working `tf.RNNCell` in the [cell.py](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/cell.py).

# Probability Distributions
One reason why VAEs are more complicated than standard neural nets are the probability distributions. Each of the latent variables $$z$$ is not just predicted by a corresponding model, by it is sampled from a probability distribution, whose parameters were predicted by the corresponding model. $$z_{what}$$ and $$z_{where}$$ both come from a Gaussian distributions with diagonal covariance meatrices with means and variances predicted by neural networks. I used `tf.NormalWithSoftplusScale` to ensure the numerical stability of the scale parameters. $$z_{pres}$$ is much more tricky.At inference time, it comes from a Beroulli distribution parametrised by an output of a neural net. When the previous sample was equal to 1, we take the current sample as is. But as soon as we draw a sample equal to 0, all subsequent samples have to be set to zero, too. This results in a modified geometrical distribution, for which we have to account when we implement the KL-divergence with the prior. 

# Piors
Every VAE requires a prior on its latent representation. 
AIR requires at least three priors for three different latent variables. 
I used a `Normal(0, 1)` prior for both $$z_{what}$$ and $$z_{where}$$ and a modified geometric-like prior for $$z_{pres}$$ (number of steps). 
This modified geometric distribution can be implemented as ancestral sampling from a Bernoulli distribution.
I implemented a `NumStepsDistribution` in [prior.py](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/prior.py) that creates the modified geometric distribution given Bernoulli probabilities at consecutive steps. 
There is also a KL-divergence implementation. 
Setting the success probabilities for the prior is tricky, however.
The paper mentions only that it is uses a "geometric prior which encourages sparse solutions", which tells us only that the success probability in the geometric distribution is low. 
When I emailed the author, however, I found out that we annealed the success probability from a value close to 1 to either $$10^{-5}$$ or $$10^{-10}$$ depending on the dataset over the course of 100k training iterations. 

Intuitevely, it makes sense.
At the beginning of training, we would like the model to take (more than 0) steps so that it can learn.
The further we go into the training, the more we can constrain it.
Very low values of the success probability are important, because the reconstruction loss is summed across the whole image 
(it has to: in the derivation of the loss pixels are assumed to be conditionally independent given $$z_{what}$$ 
and log probability of independent events results in a sum) and KL divergence has to compete with it during the optimisation. 


To Be Continued...
