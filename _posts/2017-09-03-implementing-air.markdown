---
layout: post
title:  "Implementing Attend, Infer, Repeat"
date:   2017-09-03 15:44:17 +0100
categories: ML
---

Variational Autoencoders (VAEs) are getting more and more popular in the Machine Learning community. 
While the formulation is more involved then that of a typical feed-forward neural network, VAEs have a lot of added benefits.
I've been recently playing with one of the more complicated VAE model: [Attend, Infer, Repeat (AIR)](https://papers.nips.cc/paper/6230-attend-infer-repeat-fast-scene-understanding-with-generative-models) by [Ali Eslami et. al.](http://arkitus.com/) from [DeepMind](https://deepmind.com/), and I must say it's really cool. 
In this blog post, I will describe the model and break it down into simple components. It will also cover implementation, or rather issues I had while implementing it. The full implementation is available [here](https://github.com/akosiorek/attend_infer_repeat).

# What does AIR do?
Take a look at the figure below. 
Given an image, we would like to reconstruct it.
Instead of doing it in a single shot, however, we'd like to focus on interesting parts of the image and reconstruct them one by one.
Given an image, AIR takes a look at it, figures out how many interesting parts there are, and reconstruct the image by painting one-part-at-a-time onto a blank canvas. 
Sounds easy enough? 
Well, it's not, and for two reasons: 
* It's completely unsupervised,
* It takes a variable yet discrete number of steps. 

The first one is tricky, because we don't really know how to define an object or an interesting part (more on this later).
The second leads to discrete latent variables, which are not-that-easy to deal with when computing gradients.

![AIR](air_flow.png){: style="display: block; margin: auto; max-width: 400px;"}
Let's go back to the figure. AIR is called Attend, Infear, repeat for a reason:
* It **attends** to a part of an image using spatial transformers, effectively cropping it,
* then it **infers** the latent variables best describing the crop,
* and finally it **repeats** the procedure for the rest of the image.

Technically, the order is different, because it has to infer presence of an object and its location before attending to it; and the name describes only the inference process, not reconstruction.

What is beautiful here, is that we get a variable-length representation of the image: the more complicated the image is, the longer representation we get.
What's even better, is that we know that each piece of description is tied to a particular location (and hopefully an object), which allows explicit reasoning about objects and relations between them.

# Why and how does it work?
Like every VAE, AIR is trained by maximising the evidence lower bound (ELBO) $$\mathcal{L}$$ on the log probability of the data:
  
   $$
    \begin{align*}
        \log p(x) &= \mathcal{L}(\theta, \phi) + KL(q_\phi(z \mid x) \mid\mid p(z \mid x))\\
        \mathcal{L}(\theta, \phi) &= - \mathbb{E}_{q_\phi(z)} [\log p_\theta(x \mid z)]] - KL(q_\phi(z\mid x) \mid\mid p(z))
    \end{align*}
   $$

The first term of the ELBO is a probabilistic analog of the reconstruction error and the second term acts as a regulariser. For AIR, the second term tries to keep the number of steps low, but it's also forcing the shortest encoding possible of an image part reconstructed at each step. 

Now, short encoding means that the model has to focus on parts of the image that can be explained with relatively few variables. 
It turns out that we can define an object as an image patch, where pixel correlations within that patch are strong, but correlation between pixels inside and outside of that patch is weak. 
Also pixels belonging to two different objects have very low correlation. 
That means that explaining even small parts of two different objects leads to potentially longer encoding than explaining one (potentially big) object at a time. 
This leads, at least in case of uncomplicated backgrounds as in the paper, to a  model which learns to take the minimum number of steps possible, where every step explains an internally-consistent part of the image.


# What do we need?
We will start by defining a few core components. AIR is an autoencoder, and we will need an encoder and a decoder, but there's more than that, namely:
* Input encoder: transforms the input image $$x$$ into some hidden representation $$v$$.
* RNN: Since we're taking multiple peeks at the image, we need some hidden state $$h$$ to keep track of what has already been explained. It creats the new hidden state as 

$$\begin{align}
    h^{i+1} = RNN(v, h^i, z^i),
\end{align}$$

where $$z^i = (z^i_{what}, z^i_{where}, z^i_{pres})$$ are the latent variables describing the apperance, location and presence of an object, respectively.
* Presence & Location models: Given the hidden state $$h^i$$, they predict $$z^i_{pres}$$ and $$z^i_{where}$$.
* Spatial Transformer: Given the location parameters $$z^i_{pres}$$, it extract a crop  of the original input image $x^i_{att}$. It will later place a reconstructed crop $$y^i_{att}$$ into the canvas.
* Glimpse encoder: It encodes $$x^i_{att}$$ into low-dimensional latent representation $$z^i_{what}$$.
* Glimpse decoder: It decodes $$z^i_{what}$$ in the reconstructed glimpse $$y^i_{att}$.

I defined all these components in a [single file](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/modules.py) as [Sonnet](https://github.com/deepmind/sonnet) modules. 
Since we don't want to dwell on complicated architectures, I used  small fully-connected neural networks with 2 hidden layers of 256 units each and ELU nonlinearities for every component. 
My RNN is a 256-dimensional LSTM core from Sonnet with a trainable initial state.
I put together all the modules into a working `tf.RNNCell` in the [cell.py](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/cell.py).

# Probability Distributions
One reason why VAEs are more complicated than standard neural nets are the probability distributions.
 Each of the latent variables $$z$$ is not just predicted by a corresponding model, but it is sampled from a probability distribution, whose parameters were predicted by the corresponding model. 
$$z_{what}$$ and $$z_{where}$$ both come from Gaussian distributions with diagonal covariance matrices, whose means and variances are predicted by neural networks (glimpse encoder and location model, respectively). 
I used `tf.NormalWithSoftplusScale` for numerical stability of the scale parameters. 
$$z_{pres}$$ is much more tricky.
 At inference time, it comes from a Beroulli distribution parametrised by an output of a neural net (presence mdoel). 
When the previous sample was equal to 1, we take the current sample as is. 
As soon as we draw a sample equal to 0, however, all subsequent samples have to be set to zero, too. 
This results in a modified geometrical distribution, for which we have to account when we implement the KL-divergence with the prior. For this reason, I implemented a `NumStepsDistribution` in [prior.py](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/prior.py) that creates the modified geometric distribution given Bernoulli probabilities at consecutive steps.

# Piors
Every VAE requires a prior on its latent representation. 
AIR requires at least three priors for three different latent variables. 
I used a `Normal(0, 1)` prior for both $$z_{what}$$ and $$z_{where}$$ and a modified geometric-like prior for $$z_{pres}$$ (number of steps). 
Setting its success probability is tricky, though.
The paper mentions only that it is uses a "geometric prior which encourages sparse solutions", which tells us only that the success probability in the geometric distribution is low. 
When I emailed the author, I found out that he annealed the success probability from a value close to 1 to either $$10^{-5}$$ or $$10^{-10}$$ depending on the dataset over the course of 100k training iterations. 

Intuitevely, it makes sense.
At the beginning of training, we would like the model to take a positive number of steps so that it can learn.
The further we go into the training, the more we can constrain it.
Very low values of the success probability are important, because the reconstruction loss is summed across the whole image 
(it has to: in the derivation of the loss pixels are assumed to be conditionally independent given $$z_{what}$$ 
and log probability of independent events results in a sum) and KL divergence has to compete with it during the optimisation. 


# Estimating Gradients for Discrete Variables
Discrete variables, or specifically samples from a discrete probability distribution, are difficult to back-propagate through. 
AIR uses a score-function estimator, otherwise known as REINFORCE. 
More about it [here](https://www.google.com/search?q=score-function+estimator&rlz=1C5CHFA_enGB715GB715&oq=score-function+estimator&aqs=chrome..69i57j0.3730j0j7&sourceid=chrome&ie=UTF-8). 
This estimator is difficult to work with, because the estimate has high variance. It expresses gradient of an expectation as an expectation of a gradient of the log of probability with respect to which the expectation is taken.

$$\begin{align}
\nabla_\phi \mathbb{E}_{q_\phi(z)} [ \mathcal{L} (z)] = \mathbb{E}_{q_\phi(z)} [\mathcal{L}(z) \nabla_\phi \log q_\phi(z) ]
\end{align}$$

It turns out that the expectation of this expression is equal to 0, and therefore we can either add an arbitrary term with 0 expectation without changing the result. 
If what we add is negatively correlated with $$\mathcal{L}$$, we will reduce variance. AIR uses "neural baselines" and cites [Neural Variational Inference and Learning in Belief Networks](https://arxiv.org/abs/1402.0030) by A. Mnih and K. Gregor, but doesn't give much detail.

I used a multilayer perceptron with 2 hidden layers of 256 and 128 neurons each, with a single output unit. As input I used the original flattened image concatenated with all latent variables produced by the main model. The learning rate used for training this auxiliary model (baseline) was set 10 times higher than the learning rate of the base model. 

To see how REINFORCE with a neural baseline is implemented, have a look at the `AIRModel._reinforce_` method in [model.py](https://github.com/akosiorek/attend_infer_repeat/blob/master/attend_infer_repeat/model.py).

# Conclusion
It's a really cool if a bit complicated model. I hope this post has brought you closer to understand what's going on in the paper. I've implemented it because I have a few ideas how to use it in my research. Feel free to reach out if you have any questions or comments.
