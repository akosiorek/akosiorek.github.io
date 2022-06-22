---
layout: draft
title:  "On Few-Shot Generative Modelling"
date:   2022-04-04 08:23:0 +0000
categories: ml
---


# On Few-Shot Generative Modelling


A recent discussion with a friend made me realise that the idea of few-shot generative modelling can be confusing, and might be generally poorly-understood.
Let's start with the basics.

A generative model $g$ tires to approximate a probability distribution $p(x)$ in some fashion.
Usually, we don't have access to $p(x)$, but we do have a set of samples $\{x_i\}_{i=1}^N$ from that distribution.
This forms our training set, which can be seen as the empirical data distribution $\hat{p}(x) = \frac{1}{N} \sum_{i=1}^N \delta(x \mid x_i)$.
Different generative models fit different use-cases.
Often, $g$ performs density estimation, such that for any $x \in \mathcal{X}$, $g$ returns a density value $\geq 0$.
Sometimes, instead of estimating density directly, it estimates unnormalized energy, which can be negative and is not comparable between different models.
These models are believed to be more powerful than normalized probabilities models, because they can represent complex relationships in the data but they do not have to estimate the normalizing constant, which is an insanly difficult problem [cite LeCun's interview]().
Some models learn to generate samples, but do not provide any numerical scores (GANs or diffusion models).

For any of these model types, $g$ represents the data $\mathcal{X}$ in the absence of any conditioning.
Conditional models of the form $g(\cdot
\mid c)$ with conditioning $c$ can be trained, but this requires having paired data $(x_i, c_i)$ and training models from scratch.
That is, we need to be aware of (and have the data for) and conditioning value $c$ that we might want to use to conditioning $g(\cdot \mid c)$ after it is trained.
An interesting exception here are diffusion models, which generate data by inverting a stochastic differential equation (SDE).
The forward process of this SDE starts with a datapoint and gradually perturbs it into pure noise.
These models, it turns out, can be sampled from conditionally even though the base generative model was trained as unconditional one, see e.g. [here]().

What does this have to do with few-shot modelling?
Few-shot modelling is usually considered in a discriminative setting, e.g. classification.
The task is that, given a set of context pairs $C = \{(x_i, y_i)\}_{i=1}^K$ and a query example $x_j$, predict which of the $y_i,...,y_K$ label best described the query.
We can write this probabilistically as a density estimation problem.
That is, the goal is to figure out what is the label distribution given the context set and the query, that is $p(y \mid x, C)$.
This can be easily generalised to few-shot generative modelling:
given a context set $C = \{x_i\}_{i=1}^K$ that describes some data distribution, learn a conditional generative model (any of the models described above), that estimates $p(x \mid C)$.
The difference is that there are no queries here.

How does this differ from conditional generative modelling?
In the vanilla setup, the task is usually to model distributions conditioned on some side information, e.g. labels: given a label $y=\text{"cat"}$, the model should estimate $p(x \mid \text{"cat"})$ where x are different cat images.
Other examples include gender or hair or skin colour (which is available e.g. with CelebA).
In few-shot generative modelling, the conditioning takes the form of a few datapoints.
The model has to infer which factors of variation are in common, which are not, what is their range, and what is the distribution that could have given rise to these images.
On one hand, such models allow for less-direct modulation of the estimated distribution, because we need to give examples of what we might be interested in.
On the other hand, they are much more general, because not everything can be described with words; nor is ground-truth data available for any factor of variation we can imagine.
An interesting use-case of a few-shot generative model is that, once it is trained, we could train side model that generate fine-grained conditioning sets, which in turn allow for fine-grained modulation of the few-shot model.
Where's the benefit?
These side models can be trained after the main model is trained as new requirements become known, and new data becomes available.


Few-shot modelling has generally been limited to discriminative tasks, e.g. classifcation

This paper proposes a novel hierarchical VAE.
Authors say that, given a small set of observations from some distribution, the
model should be able to approximate that distribution and allow sampling from it,
without further optimising the model parameters--this is the setting of
few-shot generative modelling.

Let u be all latent variables in the model, and X = {x_i}_{i=1}^N a set of
cardinality N.
The generative model is then p(X) = \int p(X|u)p(u) du, and the approximate
posterior is q(u|X).

The novelty is in the construction of the latent space.
The latent u separates into two types of latent variables: set-specific c and
item-specific Z = {z_i}_{i=1}^N for N set elements, both of which are modelled
hierarchically.
The idea is that c carries information about the set (or the distribution
described by the input set X), while zs model individual differences between set
elements.
Interestingly, assuming that the hierarchy is top-down with layer L being the
top-most one, one level of the Z variable conditions the next level
of c p(c^l|c^{l+1}, Z^{l+1}), which in turn conditions the next level of Z
p(Z^l|c^l, z^{l+1}).
This allows modelling correlations between individual set elements.
While the structure makes sense for set modelling, I do not see how this reflect
the problem statement of few-shot generative modelling.
See "soundness" section for more details.

This model is unconditional, which makes it easy to generate unconditional
samples of sets.
For conditional sampling, the authors propose sampling the topmost set latent
c^L ~ (c^L|X) from the approximate posterior, while sampling the rest of the
latents from the prior.
They then sample a set of size 1 from the likelihood (which the model was never
trained to produce), append it to the conditioning set \hat{X} = [x, X],
and repeat the procedure.
While there is some explanation, I failed to understand why this is a correct
way to sample conditionally from this model.

The model is evaluated on generative modelling of sets, where the authors report
the ELBO values and marginal log-likelihood estimates with 1000 importance
samples.
These metrics are reported under in- and out-of-distribution conditioning on
omniglot, mnist, cifar100 and celeba.
While the current model outperforms the baselines, I am have some doubts to
the soudness of these results, see below.

3. Soundness.
While the paper is fairly well-written and generally of high quality, I find
that the authors confuse few-shot generative modelling (A) with generative modelling of
sets (B).
A is about conditional density estimation: given a
small set of examples from a distribution, we want to infer what that
distribution is and to approximate it, such that we can sample from it and
evaluate the density.
Most often, the set elements will still be i.i.d., which is the case in this paper.
Consequently, if we have a small set of examples X that describes the distribution of
interest, we are interested in a model p(x|X) that allows
(conditionally-, given the conditioning set) i.i.d. sampling from
that distribution.
One way to implement such a model is through a conditional prior, e.g.
p(x|X) = \int p(x|u)p(u|X) d u
In particular, there is no need of modelling correlations between elements of
this distribution beside making them conditionally-independent of the
input set.

B, on the other hand, does require estimating distributions over sets, and
therefore modelling correlations between set
elements (e.g. SetVAE that is cited in the related work section).
It does not require modelling conditional distributions.
For this, the chosen model structure p(X) = \int p(X|u)p(u) d u is perfect.

Consequently, A and B are orthogonal.
While it is ok to consider the joint problem of few-shot generative modelling
of sets, there is nothing in this paper that makes me think that the authors
understand that.
In particular, they say that "learning joint distributions ... from few samples"
in R51-52.
This makes me think that the authors consider it necessary to estimate
a joint distribution over set elements.
This may be because the inputs to a few-shot models are usually a set of
examples.
But this does not mean that the output has to be a set.

I imagine that this confusion could result from the choice of
neural statistician (NS) as the underlying framework.
While NS performs few-shot distribution estimation, and that distribution is
over sets, those sets consist of i.i.d. elements! Which is equivalent to
considering distributions over single datapoints, and not sets thereof.

All this makes for a number of modelling decisions that do not make sense for
few-shot generative modelling, accompanied by a rather confusing description.

First, even if we assume that the paper is about few-shot generative modelling (A),
the evaluations are purely for (B), so the concept of few-shot generation with
this model is not validated.

Second, if we assume that the paper is about B, then there is a technical issue.
When evaluating the ELBO (both for training and evaluation), the authors
average the ELBO values across minibatch elements and across set elements.
Since in training the authors always use sets of 5 elements, this may be ok
for model comparison, but it does not reflect the true ELBO values.
To see this, look at the ELBO equation (5): it clearly does not factorize across
set elements, as some terms appear once per set, and others once per set element.
Moreover, at evaluation, the authors compare ELBO values and marginal-likelihood
estimates across different set cardinalities.
This is wrong for two reasons.
i) the amount of added information per set element decreases with the size
of the set.
This is similar to density estimation on images of increasing sizes, where
bits-per-dim increases with the image resolution, which is caused by increasingly
stronger correlation between image pixels.
ii), it is technically incorrect to compare likelihoods of sets of different
cardinality, similarly to how it does not make sense to compare area (m^2) with
volume (m^3).
Please see Vo et. al., "Model-Based Multiple Instance Learning", arXiv 2017 for a discussion of this
problem and a manner in which set-likelihood should be adjusted such that it can
be compared across different cardinalities.


2. Significance, Novelty, Relevance.
While the model is interesting, it is currently unclear if it is for generative
modelling of sets or for few-shot generative modelling, see "Soundness".
Therefore I cannot comment on significance, novelty or relevance.

Clarity.
- It is very unclear what the paper is about, see "soundness".
- Fig 1 contains terms that are explained only much later.
- "vlb" for variational lower-bound, present in several figures, is never
explained and differs from the usually-used ELBO.


4. Literature
While the authors cite the majority of relevant papers, they fail to notice
a critical link.
The Generative Query Network (Eslami et. al, 2018) that the authors cite in R359,
as well as any paper that followed GQN, does few-shot generative modelling.
The problem statement is as follows: Given one or more views of a 3D environment
(plus, perhaps, associated camera pose), estimate a distribution of the views of that
particular 3D environment. So we have conditioning on a set, and we want to estimate
a distribution that this sets belongs to. The generated samples are conditionally-independent
given that conditioning set (and/or the latent variable, depending on particular
formulation).
I suppose it is this unawareness of this link that prompted the authors to state
"Few-shot generation has been limited to simple tasks, shallow and handcrafted
conditioning mechanisms, and a pretraining for supervised few-shot learning" in
L76-80.
Note that if GQN is doing few-shot generative modelling, then so is the whole
family of neural processes.



5. Summary
6. Improvement Suggestions
- "Model Design" paragraphs from section 4 should be moved to the method section (3).
In the description of Fig 1 you mentioned that the convolutional latent space is
paramount to performance of this model class. But what about convolutional latents
makes them important? Is it the weight sharing, the local receptive fields of the kernels,
the fact that a convolutional latent has spatial structure, or perhaps the increased
dimensionality (directly resulting from the spatial structure)?
- The model seems to be very related to SetVAE. It would be nice to have side-by-side
comparison of the graphical models for the two. At the moment it's a bit difficult to judge
your contribution.

#### Acknowledgements
