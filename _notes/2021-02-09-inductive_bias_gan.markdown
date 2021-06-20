---
layout: draft
title:  "Rich Unsupervised Learning with GANs and Inductive Biases"
date:   2021-02-08 16:23:0 +0000
categories: ml
---

Generative Adversarial Networks (GANs), introduced by [Goodfellow et. al.](a) in 2014 and popularized by [Radford et. al.]() 2015 are on the frontier on large-scale generative models for images and videos, and have revolutionized gerative modelling as a whole by showing that extremely high-fidelity generations are indeed possible.
GANs have been applied to [other domains](c), and everything I write about in this post applies to these domains, but I choose to focus exclusively on images.

GANs learn by matching distribution: given a training set of data examples, they try to generate examples that look, in distribution, like training examples.
This is interesting for several reasons.
First, GANs avoid using explicit pixel-wise error between the generated and ground-truth images.
This is useful, because it leads to very sharp-looking images, albeit often with low-level statistics that do not much that of images at all, see e.g. [here](d) for an example.
Interestingly, using pixel-wise errors (and especially the mean-squared error), need not lead to blurry reconstructions, even though the community is prejudiced to think so and associates it with the blurry reconstructions and samples often produced by VAEs.
As the [NVAE](a) and the [Very Deep VAE](a) papers show, this is not the case.
The reason for blurry reconstructions is the inability to account for all possible explanations on the generative model side, or inadequete inference which can exacerbate the previous issue.
They both lead to approximating a bunch of different solutions with a mean of them all, which gives a blurry reconstruction.
More on this problem in [my previous blog](c)

Second, and most important for this blog, GANs do not need to do inference.


#### Acknowledgements
