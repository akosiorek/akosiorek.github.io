---
layout: draft
title:  "How to use a VAE for supervised learning?"
date:   2021-02-08 16:23:0 +0000
categories: ml
---

- models are never perfect; to quantify how bad our models are, we should quantify the uncertainty in their predictions
- uncertainties can be very helpful when making decisions in life-threatening situations e.g. medicine or transportation
- there are two types of uncertainties: aleatoric and epistemic. Epistemic is done by test-time drop-out or bayesian deep learning and is not a target of this post. Here, we assume we have enough data to minimize the epistemic uncertainty, and we focus on aleatoric uncertainty.
- dealing correctly with aleatoric uncertainty is especially important in continuous prediction problems, see [this post](a) for a longer discussion.
- in this post, we focus on one method of estimating aleatoric uncertainty: using a VAE for supervised training.
- what is a VAE, see [this post](a) for a longer discussion.
- a naive version of a supervised VAE
- the insufficient posterior conditioning problem
- a properly-structured supervised VAE + discussion
- some results on noisy binary mnist


Epistemic uncertainty: It's the uncertainty in our model. Given enough data, it goes away. It's the uncertainty that we see when we train several models on different chunks of data. This type of uncertainty can be estimated by Bayesian models (where we have a prior over model parameters), but not by a VAE.

Aleatoric uncertainty: It's responsible for what our data cannot explain. For example, if we see a single image of a large object, we do not know if it is close to the camera and small or far away and really big. This type uncertainty can be estimated as the uncertainty of a posterior distribution in a VAE.


## Further Reading
* [Kohl et. al., "A Hierarchical Probabilistic U-Net for Modeling Multi-Scale Ambiguities", arXiv 2018](https://arxiv.org/abs/1905.13077) is the original paper from which I learned about this idea.
* [Bayer et. al., "Mind the Gap when Conditioning Amortised Inference in Sequential Latent-Variable Models", ICLR 2020](https://arxiv.org/abs/2101.07046) shows what happens when you use insufficient conditioning for the VAE posterior.
* [Kendall and Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017](https://arxiv.org/abs/1703.04977) is useful to understand the difference between aleatoric and epistemic uncertainty. Here is an associated [blog post](https://alexgkendall.com/computer_vision/bayesian_deep_learning_for_safe_ai/).


#### Acknowledgements
I have the pleasure of working with and learning from Danilo J. Rezende who [invented VAEs](https://arxiv.org/abs/1401.4082)--it's Danilo who first taught me about supervised VAEs and the insufficient conditioning problem.
