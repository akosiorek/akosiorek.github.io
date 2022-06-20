---
layout: draft
title:  "Masking for Representation Learning in Vision"
date:   2022-06-20 08:23:0 +0000
categories: ml
---

# On Masking for Representation Learning in Vision

Have you ever covered what you see with your hand, and tried to imagine how the covered part looks like?
If not, why won't you try?
You may not be able to recreate it, to draw it or paint it, nor even see it clearly in your mind's eye.
Yet, you have a rough idea.
You know what it is.
You know what it can be used for.

You may even know that your brain is doing this all the time.
Each of your eyes has a visual blind spot.
It's roughly in the middle vertically, slightly off-centre to the outside for each eye.
It's the place where the visual nerve connects to the eye, leaving no place for light cells.
You don't see anything there, it's as good as masked.
Check out the exercise on [Wikipedia](https://en.wikipedia.org/wiki/Blind_spot_(vision)) to see for yourself.
And yet, you do see there, with perfect clarity, with clear perfection.
Your brain is filling out this area based on what is around it, based on where you are, based on what you're expecting to see.

Blind spot image: adapted from [Scientific American](https://www.scientificamerican.com/article/find-your-blind-spot/),


The brain had to learn to do this.
I don't know how, nor if learning to inpaint the blind spot is necessary for the brain to learn good visual representations[^brain_learn].
Given recent advances in machine learning, chances are that the response to the second question--whether learning to inpaint the blind spot is useful for other things--is affirmative.
Let me walk you through why that is, and show some interesting results.

# aa

Although the first vision paper that learns representation by masking I am aware of is [Context Encoder by Pathak et. al.](https://arxiv.org/abs/1604.07379), I was first convinced that this idea is good by [BERT of Devlin et. al.](https://arxiv.org/abs/1810.04805).

Coming two years later, BERT learns representations for natural language.
BERT is a large transformer trained to fill-in missing words in a sentence based on the available words.
This works, because the model has to learn dependencies between different words.
It needs to predict which words make sense in the presence of other words.
Why is this useful?
Because words are contracts.
They stand for concrete objects, or for abstract entities that are so substantial they might as well be concrete.
An object is largely defined by what you can do with it, or by its relations to other objects.
Hence, learning those relations lets you learn about the world.

So why not do this for vision?
Well, you can, but there are issues.
So even though Pathak et. al. showed a proof of concept in 2016, it took 5 years (3 years after BERT) to show that this idea is, indeed, useful.
Meet the [Masked Autoencoder (MAE) by He et. al.](https://arxiv.org/abs/2111.06377)

Why did it take so long?
For this to work, we needed the right combination of compute infrastructure (large), architectures (transformer), with some algorithmic advancement as well.
Where context encoders use relatively small CNNs, MAE uses a large vision transformer ([ViT, Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929)).
This allows to feed only the visible tokens to the model instead of the whole-image-with-masked-parts to the encoder.
This not only reduces the required compute, but also is a better way to represent a masked region (i.e. by not feeding it to the model at all).
Finally, CE used rectangular masks placed at random, with one rectangle per image.
MAE uses many small rectangles per image, masking a far greater percentage of the image than CE, but also masking distant parts of the image simultaneously[^scattered_mask].




#### Acknowledgements


[^brain_learn]: If you know any references to the contrary, please let me know in the comments or by email. Thanks!

[^scattered_mask]: It is unclear what the impact of such scattered masks is. I guess that it forces the model to reason about multiple things in every image on one hand, and reduces the variance of the gradients on the other. The latter is due to the fact that total occlusion of a certain object is less likely with such scattered masks.
