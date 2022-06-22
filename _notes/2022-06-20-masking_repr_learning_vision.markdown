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
Yet, You know what it is.
You know what it can be used for.

You may even know that your brain is doing this all the time.
Each of your eyes has a visual blind spot.
It's roughly in the middle vertically, slightly off-centre to the outside for each eye.
You don't see anything there.
You can't, cuz it's the place where the visual nerve connects to the eye, leaving no place for photoreceptors.
Check out the exercise on [Wikipedia](https://en.wikipedia.org/wiki/Blind_spot_(vision)) to see for yourself.
And yet, you do see there with perfect clarity.
Your brain is filling out this area based on what is around it and on the view from the other eye.
But also based on where you are, based on what you're expecting to see, etc[^blindspot_conjecture].

Blind spot image: adapted from [Scientific American](https://www.scientificamerican.com/article/find-your-blind-spot/),


The brain had to learn to do this.
I don't know how, nor if learning to inpaint the blind spot is necessary for the brain to learn good visual representations[^brain_learn].
Given recent advances in machine learning, chances are that the response to the second question--whether learning to inpaint the blind spot is useful for other things--is affirmative.
Why am I thinking that?
Because inpainting masked images has recently led to state-of-the-art results in representation learning for images.

# Masked-Image Modelling

I got first interested in masked-image modelling (MIM) in 2018, right after [BERT of Devlin et. al.](https://arxiv.org/abs/1810.04805) came out.
BERT is a large transformer trained to fill-in missing words in natural language sentences based on the available words.
This works, because the model has to learn dependencies between different words.
It needs to predict which words make sense in the presence of other words.
Why is this useful?
Because words are contracts that stand for concrete objects or for abstract entities.
An object is largely defined by what you can do with it, or by its relations to other objects.
Hence, learning those relations lets you learn about the world.

So why not do this for vision?
Well, you can, but there are issues.
First, an architectural issue is that BERT used a transformer, which is really good at reasoning about relations between different elements.
Until fairly recently, there was no good way of using a transformer for vision.
Second, a representation issue.
Words in natural language are fundamentally different from pixels in images.
So masking single pixels, or random groups of pixels, is unlikely to bear similar results to masking words.


The first attempt at doing BERT for vision I am aware of is the [Context Encoder (CE) by Pathak et. al.](https://arxiv.org/abs/1604.07379).
A paper from 2016, it actually predates BERT by two years.

[context encoder input output figure]

CE used a small CNN (AlexNet-based) in an encoder-decoder setup.
The images are either masked by a single large-ish rectangle, multiple smaller rectangles or the ground-truth segmentation mask from another image.
While the learned representations are ok, they do not match supervised-like performance when fine-tuned.


It was the [Masked Autoencoder (MAE) by He et. al.](https://arxiv.org/abs/2111.06377) that finally proved that image inpainting can lead to state-of-the-art representations for images.
Coming five years after CE, it did bring in recent advances.
The encoder is a large vision transformer ([ViT, Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929)).
The image is split into a rectangular grid, as in ViT, and a number of grid elements are masked.
This paper provides two insights:

* masking a significant portion of the image (75% in the paper) leads to better representations than masking less, and

* you can simply not feed the masked tokens into the transformer encoder. This not only saves computation (here 3/4, because only 1/4 of the image patches needs to be encoded), but also provides a better representation than feeding \<mask\> tokens in the place of masked patches.

MAE masks consist of small, randomly-scattered rectangles corresponding to the ViT image patches.
They cover 75% of the image, which is significantly more than in CE[^scattered_mask].

Why is this important?
Because masking in images is fundamentally different from masking words in language sentences.
A word represents an entity, its property, or a relation between entities.
A pixel represents a colour.
It's a group of pixels that represents something meaningful like an object, but also a property or a relation.
To mask "red", we need to occlude the majority of a jacket, perhaps leaving its outline.
To mask "jacket" without masking its colour, we can mask its outline but leave a pixel here or there.
To mask the fact that someone is wearing the jacket, we need to mask out a person while leaving fragments of the jacket.
Such groupings are far from random, and are extremely unlikely to occur with random masks.
Unless you mask a significant area of the image, which makes them more likely.

Therefore, MAE masks, which cover a greater image area than CE masks, are better for representation learning.
Indeed, this is what MAE authors found in their paper: masking a greater area (up to a point) leads to better representations.

# How to get word-like masks for images?

Assuming that masking words in BERT is the best you can do for representation learning, the question now is: how do we get image masks whose effect is similar to masking single words in natural language sentences?

Well, we need to identify image regions that are similar in meaning to words.
These can be object bounding boxes or segmentation masks from the respective supervised tasks.
These can also be such labels for object parts.
We can edit them to have masks corresponding to relations (e.g. taking a union of two object masks, diluting or eroding them).

The issue here is that such masks come with datasets for supervised learning: they are human-generated, and therefore expensive.
The whole point of unsupervised representation learning is to not rely on human annotations.

The alternative is to formulate the masking procedure in such a way that will lead to generation of semantically-meaningful masks.
MAE managed to do this partly via occluding large parts of the image at any given time.
If large parts of the image are occluded, it is more likely that a whole object will be occluded (or enough to make it not recognisable)[^todo1].

If we want object-like masks, we should probably take a look at what an object.
As Klaus Greff tell us in his talk at the 2020 ICML workshop on Object-Centric Learning, it is difficult to define what an object is.
But we may think of it in terms of pixels.

[embedd Klaus' talk]
{
<div id="presentation-embed-38930701"></div>
<script src="https://slideslive.com/embed_presentation.js"></script>
<script>
  embed = new SlidesLiveEmbed("presentation-embed-38930701", {
    presentationId: "38930701",
    autoPlay: false,
    verticalEnabled: true,
  });
</script>
}

If you hide a piece of the background, you can easily imagine the hidden part.
If you hide a piece of an object, you can reason about the missing part.
If you hide the whole object, you might have a hard time figuring out what it was, or if there was an object at all.
The only way to do this is to check if it would make sense for any particular object to be there given the visible surroundings.

[embedd the balloon slide from Klaus' talk]

This is because pixels belong to an object are strongly correlated with each other.
Pixels belonging to different objects or an object and the background are not correlated or are correlated only very weakly[^bg_correlation].

It turns out, we can formulate this as a training objective.
Imagine a setup where you try to inpaint an image with some parts occluded.
Given the above, it will be much easier to inpaint a part of an object rather than a whole object.
We can then instantiate a masking model, whose job is to make inpainting as difficult as possible.

I did this in an early prototype a few years ago.
For CLEVR, the result was a mask that covers objects very well, but only early in training.
Later, the mask starts covering parts of background, but in such a way that is suggestive of an object being there.
This confuses the inpainter, and forces it to paint an object where there was none.
This is perhaps ok: the goal is not to get semantic segmentation out of this, but rather semantically-meaningful masks that can force a representation-learning model to reason about objects, properties, or relations.

Since inpainting




#### Acknowledgements

[^blindspot_conjecture]: This is a conjecture on my side. I don't actually know what data the brain uses, but it is plausible that it uses the model of the environment it has.

[^brain_learn]: If you know any references to the contrary, please let me know in the comments or by email. Thanks!

[^scattered_mask]: It is unclear what the impact of such scattered masks is. I guess that it forces the model to reason about multiple things in every image on one hand, and reduces the variance of the gradients on the other. The latter is due to the fact that total occlusion of a certain object is less likely with such scattered masks.

[^todo1]: it is unclear at this stage what masks are the best; probably semantically-meaningful covering parts of objects and perhaps parts of different objects at once, but I'm not entirely clear on what properties make it useful; does this make sense?

[^bg_correlation]: See that, according to above, background behaves just like a big object behind the objects in the foreground.
