---
layout: draft
title:  "Masking for Representation Learning in Vision"
date:   2022-06-20 08:23:0 +0000
categories: ml
---

<!-- # On Masking for Representation Learning in Vision -->

Masked-image modelling (MIM) is about covering parts of an image, and then trying to recreate them from what is left.
Recently, it led to state-of-the-art representation learning in images.
In this blog, I will dive into why masked images deliver such a powerful learning signal, and think about what may constitute a good mask.
But first, let's start with some motivation.

# Masking and the Brain

Have you ever covered an object you see with your hand, and tried to imagine how the covered part looks like?
If not, why won't you try?
You may not be able to draw it or to paint it, since that requires considerable skill.
But you may not even be able to see it clearly in your mind's eye.
Yet, you know what it is or what it can be used for---you have a good representation of it.
Getting such representations is, roughly, the goal behind masked-image modelling (MIM).

Trying to reconstruct the hidden part from the visible parts is called image inpainting, or more generally, missing-data imputation[^VAE-AC].
MIM models are usually trained via image inpainting, that is, they are trained to reconstruct occluded parts in an image.
As we will see later on, this is not always necessary.


The fact that you cannot recreate an object in your head may suggest that your brain's generative model is not perfect.
It turns out, though, that your brain is doing image inpainting all the time without you even noticing.

<figure id='blind_spot'>
  <img style="display: box; margin: auto" src="{{site.url}}/resources/masked_image_modelling/blind_spot.webp" alt="blind spot"/>
  <figcaption align='center'>
  <b>Fig 1:</b> Blind spot of the human eye. The illustration is thanks to <a href="http://george-retseck.squarespace.com/">George Retseck</a>.
  </figcaption>
</figure>

Each of your eyes has a visual blind spot see [Fig. 1](#blind_spot).
It's roughly in the middle vertically, slightly off-centre to the outside for each eye.
You don't see anything there, because it's the place where the visual nerve connects to the eye, leaving no place for photoreceptors.
Check out the exercise on [Wikipedia](https://en.wikipedia.org/wiki/Blind_spot_(vision)) to see for yourself.
And yet, you are not aware that any information is missing: you seem to see what is hidden.

While I have no reference to prove it, it seems that the brain is inpainting the occluded area.
This may be based on what is around that area, but also using the view from the other eye (novel view synthesis), and what the brain is expecting to see in a given context.

I expect that the brain had to learn how to inpaint the blind spot.
Wouldn't it be cool, if this actually helped the brain to form good visual representations?
This is a conjecture, but given the SOTA representation learning results of MIM models, I wouldn't be surprised if it was true.

# BERT or Why Inpaint for Representation Learning?

I got first interested in MIM in 2018, right after [BERT of Devlin et. al.](https://arxiv.org/abs/1810.04805) came out.
BERT is a large transformer trained to fill-in missing words in natural language sentences based on the available words.
<!-- This works, because the model has to learn dependencies between different words.
It needs to predict which words make sense in the presence of other words. -->
Why is this useful?
Because words represent concrete objects or abstract entities, their properties and relations between them.
To predict which word makes sense in the presence of other words, is to analyse what objects and with what properties are represented in that sentence, and what are the relations between them.
A model that learns to do that, learns many truths about the world.


So why not do this for vision?
Well, you can, but there are issues.
First, an architectural issue is that BERT used a transformer, which is really good at reasoning about relations between different elements.
Until fairly recently, though, there was no good way of using a transformer for vision.
Second, a representation issue.
Words in natural language are fundamentally different from pixels in images.
So masking single pixels, or random groups of pixels, is unlikely to bear similar results to masking words.


The first attempt at doing BERT for vision I am aware of is the [Context Encoder (CE) by Pathak et. al](https://arxiv.org/abs/1604.07379). As a paper from 2016, it actually predates BERT by two years.

<figure id='context_encoder_in_out'>
  <img style="display: box; margin: auto" src="{{site.url}}/resources/masked_image_modelling/context_encoder_input_output.png" alt="blind spot"/>
  <figcaption align='center'>
  <b>Fig 2:</b> Context Encoder; from left: masked input, reconstruction, three examples of different masks used for CE.
  </figcaption>
</figure>

CE used a small CNN (AlexNet-based) in an encoder-decoder setup.
The images are either masked by a single large-ish rectangle, multiple smaller rectangles or the ground-truth segmentation mask from another image.
While the learned representations are ok, they are far behind supervised models of the time in performance, even when fine-tuned.


It was the [Masked Autoencoder (MAE) by He et. al.](https://arxiv.org/abs/2111.06377) that finally proved that image inpainting can lead to state-of-the-art representations for images.
Coming five years after CE, it did bring in recent advances.
The encoder is a large vision transformer ([ViT, Dosovitskiy et. al.](https://arxiv.org/abs/2010.11929)).
The image is split into a rectangular grid, as in ViT, and a number of grid elements are masked.
This paper provides two insights:

* The representation quality improves with the fraction of the image masked (up to a point).

* Instead of feeding an image with masked parts to the encoder, it is better to just not use the masked parts as an input.
This is easy to do for an image-divided-into-patches and a transformer like in ViT, but next to impossible for a CNN.

<figure id='mae'>
  <img style="width: 75%; display: box; margin: auto" src="{{site.url}}/resources/masked_image_modelling/mae.png" alt="MAE architecture"/>
  <figcaption align='center'>
  <b>Fig 3:</b> MAE architecture; note that the masked patches are not fed into the encoder.
  </figcaption>
</figure>

MAE masks consist of small, randomly-scattered rectangles corresponding to the ViT image patches.
They cover 75% of the image, which is significantly more than in CE[^scattered_mask].

Why is this important?
Because masking in images is fundamentally different from masking words in language sentences.
A word represents an entity, its property, or a relation between entities.
A pixel represents a colour.
It's a group of pixels that represents something meaningful like an object, but also a property or a relation.
Imagine a man wearing a red jacket.

* To mask "red", we need to occlude the majority of a jacket, perhaps leaving its outline.

* To mask "jacket" without masking its colour, we can mask its outline but leave a pixel here or there.

* To mask the fact that someone is wearing the jacket, we need to mask out a person while leaving fragments of the jacket.

Such groupings are far from random, and are extremely unlikely to occur with random masks.
Masking a significant area of the image, like in MAE, makes it easier to occlude whole entities.
As the paper shows, such masks are also better for representation learning.
Still, masking properties or relations remains difficult under that scheme.


# How to get word-like masks for images?

Assuming that masking single words in natural language sentences is the best you can do for representation learning, the question now is: how do we get image masks with effects similar to masking single words in natural language sentences?

Well, we need to identify image regions that are similar in meaning to words.
Object bounding boxes or segmentation masks would be a good choice if not for two issues.
First, they are human generated, which defeats the purpose of unsupervised learning.
Second, they usually cover objects, with no masks or boxes describing relations between objects or parts thereof[^mask_editing].

[^mask_editing]: The latter could be perhaps circumvented by editing ground-truth masks, e.g. taking a union of two object masks, diluting or eroding masks, etc.

<figure id='klaus_talk'>
  <div id="presentation-embed-38930701" style="width: 10%;"></div>
  <script src="https://slideslive.com/embed_presentation.js"></script>
  <script>
    embed = new SlidesLiveEmbed("presentation-embed-38930701", {
      presentationId: "38930701",
      autoPlay: false,
      verticalEnabled: false,
    });
  </script>
  <figcaption align='center'>
  "What are Objects" by Klaus Greff, Object-Oriented Learning Workshop at ICML 2020.
  </figcaption>
</figure>

The alternative is to think about the properties of the desired masks.
If we want object-like masks, we should probably take a look at what an object is---a good place to start is Klaus Greff's talk from the 2020 ICML workshop on Object-Centric Learning above.

<figure id='masked_balloons'>
  <img style="display: box; margin: auto" src="{{site.url}}/resources/masked_image_modelling/masked_balloons.png" alt="masked balloons"/>
  <figcaption align='center'>
  <b>Fig 3:</b> Inpainting a part of an object or background is easy. Inpainting a whole object is difficult. Adapted from Klaus' talk above.
  </figcaption>
</figure>

Look at the air baloon figure above.
If you hide a piece of the background, in this case an empty sky, you can easily imagine the hidden part.
If you hide a piece of an object, you can reason about the missing part.
If you hide the whole object, you might have a hard time figuring out what it was, or if there was an object at all.
The only way to do this is to check if it would make sense for any particular object to be there given the visible surroundings.

This is because pixels belonging to an object are strongly correlated with each other.
Pixels belonging to different objects or an object and the background are not correlated or are correlated only very weakly[^bg_correlation].
By now, this is a widely-accepted view.
I would go a step further, and say that pixels representing a relation (e.g. two objects that often appear together), or a property, are also strongly correlated, therefore possible to infer from a partial observation.

The above intuition can be formalised as a training objective.
Imagine a setup where you try to inpaint an image with some parts occluded.
We can instantiate a masking model, whose job is to make inpainting as difficult as possible.
The result?
You get masks that seem to hide objects or their parts.
You also get better representation learning results than with using MAE's masks[^learned_masks_for_mae].

[^learned_masks_for_mae]: The caveat is that using such learned masks requires feeding the whole image into the encoder. This results in a significantly increased computation cost for MAE and might not be practical.

I did this in an early prototype a few years ago.
For CLEVR, the result was a mask that covers objects very well, but only early in training.
Later, the mask starts covering parts of background, but in such a way that is suggestive of an object being there.
This confuses the inpainter, and forces it to paint an object where there was none.
This is perhaps ok: the goal is not to get semantic segmentation out of this, but rather semantically-meaningful masks that can force a representation-learning model to reason about objects, properties, or relations.

The main disadvantage of MIM is that you need to reconstruct the image.
Recall that when you cover an object, you might not be able to reconstruct in mentally, but you still know what it is.
Fortunately, we can combine the adversarial masking idea with siamese-style representation learning, which is reconstruction-free.

The result? Meet [ADIOS](https://arxiv.org/abs/2201.13100)!

# [Adversarial Masking for Self-Supervised Learning (ADIOS)](https://arxiv.org/abs/2201.13100)


#### Acknowledgements


#### Footnotes

[^VAE-AC]: ["Variational Autoencoder with Arbitrary Conditioning" by Ivanov et. al.](https://arxiv.org/abs/1806.02382) was the first paper that got me thinking about image inpainting.

[^brain_learn]: If you know any references to the contrary, please let me know in the comments or by email. Thanks!

[^scattered_mask]: It is unclear what the impact of such scattered masks is.
It might force the model to reason about multiple things in every image.
It may also reduces the variance of the gradients due to the fact that total occlusion of a certain object is less likely with such scattered masks.

[^bg_correlation]: See that, according to above, background behaves just like a big object behind the objects in the foreground.
