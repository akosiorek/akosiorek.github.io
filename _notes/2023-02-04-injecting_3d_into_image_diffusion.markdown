---
layout: draft
title:  "Injecting Geometry into Image Diffusion Models"
date:   2023-02-01 16:23:0 +0000
categories: ml
---
<!--
Outline:
- takeoff of large-scale text-to-image generative models: diffusion + big data
- inpainting with these models: a cool feature that comes prepackaged
- why we need 3d instead of just images
- in principle we could apply similar methods to model 3d scenes or objects, but we don't have data
- but do we really need to train from 3d data? clearly, the 2d image models know about geomtry
- it turns out that we can, at least in some cases, leverage pretrained image diffusion models as priors for 3D
- in this blog, I'm going to talk about one such case: injecting geometry into a text-to-image diffusion model

- algorithm
things needed to make it work
- limiations
closing thoughts
-->

<!-- takeoff of large-scale text-to-image generative models: diffusion + big data -->
Generative modelling is as old as machine learning. But until recently, generative models were a bit like neural nets pre-2012 when AlexNet came out. People knew about them, but kept asking what you could really use them for. Well, 10 years after the original AlexNet, generative modelling had its AlexNet moment marked by the release of StableDiffusion (even has a [Wiki page](https://en.wikipedia.org/wiki/Stable_Diffusion)), and later, ChatGPT. The revolution really isn't in the technology (which was present for a few years now, but rather in availability of the tech: such huge and well-working generative models were never available for public use).

<!-- Diffusion models very suddenly became the go-to models for generative modelling of images and videos, especially conditional on text. -->
<!-- There are other models based on GANs and VQVAEs with autoregressive priors: these models didn't die, but I will not go into them here. -->

Both [language models](https://en.wikipedia.org/wiki/Language_model) (like ChatGPT[^ChatGPT_is_different]) and [text-to-image models](https://en.wikipedia.org/wiki/Text-to-image_model) (like StableDiffusion) have a lot of applications. The possible usecases expand much further when you combine the two. And I don't mean in the sense of a joint text + image model. Rather, you can use a language model to seed the text-to-image model with "interesting" prompts (ChatGPT, tell me a story and describe the scene in detail every time it changes) that will then translate into a sequences of images telling a visual story. This could in principle be used to design a fly-through through a complex environment that gradually changes from one prompt to the next.

[^ChatGPT_is_different]: [Yoav Golberg's post](https://gist.github.com/yoavg/59d174608e92e845c8994ac2e234c8a9) discussess the differences between vanilla language models and ChatGPT.

<!-- - inpainting with these models: a cool feature that comes prepackaged -->
This is generally possible because
TODO: inpainting capabilities

<!-- why we need 3d instead of just images -->
But here's the thing: a text-to-image model cannot do that. You cannot make it, at least not without retraining, to generate a picture consistent with another picture, unless the two pictures overlap (but even then it won't be able to reason about the whole picture it should be consistent with). Or maybe you can, but the results look a bit crappy: this is because just zooming out scales everything the same way, but scaling should really depend on the distance from the camera; Also you cannot walk forward, walk through doors, model occlusions or walk around and come back to the same place--the result would not be consistent. Here's an example of how zooming out ad inifitum looks like.

hardmaru's tweet:
<div class='jekyll-twitter-plugin' align="center">
    {% twitter https://twitter.com/hardmaru/status/1611641202654457856 maxwidth=500 limit=5 %}
</div>

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">halfin (@halfin) <a href="https://twitter.com/halfin/status/1110302988">https://twitter.com/halfin/status/1110302988</a></p>&mdash; halfin (@halfin) <a href="https://twitter.com/halfin/status/1110302988"></a></blockquote>
<script async="" src="//platform.twitter.com/widgets.js" charset="utf-8"></script>


To make the above work really well, we would need to model not only views of a given scene (images), but also the geometry (where things are, where was the camera that captured those views). If we have the geometry, we can explicitly move the camera into a new position and capture the next image from there. If you can do this, you unlock a plethora of additional applications like generating whole scenes or 3D assets for virtual reality, computer games or special effects in movies, for interior design, or even any other artistic endevour, really.

But this is not easy. In my work I focused on VAE-based generative models of NeRF ([NeRF-VAE](https://arxiv.org/abs/2104.00587) and [Laser-NV](https://laser-nv-paper.github.io/index.html)). In principle, these models offer all the above capabilities (we never did text-conditional modelling, but it's easy to add text-conditioning to the prior). In practice, the quality of the generated 3D content is far far behind what text-to-image diffusion model generate these days. This is in part due to a different framework (diffusion models instead of VAEs) and in part due to data availability.

<!-- - in principle we could apply similar methods to model 3d scenes or objects, but we don't have data -->
In principle, we could take the diffusion modelling techniques used for image generation and apply them to 3D. That's what e.g. [GAUDI](https://arxiv.org/abs/2207.13751) does. However, even then we are limited by the availability of the data.
While it is easy to scrape billions of images and associated text captions from the Internet, this really isn't the case for 3D. To do 3D modelling with NeRF (used in my work and in GAUDI above), you need several images and associated camera viewpoint for every scene, and then you need millions if not billions of scenes in your dataset. This data does not exist on the Internet, because that's not how people take picture (or post them online). Considering the scale, manually capturing such datasets is out of the question. The only respite is video, where different frames are captured from slightly different viewpoints, but it opens up another can of worms: since the scene isn't static, it is diffiult to learn a scene representation that will be consistent across views (that preserves the geometry). Nevertheless, video-modelling with GAUDI-like models is the most promosing direction for future large-scale 3D models.

<!-- - but do we really need to train from 3d data? clearly, the 2d image models know about geomtry -->
But here's the thing. We can play with the text-to-image models by manipulating the text prompt, which then shows that these models actually know about geometry. If this is the case, maybe we don't need all that 3D data? Maybe we can just use the image models and either extract their 3D knowledge, or perhaps nudge them somehow to preserve geometry across multiple generated images.

<!-- - it turns out that we can, at least in some cases, leverage pretrained image diffusion models as priors for 3D -->
It turns out that both approaches are possible, do not require re-training the text-to-image models, and correspond to [DreamFusion](https://dreamfusion3d.github.io)/[Score Jacobian Chaining](https://pals.ttic.edu/p/score-jacobian-chaining) and [SceneScapes](https://scenescape.github.io), respectively.

<!-- - in this blog, I'm going to talk about one such case: injecting geometry into a text-to-image diffusion model -->
In this blog I'm going to talk about the latter, which injects explicit geometry into a text-to-image diffusion model.
I've been actually working on this idea for a couple of weeks now, but now that it is published (as of 3 days ago), I don't have to work on it any more and can write this blog instead :)



<!-- - algorithm -->
Assume a camera model that you want and an initial arbitrary position and orientation
Generate an rgbd image
Using the assumed camera model, the known position and the depth project the rgb colour into 3d space to get a point cloud
Pick a desired new position (this can mimic the agent executing a movement action, say)
Reproject that point cloud into the new camera position
Feed the reprojected image into the diffusion model and ask it to inpaint the blanks -> 2.

<!-- things needed to make it work -->
- in fact, a paper describing it just came out https://scenescape.github.io/ https://arxiv.org/abs/2302.01133 Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel Weizmann Institute of Science 2NVIDIA Research
- only it turns out that there are rough edges that need to be smoothed out:
  - reprojection from previously captured rgbd images is not great and is much better done by building a mesh as a global scene representation
  - depth is inconcistent, so they fine-tune the depth predictor; so after projecting the mesh on a new camera they fine-tune the depth predictor to agree with the depth that came out from that projection. Once it agrees, they use the predictor for fill in any holes that the projected depth has. So they do optimization at every frame, don't mention how many iterations.
  - the diffusion model they use is based on quantized VQVAE embeddings; they need to finetune the decoder for good reconstruction quality; similarly, they optimize it so that it agrees on the parts that are reprojected and then use the finetuned decoder to fill in any holes (rgb and colour will have the same holes)
  - the inpainted part of the frame may not agree semantically very well with the text prompt; they generate multiple frames and then use cosine distance between the CLIP embeddings of the text and the generated frames to choose the frame that is best aligned with the prompt.

- Limitations:
  - the mesh representation doesn't work well for outdoor scenes (depth disconituities between objects and the sky)
  - there is error accumulation in long generated sequences that screw things up
