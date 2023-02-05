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
Generative modelling is as old as machine learning. But until recently, generative models were a bit like neural nets pre-2012 when AlexNet came out. People knew about them, but kept asking what you could really use them for. Well, 10 years after the original AlexNet, generative modelling had its AlexNet moment marked by the release of StableDiffusion (it even has a [Wiki page](https://en.wikipedia.org/wiki/Stable_Diffusion)), and later, ChatGPT. The revolution really isn't in the technology (which was present for a few years) but rather in the general capability of the models due to their scale and the size of the datasets used, and in their public availability.

See [Sander Dieleman's post](https://benanne.github.io/2022/01/31/diffusion.html) for an intro to diffusion models and [another one](https://benanne.github.io/2022/05/26/guidance.html) on how to make them conditional.

<!-- Diffusion models very suddenly became the go-to models for generative modelling of images and videos, especially conditional on text. -->
<!-- There are other models based on GANs and VQVAEs with autoregressive priors: these models didn't die, but I will not go into them here. -->

Both [language models](https://en.wikipedia.org/wiki/Language_model) (like ChatGPT[^ChatGPT_is_different]) and [text-to-image models](https://en.wikipedia.org/wiki/Text-to-image_model) (like StableDiffusion) have a lot of applications. But thanks to the public availability of these models, new use cases are discovered every day--far beyond what anyone could have expected. And perhaps not surprisingly, the variety of use cases expands much further when you combine a language model with a text-to-image one. And I don't mean in the sense of a joint text + image model, which would require a lot of work. Rather, you can use a language model to seed the text-to-image model with "interesting" prompts. The simple example of this is [prompt engineering](https://stable-diffusion-art.com/chatgpt-prompt/)[^prompt_engineering] to get nice- or interesting-looking pictures. A more advanced would be asking an LM to tell a story while describing the scenary in detail every time it changes (this won't work as is, but you get the idea). Using a text-to-image model we can then translate that story into a sequence of images forming a visual story. This could in principle be used to design a fly-through through a complex environment that gradually changes from one prompt to the next.

[^ChatGPT_is_different]: [Yoav Golberg's post](https://gist.github.com/yoavg/59d174608e92e845c8994ac2e234c8a9) discussess the differences between vanilla language models and ChatGPT.
[^prompt_engineering]: Getting a nice picture out of a text-to-image model may require tinkering with the prompt a bit. It's not as easy as one might think.

<!-- - inpainting with these models: a cool feature that comes prepackaged -->
Such image chaining is possible with diffusion models due to their ability to inpaint missing information (or to do image-to-image translation). See [here for a cool demo](https://ahrm.github.io/jekyll/update/2023/01/02/three-eyed-forehead.html) of inpainting with StableDiffusion. So we can just mask (think erase) a part of an image and ask a diffusion model to fill the blank. The blank will generally be compatible with the unmasked parts of the image and the text prompt used to condition the model[^seeding_masked_parts]. You can read more about the details of how StableDiffusion works, and how inpainting is done, in [this tutorial](https://stable-diffusion-art.com/how-stable-diffusion-work/).
Note that if you can inpaint, you can also outpaint: by simply translating the image to the left, you can pretend that you masked the right side of the image (which is non-existent, but it doesn't matter). The model will complete that right side of the image, effectively extending it.

[^seeding_masked_parts]: You can also place a fragment of a different image in the masked part to seed the result. E.g. in the demo above the author erases a part of the foreground, puts a lamp in there, and let's the model do its magic. The result is a lamp that fits stylistically with the rest of the image.

<!-- notes on inpainting:
- mask a part of the image, add noise (strength in [0, 1]), and denoise
- masking can be black, but probably better fill-in with nearest neighbour colour (flood-fill-like algo)
- the masked region can be seeded by another image; if the masked area is smaller than the unmasked area,
the content will be style-adapted to the unmasked area
- often after inpainting there may be a visible boundary between the masked and unmasked regions; you can then
mask the boundary and inpaint it to have it blend nicely
- inpainting can be conditional as well, usually on text; StableDiffusion v2 allows conditioning on depth (via additional input, it's not trained to model RGBD); it uses [MiDaS](https://github.com/isl-org/MiDaS)

[how stable diffusion works](https://stable-diffusion-art.com/how-stable-diffusion-work/)
[depth to image with SD](https://stable-diffusion-art.com/depth-to-image/): it allows to preserve composition while completely changing the styles
-->

<!-- why we need 3d instead of just images -->
So you have a model that can inpaint and outpaint images; you can change prompts every time you do it, and you can even seed the missing regions with fragments of different pictures. Coming back to our visual story: amazing as the capabilities of text-to-image diffusion models are, you cannot use it to create a coherent visual story. At least not out of the box.
Or maybe you can, but the results will look a bit crappy: this is because you are constrained to zooming in and out and panning the camera, but you cannot actually move within the generated scene. Zooming out scales down everything the same way, but scaling as you move should really depend on the distance from the camera (depth); Also you cannot walk forward, walk through doors, model occlusions or walk around and come back to the same place--the result would not be consistent with the previously-generated images. Here's an example of how zooming out ad inifitum looks like.

<div style="max-width: 400px; display: block; margin: auto;">
  <blockquote class="twitter-tweet" data-lang="en">
    <p lang="en" dir="ltr">
      <a href="https://twitter.com/hardmaru/status/1611569188144807943"></a>
    </p>
  </blockquote>
  <script async="" src="//platform.twitter.com/widgets.js" charset="utf-8"></script>
</div>

To make the above work really well, we would need to model not only views of a given scene (images), but also the geometry (where things are, where was the camera that captured those views). If we have the geometry, we can explicitly move the camera into a new position and capture the next image from there. If you can do this, you unlock a plethora of additional applications like generating whole scenes or 3D assets for virtual reality, computer games or special effects in movies, for interior design, or any other artistic endevour, really.

But building generative models of 3D scenes or objects is not easy. In my work I focused on VAE-based generative models of NeRFs ([NeRF-VAE](https://arxiv.org/abs/2104.00587) and [Laser-NV](https://laser-nv-paper.github.io/index.html)). In principle, these models offer very similar capabilities[^nerf_vae_text_cond]. In practice, the quality of the generated 3D content is far far behind what text-to-image diffusion model generate these days. This is in part due to a different framework (diffusion models instead of VAEs) and in part due to data availability.

[^nerf_vae_text_cond]: We never did text-conditional modelling, but it's easy to add text-conditioning to the prior if you have paired text-3D data.

<!-- - in principle we could apply similar methods to model 3d scenes or objects, but we don't have data -->
In principle, we could take the diffusion modelling techniques used for image generation and apply them to 3D. That's what e.g. [GAUDI](https://arxiv.org/abs/2207.13751) does. However, even then we are limited by the availability of the data.
While it is easy to scrape billions of images and associated text captions from the Internet, this really isn't the case for 3D. To do 3D modelling with NeRF (used in my work and in GAUDI above), you need several images and associated camera viewpoints for every scene, and then you need millions if not billions of scenes in your dataset. This data does not exist on the Internet, because that's not how people take (or post) pictures. Considering the scale, manually capturing such datasets is out of the question. The only respite is video, where different frames are captured from slightly different viewpoints, but video-modelling opens up another can of worms: since the scene isn't static, it is diffiult to learn a scene representation that will be consistent across views (that preserves the geometry). The video diffusion models certainly do not offer multi-view consistency ([Imagen Video](https://imagen.research.google/video/), [Make-a-Video](https://makeavideo.studio/)). Nevertheless, video-modelling with GAUDI-like models is the most promosing direction for future large-scale 3D models.

<!-- - but do we really need to train from 3d data? clearly, the 2d image models know about geomtry -->
But here's the thing. We can play with the text-to-image models by manipulating the text prompt, which then shows that these models actually know about geometry. Perhaps the best example of this is [DreamBooth](https://dreambooth.github.io/).

<figure id='dreambooth'>
  <img style="width: 100%; display: box; margin: auto" src="{{site.url}}/resources/3d_diffusion/dreambooth.png" alt="DreamBooth"/>
  <figcaption align='center'>
  <b>Fig 4:</b> <a href="https://dreambooth.github.io/">DreamBooth</a> allows to associate a specific object with a text token and then and place that token within different text promps.
  </figcaption>
</figure>

If text-to-image models really know about 3D geometry, maybe we don't need all that 3D data? Maybe we can just use the image models and either extract their 3D knowledge, or perhaps somehow nudge them to preserve geometry across multiple generated images.
<!-- - it turns out that we can, at least in some cases, leverage pretrained image diffusion models as priors for 3D -->
It turns out that both approaches are possible, do not require re-training the text-to-image models, and correspond to [DreamFusion](https://dreamfusion3d.github.io)/[Score Jacobian Chaining](https://pals.ttic.edu/p/score-jacobian-chaining) and [SceneScapes](https://scenescape.github.io), respectively.

<!-- - in this blog, I'm going to talk about one such case: injecting geometry into a text-to-image diffusion model -->
In this blog I'm going to talk about the latter, which injects explicit geometry into a text-to-image diffusion model.
I've been actually working on this idea for a couple of weeks now, but now that it is published (as of 3 days ago), I don't have to work on it any more and can write this blog instead :)


<!-- - algorithm -->

A naive version of the SceneScapes algorithm requires:
- a pretrained text-to-image diffusion model capable of inpainting missing values,
- a pretrained depth-from-a-single-image predictor,
- a method to infer intrinsic camera parameters for an RGBD image,
- and a text prompt,
- optionally also an image to start from.

We then do the following:
1. Generate an initial image (or use the one you want to start with). Initialize the camera position and orientation to an arbitrary value.
2. Predict the depth for that image.
3. Infer instrinsics for the RGBD image that you now have. You will only have to do this once as hopefully the diffusion model will preserve the camera parameters when inpainting missing values.
4. Change the camera position and orientation.
5. Project the previously-generated RGBD images onto the new camera pose (this is where intrinsics come into play). It will contain holes.
6. Feed the projected RGB image into the diffusion model and fill in any missing values. Go to step 2.

<figure id='scenescape'>
  <img style="width: 100%; display: box; margin: auto" src="{{site.url}}/resources/3d_diffusion/scenescapes.png" alt="SceneScape"/>
  <figcaption align='center'>
  <b>Fig 4:</b> <a href="https://scenescape.github.io/">SceneScape</a> is a bit more advanced than the simplified algorithm described above, but the idea is the same.
  </figcaption>
</figure>

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
