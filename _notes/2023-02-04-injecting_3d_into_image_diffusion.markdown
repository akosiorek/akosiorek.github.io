---
layout: draft
title:  "Injecting Geometry into Image Diffusion Models"
date:   2023-02-01 16:23:0 +0000
categories: ml
---

Generative modelling is going through a renaissance, mostly due to diffusion models.
Diffusion models very suddenly became the go-to models for generative modelling of images and videos, especially conditional on text.
There are other models based on GANs and VQVAEs with autoregressive priors: these models didn't die, but I will not go into them here.
These text-to-image models are really impressive; perhaps more impressive than unconditional models because they afford a good degree of control over the image content.
To train them, we need a very large dataset of (text, image) pairs.

And while image models are cool, we would really like to move into generation of 3D scenes. This would unlock additional applications like VR or design, or even just give you a fine-grained control over the image composition (because you could move the camera to a different place).
However, data collection becomes a problem for 3D. There simply isn't enough real world scene data where each scene would be represented by several images corresponding to different camera viewpoints. Capturing such that is expensive, and is unnatural doesn't really exist on the internet. This changes if we go to video, but it open another can of worms because if the scene isn't static, it is diffiult to learn a scene representation that will be consistent across views (that the geometry will be preserved).

So here's the thing. We can play with the text-to-image models by manipulating the text prompt, which then shows that these models actually know about geometry. If this is the case, maybe we don't need all that 3D data? Maybe we can just use the image models and either extract their 3D knowledge, or perhaps nudge them to preserve geometry by injecting a little bit more of that 3D knowledge?

It turns out that both approaches are possible, and correspond to DreamFusion and SceneScape, respectively.
In this blog I'm going to talk about the latter, which injects explicit geometry into a text-to-image diffusion model.

I've been actually working on this idea for a couple of weeks now, but now that it is published (as of 3 days ago), I don't have to work on it any more and can write this blog instead :)

- takeoff of large-scale text-to-image generative models: diffusion + big data
- in principle we could apply similar methods to model 3d scenes or objects, but we don't have data
- but do we really need to train from 3d data? clearly, the 2d image models know about geomtry
- it turns out that we can, at least in some cases, leverage pretrained image diffusion models as priors for 3D
- in this blog, I'm going to talk about one such case: injecting geometry into a text-to-image diffusion model

- what is a text-to-image diffusion model?
- inpainting with these models: a cool feature that comes prepackaged

- algorithm
Assume a camera model that you want and an initial arbitrary position and orientation
Generate an rgbd image
Using the assumed camera model, the known position and the depth project the rgb colour into 3d space to get a point cloud
Pick a desired new position (this can mimic the agent executing a movement action, say)
Reproject that point cloud into the new camera position
Feed the reprojected image into the diffusion model and ask it to inpaint the blanks -> 2.

- in fact, a paper describing it just came out https://scenescape.github.io/ https://arxiv.org/abs/2302.01133 Rafail Fridman, Amit Abecasis, Yoni Kasten, Tali Dekel Weizmann Institute of Science 2NVIDIA Research
- only it turns out that there are rough edges that need to be smoothed out:
  - reprojection from previously captured rgbd images is not great and is much better done by building a mesh as a global scene representation
  - depth is inconcistent, so they fine-tune the depth predictor; so after projecting the mesh on a new camera they fine-tune the depth predictor to agree with the depth that came out from that projection. Once it agrees, they use the predictor for fill in any holes that the projected depth has. So they do optimization at every frame, don't mention how many iterations.
  - the diffusion model they use is based on quantized VQVAE embeddings; they need to finetune the decoder for good reconstruction quality; similarly, they optimize it so that it agrees on the parts that are reprojected and then use the finetuned decoder to fill in any holes (rgb and colour will have the same holes)
  - the inpainted part of the frame may not agree semantically very well with the text prompt; they generate multiple frames and then use cosine distance between the CLIP embeddings of the text and the generated frames to choose the frame that is best aligned with the prompt.

- Limitations:
  - the mesh representation doesn't work well for outdoor scenes (depth disconituities between objects and the sky)
  - there is error accumulation in long generated sequences that screw things up
