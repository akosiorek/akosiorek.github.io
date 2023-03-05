---
layout: draft
title:  "Geometry in Text-to-Image Diffusion Models"
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

See Sander Dieleman's blog for an [intro to diffusion models]((https://benanne.github.io/2022/01/31/diffusion.html)) and a guide on [how to make them conditional](https://benanne.github.io/2022/05/26/guidance.html).

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

<!-- This is a bit sad, because we know that text-to-image diffusion models, at least the large-scale ones, knows about geometry. -->

To make the above work really well, we would need to model not only views of a given scene (images), but also the geometry (where things are, where was the camera that captured those views). If we have the geometry, we can explicitly move the camera into a new position and capture the next image from there. If you can do this, you unlock a plethora of additional applications like generating whole scenes or 3D assets for virtual reality, computer games or special effects in movies, for interior design, or any other artistic endevour, really.

But building generative models of 3D scenes or objects is not easy. In my work I focused on VAE-based generative models of NeRFs ([NeRF-VAE](https://arxiv.org/abs/2104.00587) and [Laser-NV](https://laser-nv-paper.github.io/index.html)). In principle, these models offer very similar capabilities[^nerf_vae_text_cond]. In practice, the quality of the generated 3D content is far far behind what text-to-image diffusion model generate these days. This is in part due to a different framework (diffusion models instead of VAEs) and in part due to data availability.

[^nerf_vae_text_cond]: We never did text-conditional modelling, but it's easy to add text-conditioning to the prior if you have paired text-3D data.

<!-- - in principle we could apply similar methods to model 3d scenes or objects, but we don't have data -->
In principle, we could take the diffusion modelling techniques used for image generation and apply them to 3D. That's what e.g. [GAUDI](https://arxiv.org/abs/2207.13751) does. However, even then we are limited by the availability of the data.
While it is easy to scrape billions of images and associated text captions from the Internet, this really isn't the case for 3D. To do 3D modelling with NeRF (used in my work and in GAUDI above), you need several images and associated camera viewpoints for every scene, and then you need millions if not billions of scenes in your dataset. This data does not exist on the Internet, because that's not how people take (or post) pictures. Considering the scale, manually capturing such datasets is out of the question. The only respite is video, where different frames are captured from slightly different viewpoints, but video-modelling opens up another can of worms: since the scene isn't static, it is diffiult to learn a scene representation that will be consistent across views (that preserves the geometry). The video diffusion models certainly do not offer multi-view consistency ([Imagen Video](https://imagen.research.google/video/), [Make-a-Video](https://makeavideo.studio/)). Nevertheless, video-modelling with GAUDI-like models is the most promosing direction for future large-scale 3D models.

### Text-to-Image Models Known About Geometry

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
It turns out that both approaches are possible, do not require re-training the text-to-image models, and correspond to extracting geometry from an image model ([DreamFusion](https://dreamfusion3d.github.io) and [Score Jacobian Chaining (SJC)](https://pals.ttic.edu/p/score-jacobian-chaining)), and injecting geometry into an image model ([SceneScapes](https://scenescape.github.io)), respectively.


### Extracting Geometry from an Image Model
Given that a text-to-image diffusion model knows about geometry (see DreamBooth above, though we don't need DreamBooth for this), it is reasonable to expect that we can use such a modal as a prior over 3D representations. Why? Because any 2D rendering of a 3D representation is an image, and if that representation contains an object that is similar to the training set of the image model, that rendered image should have high likelihood under the image model. Conversly, if the represented object is different from what the image model is trained on, the rendered image will have a low likelihood. Therefore, if we start from a representation of a random object, the generated images will have a low likelihood. But if we then manage to compute the gradients of the image model likelihood with respect to the 3D representation, we'll be able to nudge the 3D representation into something that has a bit higher likelihood under that image model. In practice, DreamFusion and SJC use pre-trained large-scale text-to-image diffusion models and evaluate the gradient with respect to the image as the score-matching objective, but it doesn't really matter. Any image model that is able to score a generated image is will do (a VAE, a discriminator of a GAN, or [even your classifier](https://arxiv.org/abs/1912.03263); it'd better be conditional, though; see the end of the blog).

<figure id='dreamfusion_algo'>
  <img style="width: 100%; display: box; margin: auto" src="{{site.url}}/resources/3d_diffusion/dreamfusion_algo.png"/>
  <figcaption align='center'>
  <b>Fig 5:</b> Extracting geometry from an text-to-image model into a NeRF, taken from <a href="https://dreamfusion3d.github.io">DreamFusion</a>.
  </figcaption>
</figure>

The specific algorithm is as follows (the DreamFusion version):
1. Initialize a random NeRF.
2. Pick a random camera pose.
3. Render an image at that camera pose using the NeRF.
4. Compute the score-matching loss under a pre-trained diffusion model.
5. Use the score-matching loss as a gradient with respect to the rendered image, and backpropagate it to NeRF's parameters.
6. Go to step 2.

Of course life is never that easy, and DreamFusion comes with a number of hacks, including: clipping the scene represented by the NeRF to a small ball around the origin (any densities outside of the ball are set to zero), putting the rendered object on different backgrounds, additional losses that ensure e.g. that most of the space is unoccupied.

<figure id='sjc_examples'>
  <img style="width: 100%; display: box; margin: auto" src="{{site.url}}/resources/3d_diffusion/sjc_examples.png"/>
  <figcaption align='center'>
  <b>Fig 6:</b> Images + depth maps generated by extracting geometry from StableDiffusion, taken from <a href="https://pals.ttic.edu/p/score-jacobian-chaining">Score Jacobian Chaining</a>.
  </figcaption>
</figure>

As you can see in the above examples, extracting geometry from image models is able to, so far, produce cartoon-ish looking 3D models of single objects. Why is that? While noone really knows, I have some theories. The first hint is that this approach relies on classifier-free guidance with a very high guidance weight, which decrease the variance of the distribution (and its multimodality, see the end of this blog for a further discussion). The second bit is associated with the distribution of the cameras. If you are trying to model a general 3D scene (a part of a city or an apartment), the distribution of viable cameras is tightly coupled to the layout of the scene. In an apartment, say, randomly sampling cameras will yield cameras that are within walls and other objects. This will result in an empty image, which is unlikely under the model. Optimization in such case will lead to removing any objects that occlude the scene from the camera: in this case, it will remove everything, resulting in an empty scene.

This approach predates injecting geometry into a diffusion model (described below), and already has view-conditioned follow-ups that allow generating a NeRF out of a single image.

[RealFusion](https://arxiv.org/abs/2302.10663) is a view-conditioned version of DreamFusion. It does everything that DreamFusion does, but instead of a vanilla text-to-image diffusion model the authors use DreamBooth to constrain the diffusion model to a specific object shown by a target image. This constrains the diffusion model distribution to represent images of just a single object. In addition to forcing the NeRF to represent that object, it should result in lower-variance gradients for the NeRF and therefore better NeRF quality.

[NerfDiff](https://arxiv.org/abs/2302.10109) is similar, but instead of fitting a NeRF from scratch, they train a view-conditioned (amortized) NeRF. Another difference is that instead of using a pretrained text-to-image diffusion model, they fit a new view-conditioned diffusion model (so conditioning is an image, not text) jointly with the amortized NeRF. Why? Because diffusion models tend to achieve much better image quality than amortized NeRFs at the cost of not being consistent across different views. The amortized NeRF allows a fast NeRF initialization from a single image, which is than fine-tuned with distillation from the diffusion model. The authors also introduce a novel distillation algorithm that improves on DreamFusion/SJC quite a bit (but is quite a bit more expensive). NerfDiff can produce NeRFs only from images that are similar to the training images; RealFusion doesn't have this issue because it uses a pretrained large-scale diffusion model.

### Injecting Geometry into an Image Model
I've actually started working on this idea a couple of weeks before [SceneScapes](https://scenescape.github.io) was published. Now, with the problem mostly solved, I can describe the (published) approach in this blog :)

The main idea behind the SceneScapes algorithm is that an image diffusion model is able to correct image imperfections with its superb inpainting abilities. Now imagine that we have an image captured from a given camera position, and we pretend to move to a different camera position. We can now warp the image we have into what that image would look like if seen from the new camera position. This image will be imperfect:
- Specularities and other view-dependent lighting effects will be incorrect.
- It will have holes because not everything was observed.

But mostly, the image will look ok. The diffusion model can fill in the holes, and possibly even fix the lighting artifacts: there you go, we just created a new image, taken from a different camera position, that is geometrically consistent (distances are the same) and semantically consistent (the things visible in the first image are still there and are the same). The best part? We used an off-the-shelf pretrained image model. It doesn't even have to be a diffusion model: all we need is the inpainting ability.

<figure id='scenescape_examples'>
  <img style="width: 100%; display: box; margin: auto" src="{{site.url}}/resources/3d_diffusion/scenescape_example.png"/>
  <figcaption align='center'>
  <b>Fig 7:</b> <a href="https://scenescape.github.io/">SceneScape</a> is a bit more advanced than the simplified algorithm described above, but the idea is the same.
  </figcaption>
</figure>

#### Technical: SceneScapes Algorithm

<!-- - algorithm -->
A naive version of the [SceneScapes](https://scenescape.github.io) algorithm requires:
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
  <b>Fig 8:</b> <a href="https://scenescape.github.io/">SceneScape</a> is a bit more advanced than the simplified algorithm described above, but the idea is the same.
  </figcaption>
</figure>

<!-- things needed to make it work -->
Only it turns out that there are rough edges that need to be smoothed out (as done in the paper):
- Reprojection from previously captured RGBD images is not great and is much better done by building a mesh as a global scene representation.
- The depth predicted from single images is inconsistent across the images (the differences between depth do not respect the changes in camera position), so the authors fine-tune the depth predictor: after projecting the mesh on a new camera they fine-tune the depth predictor to agree with the depth that came out from that projection. Once the depth predictor agrees with the mesh, we can predict the values for the holes in the depth map. This requires optimization of the depth-predictor at every generated frame. The authors don't mention how many gradient steps it takes.
- The diffusion model used in the paper is based on quantized VQVAE embeddings (I'm not sure but I'm guessing it's StableDiffusion, which follows Latent Diffusion?). Since VQVAE autoencoding results is somewhat low reconstruction quality, the authors need to finetune the VQVAE decoder as well to obtain good reconstruction quality. Similarly to the depth predictor, they first optimize it so that it agrees on these parts of the image that are reprojected from the mesh and then use the finetuned decoder to fill in any holes (RGB and depth will have the same holes).
- Lastly, the inpainted part of the frame may not agree semantically with the text prompt very well; they generate multiple frames and then use cosine distance between the CLIP embeddings of the text and the generated frames to choose the frame that is best aligned with the prompt.

Limitations:
- The mesh representation doesn't work well for outdoor scenes (depth disconituities between objects and the sky).
- There is error accumulation in long generated sequences that sometimes lead to less-than-realistic results.

### Why is it important for the image model to be text conditioned?
I left this discussion until after describing the two approaches of extracting and injecting geometry, because it requires understanding some technical details about how these methods work.

Generally speaking, modelling conditional probability distributions is easier than modelling unconditional ones. This may seem counter-intuitive at first, because to modal a conditional probability $$p(x \mid z)$$ you have to learn the relationship between $$x$$ and $$z$$, which you don't have to do if you are modelling just $$p(x)$$. While that is true, $$p(x)$$ is generally a much more complicated object than $$p(x \mid z)$$. To see this, let's look at a Gaussian mixture with K components. In this case, to recover the true $$p(x)$$ with a learned $$\widetilde{p}(x)$$, we have to parametrize $$\widetilde{p}(x)$$ with a family of distributions expressive enough to cover the 10 different modes.  If, however, we model the conditional $$p(x \mid z)$$ where $z$ now is an index telling us which mode we care about, the learned $$\widetilde{p}(x \mid z)$$ has to model just one mode at a time. In this example, it can be just Gaussian. A larger-scale example is that of ImageNet with a 1000 different classes. In that case you can think of the data distribution as a mixture of a 1000 components, but now the components are very high-dimensional (images of shape 224x224x3), and the individual components are highly non-Gaussian, so the problem is much more difficult. Modelling conditionals in this case is way simpler.

So what does this have to do with image models and geometry?

I did some basic experiments with a DreamFusion-like setup, and where I played with an unconditional image model as well (trained from scratch on a smaller dataset), where I was interested in inferring NeRFs of whole scenes rather than single objects. It turns out that. It turns out that if the image model is unconditional, the gradients that it produces to train the NeRF point in a multitude of different directions. What happens in practice is that the NeRF initially starts to represent a scene, but eventually that scene disappears and the NeRF represents just empty space. This changes when we introduce conditioning: either text describing object (like in DreamFusion or SJC), or an image (like in RealFusion or NerfDiff). The bottom line: too many modes lead to too high variance of the gradients used to train the NeRF. Decreasing the number of modes leads to better-behaved gradients and thus learning.

A very similar argument applies to injecting geometry into an image model. One of the limitations of SceneScape is the accumulation of errors. This is partly mitigated by generating more than just one inpainting of the image from a new camera position, and then choosing the one that best aligns with the **text prompt**. So the image model had many more modalities (if it was unconditional), it would be much more likely to inpaint missing parts of the image in a way that is not very consistent with the presented image, leading to faster error accumulation. If the model wasn't text-conditioned, the authors couldn't have done the CLIP trick of choosing the most suitable image in the first place, which would have significantly exacerbated the error accumulation.

So we see that the ability to model insanely complex distribution (unconditional distributions of real images) is actually counter-productive. Perhaps that's ok, because whenever we want to generate an image, we would like to have some control over what we're generating. However, this suggests a future failure case. As the generative models get bigger, more expressive, and trained on more data, they will represent distributions with more and more modes. This is true even for conditional models. Does this mean that with the advances in generative modelling, the approaches of injecting and extracting geometry (and anything that requires constraining the variance of the distribution) will stop working? As with anything, there will be workaround. But it's an interesting failure cases and it might be useful to keep it in mind.



### Conclusions

While I'm not sure what I actually said with this blog, what I wanted to say is this[^gaiman]:

[^gaiman]: This is a paraphrase of Neil Gaiman from one of his speeches, taken from his book "The View from the Cheap Seats: Selected Nonfiction".

#### Acknowledgements

### Footnotes
