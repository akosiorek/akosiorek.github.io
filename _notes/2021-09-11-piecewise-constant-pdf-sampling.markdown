---
layout: draft
title: "On Differentiable Sampling Piecewise Constant PDFs"
date: 2021-09-11 14:49:0 +0000
comments: False
share: False
categories: ml

If you are anything like me, you probably never thought much about piecewise-constant porbability density functions (PDFs).
But you do see them every now and then.
For example, whenever you look at a histogram (e.g. colour histogram of an image), you are looking at a (potentially unnormalized) piecewise-constant PDF.

As it turns out, such PDFs are much more useful than I (and maybe you?) imagined, and they deserve to be known and used by ML practitioners.
In this post, I will describe how piecewise-constant PDFs are used in  Neural Radiance Fields (NeRF).
We will then look at three potential applications of piecewise-constant PDFs.
The first one deals with NeRF, and shows how it can be improved by slightly tweaking the PDF.
The remaining ones deal with image segmentation and continous prediction (regression) problems.

# Piecewise-constant PDFs in Neural Radiance Fields (NeRF)
To explain this application, I need to explain what NeRF is, so bear with me.

NeRF is a method to represent the colour and geometry of a 3D scene.
Specifically, NeRF is a function $$f(\mathbf{x}, \mathbf{d}) -> \mathbf{c}, \sigma$$ which outputs the colour $$\mathbf{c}$$ and a density value $$\sigma$$ at a 3D position $$\mathbf{x}$$ as observed from a direction $$\mathbf{d}$$.
A high value of density indicates that the corresponding 3D position is occupied, i.e., there is something at that location, so if nothing blocks the view from the camera, we will see that thing.

You can turn this representation into an image taken by an arbitrarily-positioned camera.
An image consists of pixels.
We can approximate the colour of a pixel as the colour of the light travelling towards the camera on a straight line that connects the pixel to the optical center of the camera.
This means that the colour of the pixel will (approximately) be the same as the colour of the surface intersected by the ray closest to the camera.
So to render an image, all you need to do is to find the closest surface along every ray corresponding to an image.

In the paper, the authors use volumetric rendering, where they simultaneously evaluate the function at a number (K, say) of points along a ray.
This is perhaps computationally wasteful, as you could get away with fewer evaluations by starting at the camera and following the ray until you reach a point of high density.
But it is parallelizable, so it is uses modern accelerators quite efficiently.

The ray can be quite long (L meters, say), so this is quite imprecise as you cannot really represent anything smaller than around $$\frac{L}{K}$$ meters.
The trick is to take the density values at those K points and approximate density along the ray by---yes, you guessed correctly---a piecewise-constant PDF.
If we now resample N points from that PDF, and if we assume that the objects sits in only one of the K intervals, the effective resolution is now $$\frac{L}{KN}$$ meters!

There is a detail: for this to work, the first evaluation has to be done with a coarser version of $$f$$ that learns a lower-resolution representation (where it learns average density in each interval, otherwise we would still miss small objects).

Now, this detail is a bit ugly, since that coarer representation has to be trained, too, and so the authors train two separate scene representation, one coarse and one fine, where each is trained by reconstructing images.

This can be improved twofold.
First, if it was possible to differentiate through the samples taken from the piecewise-constant density, we could train it simply by back-propagating through those samples.
This way, the coarse representation would be jointly optimized with the fine representation---directly for high-quality reconstruction.
This requires taking the derivative of the sampled value with respect to the density of the corresponding bin.

Second, since we are learning a coarse representation of density, we might as well learn a distribution of densities along the ray.
This would be not consistent between viewpoints (while the current scheme is), but would lead to sampling in regions of higher density, effectively increasing the resolution of this representation.
The inconsistencies would be fixed by resampling from the piecewise-constant PDF in the second stage.
In this case we would also like to learn where bin boundaries should be. This requires differentiability with respect to bin boundaries.

Since we do not know where the closest object to the camera is, we evaluate $$f$$ along the whole ray.
This is not very precise, so we do it twice: first, we sample points uniformly at random along the ray. This allows computing density with a coarses version of $$f_c$$, a coarser version of $$f$$.
Then, the authors approximate the density along the ray as---yes, you guessed correct---a piecewise-constant PDF and resample points from that PDF.
This second step allows taking more samples from the regions of high density, so that $$f$$ is more likely to be evaluated in occupied parts of the scene.

# Piecewise-constant PDFs for Regression
# Piecewise-constant PDFs for Image segmentation

A good example is their application in Neural Radiance Fields (NeRF).
NeRF learns a representation of a 3D scene as parameters of a neural net.
To get a value of a pixel, we shoot out a ray from a camera passing through that pixels, and we integrate colours along that ray.
Now, colours are important only if something occupies the space the ray passes through, which is quantified by density.
We first evaluate densities at samples taken uniformly at random along the ray.
We then approximate the density PDF as a piecewise-constant PDF and resample.
This is a form of importance sampling, which allows taking more samples in regions that are more likely to be occupied by an object.

This is great, but this mechanism requires training two separate scene representations: one coarse, to provide density for the piecewise-constant resampling, and one fine, to compute the final values of the pixel colours.

The way this is handled in NeRF is that the two separate neural networks are used to reconstruct two versions of an image, where one is necessarily lower quality than the other one.
This is partly because sampling from the piecewise-constant PDF is implemented in a non-differentiable manner.
If we could differentiate with respect to the the density, we could train the coarse representation simply by back-propagation through sampling from the piecewise-constant PDF.


Further, since the initial distribution we use for sampling along the ray is uniform, we need to take a relatively large number of samples to cover the space.
This could be improved if we learned a view-dependent distribution of densities along a ray.
Since it is view-dependent, it would not be consistent between different views.
This is ok, though, as we are only learning a proposal for important sampling.
Once we evaluate the density at obtained points, we still have to resample from the piecewise-constant.
This second improvement requires differentiability with respect to bin boundaries as well.




Piecewise constant PDFs are just (normalized) histograms.
I haven't really seen many applications in ML for such PDFs, but this changed with [Neural Radiance Fields (NeRF)]().
In NeRF, we evaluate densities on a bunch of points within an interval, and then approximate the density on that whole interval but bars centered on these points, where the height of each bar is equal to the normalized density.
That forms a piecewise constant PDF, from which we can then sample.
This is better than uniform sampling along whole interval, because it provides samples where they are most needed: in regions of high density.

But we can use a piecewise constant PDF to reparametrize sampling from a categorical distribution in cases where the categories are quantized real values.
This happens often in RL [ref]() but has been also done in regression tasks [ref nerf-with-learned-depth]().

To do that, though, we need to sample in such a way that we can backpropagate through the same.
More formally, we need to be able to evaluate the derivate of the sampled value with respect to the parameters of the distribution: the bin boundaries and the density values.

While this is not difficult, the most-commonly used sampling method from a piecewise-constant PDF is not differentiable with respect to either the bin boundaries or both the boundaries and the densities.

In this blog post, we will look at the naive non-differentiable and the reparametrized differentiable sampling for a piecewise-constant PDF.
The latter uses the inverse transform idea.
Since continuous sampling from a quantized space is useful in higher-dimensional spaces (so e.g. 2d or 3d instead of just a 1d interval), we will also look at how to generalize inverse transform to these higher-dimensional settings.

# Piecewise Constant PDF
Consider a normalized histogram on the [0, 1] interval with k bins.
Such histogram is defined by k+1 bin boundaries $$x_i$$ such that $$x_0 = 0$$, $$x_{i} < x_{i+1}$$ and $x_k = 1$$.
We also need k bin heights, or the piecewise constant densities $$d_i > 0$$ with $$\sum_{i=1}^k d_i(x_i - x_{i-1}) = 1$$.

Importantly, the corresponding cumulative distribution function (CDF) is piecewise linear, with pieces delineated by the same bin boundaries, and slopes defined by the densities $$d_i$$.

Now, the naive way to sample from such a pdf is to 1) sample the bin $$j$$ from a categorical, where the probability of every category is defined as $$p_i == d_i(x_i - x_{i-1})$$, and then sampling uniformly at random from the corresponding interval $$[x_{j-1}, x_j]$$.
This is very simple to implement, but is not differentiable with respect to densities nor the bin boundaries.


#### Acknowledgements
