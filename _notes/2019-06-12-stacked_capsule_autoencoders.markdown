---
layout: draft
title:  "Stacked Capsule Autoencoders"
date:   2019-06-12 15:15:0 +0000
categories: ML
---

<!--
1. skim the paper and make notes on what's missing or what's confusing
2. try to make a simple story and focus on the notes from above

  1) object-part relationships are viewpoint invariant
  2) parts are simpler than whole objects (less variety), and therefore should be easier to detect than whole objects
  3) once we know about parts, we can use this knowledge to reason about objects
    - this requires knowing canonical object-part relationships
  4) here we assume that both object-part relationships and parts can be learned end-to-end

3. sketch sections
4. write!
-->

Objects play a central role in computer vision and, increasingly, machine learning research.
With many applications depending on object detection in images and videos, the demand for accurate and efficient algorithms has always been high.
More generally, knowing about objects is essential for understanding and interacting with our environments.
Usually, object detection is posed as a supervised learning problem, and modern approaches typically involve training a CNN to predict the probability of whether an object exists at a given image location (and maybe the corresponding class), see e.g. [here](https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4).

<figure id='SCA_overview'>
  <img style="display: box; margin: auto" src="{{site.url}}/resources/scae/blocks_v4.svg" alt="SCAE"/>
  <figcaption align='center'>
  <b>Fig 1:</b> The Stacked Capsule Autoencoder (SCAE) is composed of a Part Capsule Autoencoder (PCAE) followed by an Object Capsule Autoencoder (OCAE). It can decompose an image into its parts and group parts into objects.
  </figcaption>
</figure>

While modern methods can achieve superhuman performance in object detection, they need to consume staggering amounts of data to do so. This is in stark contrast to kids (or mammals, for that matter), who learn to recognise and localise objects with very little guidance from their seniors.
It is difficult to say what exactly makes mammals so good at learning, but I can imagine that _self-supervision_ and _inductive biases_ present in their sophisticated computing hardware (brains) both play a huge role.
These intuitions have led us to develop an [unsupervised version of capsule networks](https://www.google.com/search?q=stacked+capsule+autoencoders&rlz=1C5CHFA_enGB715GB715&oq=stacked+caps&aqs=chrome.0.69i59j69i57j69i60l2j69i61j35i39.4385j1j7&sourceid=chrome&ie=UTF-8), see [Figure 1](#SCA_overview) for an overview, whose inductive biases give rise to object-centric latent representations, which are learned simply by reconstructing input images.
In the reminder of this blog, I will try to explain what those inductive biases are, how they are implemented and what kind of things are possible with the new capsule architecture.
I will also try to explain how this new version differs from previous versions of [capsule networks](https://openreview.net/forum?id=HJWLfGWRb&noteId=rk5MadsMf).


# Why do we care about equivariances?
I think it is fair to say that deep learning would not be so popular if not for CNNs and the [2012 AlexNet paper](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf).
CNNs learn faster than non-convolutional image models due to (1) local connectivity and (2) parameter sharing across spatial locations.
The former restricts what can be learnt, but is sufficient to correlate nearby pixels, which turns out to be important for images.
The latter makes learning easier, since parameter updates benefit from more signal.
It also results in _translation equivariance_, which means that, when the input to a CNN is shifted, the output is shifted by an equal amount, while remaining unchanged otherwise.
Being equivariant helps with learning and generalisation---for example a model does not have to see the object placed at every possible spatial location in order to learn how to classify it.
For this reason, it would be great to have neural nets that are equivariant to other affine degrees of freedom like rotation, scale and shear, but this is not very easy to achieve, see e.g. [group equivariant conv nets](https://arxiv.org/abs/1602.07576).

Equivariance to different transformations can be learned approximately, but it requires vast data augmentation.
Augmenting data with random crops or shifts helps even with training translation-equivariant CNNs---these are typically followed by fully-connected layers, which have to learn to handle different positions.
Augmenting data with other affine transformations is not easy, as it would require access to full three-dimensional scene models.
Even if scene models were available, we would need to augment data with combinations of different transformations, which would result in an absolutely enormous dataset.
The problem is exacerbated by the fact that objects are often composed of parts, and it would be best to capture all possible configurations of object parts.
[Spatial Transformer Networks](https://arxiv.org/abs/1506.02025) and [this followup work](https://arxiv.org/abs/1901.11399) provide one way of learning affine equivariances, but do not address the fact that objects can undergo local transformations.


# Capsules learn equivariant object representations
<!-- why capsules are a good idea?
- being equivariant to many things at the same time is difficult
- equivariance can be learnt to some extent, but this requires tremendous amounts of data, since the model has to see every objects from every possible viewpoint
- objects are often complicated and can be expressed as compositions of parts
- part-object relationships are not fixed, generally speaking, but there definitely is correlation between objects and parts; you can think of this as an objet defining some sort of possible configurations of its parts
- parts are by definition less complicated, so it should be easier to learn to discover them than to discover whole objects, which lowers the complexity of the learning problem
- this can be done by introducing an explicit inductive bias, which says that objects are composed of parts
- the goal is to learn equivariant representations for parts, and something that is close-to-invariant for object-part relationships

 -->
 <figure id='old_capsules'>
   <img style="display: box; margin: auto" src="{{site.url}}/resources/scae/old_capsules.svg" alt="Capsule Network"/>
   <figcaption align='center'>
   <b>Fig 2:</b> Capsule networks work by inferring parts & their poses from an image, and then using parts and poses to reason about objects.
   </figcaption>
 </figure>

Instead of building models that are globally equivariant to affine transformations, we can rely on the fact that complicated objects are composed of simpler parts.
By definition, parts exhibit less variety in their appearance and shape than full objects, and consequently, it should be easier to learn about them from raw pixels.
Objects can then be recognised from parts and their poses, given that we can learn typical part configurations for different objects, as in [Figure 2](#old_capsules).
The caveat here is that we still need to learn a part detector, and it needs to predict part poses (translation, rotation, scale), too.
This _should_ be much simpler than learning a full-blown object detector with similar capabilities, though.

Since poses of any entities present in a scene change with the location of an observer (or rather the chosen coordinate system), then a detector that can correctly identify pose of a part produces a viewpoint equivariant part representation.
Since object-part relationships do not depend on the particular vantage point, they are viewpoint invariant.
These two properties, taken together, result in viewpoint equivariant object representations.

The issue with the above is that the corresponding inference process, that is, using previously discovered parts to infer objects, is difficult, since every part can belong to at most one object.
Previous versions of capsules solved this by refining the assignment of parts to objects (also known as _routing_) iteratively. This proved to be inefficient in terms of both computation and memory, and made it impossible to scale to bigger images.
See e.g. [here](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) for an overview of previous capsule networks.

# Can an arbitrary neural net learn capsule-like representation?
Original capsules are a type of a feed-forward neural network with a specific structure, and are trained for classification.
Incidentally, we know that classification corresponds to inference, which is the inverse process of generation, and as such is more difficult.
To see this, think about Bayes' rule: this is why posterior distributions are often much more complicated than the prior or likelihood terms.

Instead, we can use the principles introduced by capsules to build a generative model (decoder) and a corresponding inference network (encoder).
Generation is simpler, since we can have any object generate arbitrarily many parts, and we do not have to deal with constraints encountered in inference.
Complicated inference can then be left to your favourite neural network, which is going to learn appropriate latent representations.
Since the decoder uses capsules machinery, it is viewpoint equivariant by design.
It follows that the encoder has to learn representations that are also viewpoint equivariant, at least approximately.

A potential disadvantage is that, even though the latent code might be viewpoint equivariant, in the sense that it explicitly encodes object coordinates, the encoder itself need not be viewpoint equivariant. This means that, if the model sees an object from a very different point of view than it is used to, it may fail to recognise the object.
Interestingly, this seems to be in line with human perception, as noted recently by [Geoff Hinton in his Turing Award lecture](https://youtu.be/VsnQf7exv5I?t=2167), where he uses a thought experiment to illustrate this.
If you are interested, you can watch the below video for about 2.5 minutes.

<div align='center' style='display: box;'>
  <iframe width="560" height="315" src="https://www.youtube.com/embed/VsnQf7exv5I" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

If you did not watch the video, here is a simplified version of it, c.f. [Figure 3](#diamond_square): imagine a square, and tilt it by 45 degrees, look away, and look at it again. Can you see a square? Or does the shape resemble a diamond (rhombus)?
Humans tend to impose coordinate frames on the objects they see, and the coordinate frame is one of the features that let us recognise the objects.
If the coordinate frame is very different from the usual one, we may have problems recognizing the correct shape.

<figure id='diamond_square'>
  <img style="max-width: 450px; display: box; margin: auto" src="{{site.url}}/resources/scae/square_diamond.svg" alt="Is it a square, or is it a rhombus?"/>
  <figcaption align='center'>
  <b>Fig 3:</b> Is it a square, or is it a diamond (rhombus)?
  </figcaption>
</figure>



# Why bother?
talk about results


I hope I have managed to convince you that learning capsule-like representation is possible, and that it might be a good idea.
This is the end of high-level intuitions, and we now proceed to some technical descriptions, albeit also high-level ones.
This might be a good place to stop reading if you are not into that sort of thing.

In the following, I am going to describe the decoding stack of our model, and you can think of it as a (determinisitc) generative model.
Next, we will follow with a description of an inference network, which provides the capsule-like representations.

# How can we turn object prototypes into an image?
Let us start by defining what a _capsule_ is.

We define a _capsule_ as a specialised part of a model that describes an abstract entity, e.g. a part or an object.
In the following, we will have _object capsules_, which recognise objects from parts, and _part capsules_, which extract parts and poses from an input image.
Even though capsules are present only in inference, or the encoding stage, it is useful to introduce them now.
Let _capsule activation_ be a group of variables output by a single capsule.
To describe an object, we would like to know (1) wether it exists, (2) how it looks like and (3) where it is located.
This is very similar to [Attend, Infer, Repeat (AIR)](https://github.com/akosiorek/attend_infer_repeat), also described in [my previous blog post](http://akosiorek.github.io/ml/2017/09/03/implementing-air.html), as well as [SQAIR](https://github.com/akosiorek/sqair), which extends AIR to videos and allows for unsupervised object detection and tracking.
Therefore, for an object capsule $$k$$, its activations consists of (1) a presence probability $$a_k$$, (2) a feature vector $$\mathbf{c_k}$$, and (3) a $$3\times 3$$ pose matrix $$OV_k$$, which represents the geometrical relationship between the object and the viewer (or some central coordinate system).
Similar features can be useful to describe parts, but we will simplify the setup slightly and assume that parts have a fixed appearance and only their pose can vary.
Therefore, for the $$m^\mathrm{th}$$ part capsule, its activations consist of the probability $$d_m$$ that the part exists, and a $$6$$-dimensional pose $$\mathbf{x}_m$$, which represents the position and orientation of the given part in the image.
As indicated by notation, there are several object capsules and potentially many part capsules.

Since every object can have several parts, we need a mechanism to turn an object capsule activation into several part capsule activations.
To this end, for every object capsule, we learn a set of $$3\times 3$$ transformation matrices $$OP_{k,m}$$, representing the geometrical relationship between an object and its parts.
These matrices are encouraged to be constant, although we do allow a weak dependence on the object features $$\mathbf{c}_k$$ to account for small deformations.
Since any part can belong to only one object, we gather predictions from all object capsules corresponding to the same part capsule and arrange them into a mixture, and we explain every part by independent Gaussian mixtures.

<figure id='mnist_strokes'>
  <div align='center' style="max-width: 800px; display: box; float: margin: auto;">
    <img style="width: 200px; padding: 5px;" src="{{site.url}}/resources/scae/mnist_strokes.png" alt="Object Capsules"/>
    <!--  -->
    <img style="max-width: 400px; padding:5px; filter:gray; -webkit-filter: grayscale(1); -webkit-filter: grayscale(100%);" src="{{site.url}}/resources/scae/transformed_mnist_strokes.png" alt="Object Capsules"/>
    <!--  -->
    <img style="max-width: 80px; padding:5px; filter:gray; -webkit-filter: grayscale(1); -webkit-filter: grayscale(100%);" src="{{site.url}}/resources/scae/mnist_rec.png" alt="Object Capsules"/>
  </div>

  <figcaption align='center'>
    <b>Fig 4:</b> <i>Left</i>: learned parts, or templates. <i>Center</i>: affine-transformed parts. <i>Right</i>: MNIST digits assembled from the transformed parts.
  </figcaption>
</figure>

Having generated part poses, we can take parts, apply affine transformations parametrised by the corresponding poses as in [Spatial Transformer](https://arxiv.org/abs/1506.02025), and assemble the transformed parts into an image ([Figure 4](#mnist_strokes)).
But wait---we need to get the parts first!
Since we assumed that parts have fixed appearance, we are going to learn a bank of fixed parts by gradient descent.
Each part is like an image, just smaller, and can be seen as a "template".
To give an example, a good template for MNIST digits would be a stroke, like in [the famous paper from Lake et. al.](http://www.sciencemag.org/content/350/6266/1332.short) or the left-hand side of Figure 4.

<!-- -->
# Where do we get capsule parameters from?
Above, we define a _generative process_ that can transform object and part capsule activations into images.
But to obtain capsule activations describing a particular image, we need to run some sort of inference.
In this case, we will just use neural networks to amortise inference.
In other words, neural nets will predict capsule activations directly from the image.
We will do this in two stages.
Firstly, given the image, we will have a neural net predict pose parameters and presence probabilities for every part from our learnable bank of parts.
Secondly, a separate neural net will look at the part parameters and will try to directly predict object capsule activations.
These two stages correspond to two stages of the generative process we outlined;
we can now pair each of the stages with the corresponding generative stage and arrive at two autoencoders.
The first one, Part Capsule Autoencoder (PCAE), detects parts and recombines them into an image.
The second one, Object Capsule Autoencoder (OCAE), organises parts into objects.
Below, we describe their architecture and some of our design choices.

## Inferring parts and poses

<figure>
  <img style="max-width: 650px; display: box; margin: auto" src="{{site.url}}/resources/scae/part_capsule_ae.svg" alt="Part Capsules"/>
  <figcaption align='center'>
  <b>Fig 5:</b> The Part Capsule Autoencoder (PCAE) detects parts and their poses from the image and reconstructs the image by directly assembling it from affine-transformed parts.
  </figcaption>
</figure>

The PCAE uses a CNN-based encoder, but with tweaks.
Firstly, notice that for $$M$$ parts, we need $$M \times (6 + 1)$$ predicted parameters.
That is, for every part we need $$6$$ parameters of an affine transformation $$\mathbf{x}_m$$ (we work in two dimensions) and a probability $$d_m$$ of the part being present; we could also predict some additional parameters; in fact we do, but we omit the details here for clarity---see the paper for details.
It turns out that using a fully-connected layer after a CNN does not work well here, see paper for details.
Instead, we project the outputs of the CNN to $$M \times (6 + 1 + 1)$$ feature maps using $$1\times 1$$ convolusions, where we added an extra feature map for every part capsule.
This extra feature map will serve as an attention mask: we normalise it spatially via a softmax, multiply with the remaining 7 feature maps, and sum each dimension independently across spatial locations.
This is similar to global-average pooling, but allows the model to focus on a specific location; we call it _attention-based pooling_.

We can then use part presence probabilities and part poses to select and affine-transform learned parts, and assemble them into an image.
Every transformed part is treated as a spatial Gaussian mixture component, and we train PCAE by maximising the log-likelihood under this mixture.

## Organizing parts into objects

<figure>
  <img style="max-width: 500px; display: box; margin: auto;" src="{{site.url}}/resources/scae/object_capsule_ae.svg" alt="Object Capsules"/>
  <figcaption align='center'>
  <b>Fig 6:</b> The Object Capsule Autoencoder (OCAE) tries to explain part poses as a sparse set of objects, where every present object predicts several parts. It automatically discovers structure in the data, whereby different object capsules specialise to different objects.
  </figcaption>
</figure>

Knowing what parts are there in an image, and where they are, might be useful, but in the end, we care about the objects that they belong to.
[Previous capsules](https://openreview.net/forum?id=HJWLfGWRb&noteId=rk5MadsMf) used an EM-based inference procedure to make parts vote for objects.
This way, each part could start by initially disagreeing and voting on different objects, but eventually the votes would converge to a set of only a few objects.
We can also see inference as compression, where a potentially large set of parts is explained by a potentially very sparse set of objects.
Therefore, we try to predict object capsule activations directly from the poses and presence probabilities of parts.
The EM-based inference tried to cluster part votes around objects.
We follow this intuition, and use the [Set Transformer](https://arxiv.org/abs/1810.00825) with $$K$$ outputs to encode part activations.
Set Transformer has been shown to work well for amortised-clustering-type problems, and it is permutation invariant.
Part capsule activations describe parts, not pixels, which can have arbitrary positions in the image, and in that sense have no ordering.
Therefore, set-input neural networks seem to be a better choice than MLPs---a hypothesis corroborated by an ablation study we have in the paper.

Each output of the Set Transformer is fed into a separate MLP, which then outputs all activations for the corresponding object capsule.
We also use a number of sparsity losses applied to the object presence probabilities; these are necessary to make object capsules specialise to different types of objects, please see the paper for details.
The OCAE is trained by maximising the likelihood of part capsule activations under a Gaussian mixture of predictions from object capsules, subject to sparsity constraints.

# Summary
<figure>
  <img style="display: box; margin: auto" src="{{site.url}}/resources/scae/blocks_v4.svg" alt="SCAE"/>
  <figcaption align='center'>
  <b>Fig 7:</b> The Stacked Capsule Autoencoder (SCAE) is composed of a PCAE followed by an OCAE. It can decompose image into its parts and group parts into objects.
  </figcaption>
</figure>

In summary, a Stacked Capsule Autoencoder is composed of:
- the PCAE encoder: a CNN with attention-based pooling,
- the OCAE encoder: a Set Transformer,
- the OCAE decoder:
  - $$K$$ MLPs, one for every object capsule, which predict capsule parameters from Set Transformer's outputs,
  - $$K \times M$$ constant $$3 \times 3$$ matrices representing constant object-part relationships,
- and the PCAE decoder, which is just $$M$$ constant part templates, one for each part capsule.

SCAE defines a new method for representation learning, where an arbitrary encoder learns viewpoint-equivariant representations by inferring parts and their poses and groups them into objects.
This post provides a motivation as well as high-level intuitions behind this idea, and an overview of the method.
The major drawback of the method, as of now, is that the part decoder uses fixed templates, which are insufficient to model complicated real-world images.
This is also an exciting avenue for future work, together with deeper hierarchies of capsules and extending capsule decoders to three-dimensional geometry.
If you are interested in the details, I would encourage to read the original paper: [A. R. Kosiorek, S. Sabour, Y.W. Teh and G. E. Hinton, "Stacked Capsule Autoencoders", arXiv 2019](https://arxiv.org/abs/1906.06818).

# Further reading:
- [a series of blog](https://medium.com/ai³-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b) post explaining previous capsule networks,
- [the original capsule net paper](http://papers.nips.cc/paper/6975-dynamic-routing-between-capsules) and [the version with EM routing](https://openreview.net/forum?id=HJWLfGWRb),
- [a recent CVPR tutorial on capsules](https://youtu.be/zRg3IuxaJ6I) and [slides](https://www.crcv.ucf.edu/cvpr2019-tutorial/slides/intro_sara.pptx) by Sara Sabour,


#### Acknowledgements
This work was done during an internship at Google Brain in Toronto in Geoff Hinton's team. I would like to thank my collaborators:
 [Sara Sabour](https://www.linkedin.com/in/sara-sabour-63019132/?originalSubdomain=ca),
 [Yee Whye Teh](https://www.stats.ox.ac.uk/~teh/) and
 [Geoff Hinton](http://www.cs.toronto.edu/~hinton/).
 I also thank [Sandy H. Huang](https://people.eecs.berkeley.edu/~shhuang/) who helped with making figures and editing the paper, as well as [Ali Eslami](http://arkitus.com/research/) and [Danijar Hafner](https://danijar.com/) for helpful discussions.
