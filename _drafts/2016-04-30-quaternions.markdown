---
layout: post
title:  "Quaternions"
date:   2016-04-30 14:10:00 +0200
categories: maths
---
I'm coding augmented/virtual reality for Oculus Rift and one of the challanges I'm facing is estmating head orientation. Oculus comes with an IMU sensor that measures position and orientation with an easy access via API. As measurements are quite noisy, it pays to try and smooth them out. We're also using a camera on top of Oculus for SLAM and we'd like to do sensor fusion. I'll write a series of posts about imlementing (Extended) Kalman Filter. We'll have to start with quaternions, though.

Quaternion \\(q = (w, x, y, z)\\) is a hypercomplex number. Thea real part \\(Re(q) = w\\) and the imaginary part \\( Im(q) = (x, y, z) \\). For us it's important that it's a 4 element vector. It's useful for representing rotations in 3D. If we want to rotate a point \\(p\\) by a quaternion \\(q\\) we do it as:

  $$\tilde{p}_{r} = q\tilde{p}q^{-1}$$

Where \\( \tilde{p} = (0, p) \\) is a quaternion with zero real part and the point \\(p\\) as the imaginary part and \\(q^{-1} = (w, -Im(q)) \\) is an inverse (or conjugate)  quaternion. Quaternions should be also normalized, that is \\( \|\|q\|\|^2 = w^2 + x^2 + y^2 + z^2 = 1\\). Where does it come from? If we have an angle \\( \theta \\) and an axis \\( \mathbb{n} \\) we get a corresponding quaternion as 

$$ q = sin \left(\frac{\theta}{2} \right) + cos \left( \frac{\theta}{2} \mathbb{n} \right) $$

If you take the norm of this expression, you'll see it's equal to 1. 

Quaternion Derivatives.

For Kalman filtering we'll need quaternion time derivative and the jacobian matrix. We can write quaternions in exponent form, since they're complex numbers.  Since physicial units do not like exponentiation, we'll introduce a constant \\(k = \frac{1}{s} \\).

$$ q(t) = e^{\frac{\theta(t)}{2}\mathbb{n}} k $$


$$ \dot{q(t)} = \frac{dq(t)}{dt} = \frac{\theta(t)}{2} \mathbb{n} k e^{\frac{\theta(t)}{2}\mathbb{n}k} = \frac{\theta(t)}{2} k \mathbb{n} q(t) = \frac{1}{2} \tilde{\omega}(t) q(t)$$
 

Where \\( \tilde{\omega}(t) = (0, \omega(t)) \\) is the angular velocity in quaternion form. 



 

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: http://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/
