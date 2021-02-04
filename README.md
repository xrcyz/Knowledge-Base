# Knowledge Base

**What is a hash in shader programming?**

TBA.

**What are reaction-diffusion models?**

TBA.

**What is the difference between dot product and element-wise multiplication?**

From [here](https://stackoverflow.com/a/48201957):
```
Dot product
|A, B| . |E, F| = |A*E+B*G, A*F+B*H|
|C, D|   |G, H|   |C*E+D*G, C*F+D*H|

Multiply aka Hamard Product
|A, B| ⊙ |E, F| = |A*E, B*F|
|C, D|    |G, H|   |C*G, D*H|

```

**What is a Fourier Transform?**

We want to know what frequencies, if any, are present in a function. To do that, transform the function to polar coordinates (each data point becomes a vector with magnitude f(y) and a rotation). The rotation step is a multiple of the frequency we are inspecting. We sum the vectors to get a "centroid" of the function in polar coordinates. The further this centroid is from origin, the greater the effect of this frequency. If we repeat this process for all frequencies from zero to N, and plot the result (the magnitude of each centroid), we get the fourier transform charts that show up in google images. 

What about images? An image is a function f(x,y) on a plane that describes intensity at each point. As the 1D case, we can express this landscape as a sum of interfering waves on a surface. To extract the waves, we can rotate the image by the wave heading, wrap the image around a cylinder, then each cross-section of the cylinder maps to a one-dimensional fourier transform. We want to know which "frequency" (x,y) matches a wave, so we can do a second fourier transform where rotation is the heading and vector magnitude is the current step along the cylinder. This should return an intensity at each (x,y) which peaks if it matches a wave.

**What is a support vector machine?**

TBA.

**What is a kernel in machine learning?**

A kernel is a matrix of weights which are multiplied with the input to extract relevant features [(thank you Prakhar Ganesh)](https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37). 

In fancy language: 
> If a data set is not linearly separable in N dimensional space, we can project it to M dimensional space and draw a hyperplane through it there. The kernel function "computes a dot product in M-dimensional space without explicitly visiting this space." - [source](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is).

**What is a convolutional layer?**

A convolutional neural network is doing a fancy moving weighted average. Consider the N-period weighted average of a time series. Each element of the weighted-average performs a sumproduct of the previous N elements by some weighting (the connection weights). The convolution function takes this sumproduct, adds an offset (the bias) and feeds in into a [logistic function](https://en.wikipedia.org/wiki/Logistic_function) to return a number between 0 and 1. It's equivalent to a function that returns true or false when the moving weighted average is higher or lower than some threshold. That's it. The weights are modified such that the weighted average returns higher or lower on certain distributions of values, which effectively turns it into a feature detection function. 

Images extend this to the 2D case, with a small change. The "weighted moving average" is now a moving dot product; and instead of mapping a series to another series, we map an array to another array. 

**What is a pooling layer?**

A pooling layer takes the max/average/sum of a block of values. The idea is that reduces the array size while preserving important features. 

**What is a Variational Auto-Encoder?**

TBA.

**What is a Compositional Pattern Producing Network?**

TBA.

**What is a Gaussian Mixture Model?**

TBA.

**What is an LTSM?**

TBA.

**What is a Transformer in machine learning?**

TBA.

**What is an attention mechanism in machine learning?**

**What is a Neural Turing Machine?**

[Answer](https://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html)

**What is a graph neural network?**

No idea.
