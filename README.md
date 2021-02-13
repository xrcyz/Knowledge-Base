# Knowledge Base

**What is a hash in shader programming?**

TBA.

**What are some fun project ideas?**

- dungeon generator
- asteroid field generator
- minecraft
- neural networks
- reaction-diffusion models
- diffusion-limited aggregation
- flocking simulation
- cellular automata
- genetic algorithms 
- simulated annealing 
- physarum transport networks 
- auxin growth models
- ant colony algorithm 
- lambda flow reactor

See also [Morphogenic resources](https://github.com/jasonwebb/morphogenesis-resources)

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

**What is supervised and unsupervised machine learning?**

TBA.

**What is a support vector machine?**

TBA.

**What is a one-hot vector?**

This: {0, 0, 1, 0, ..., 0}. The array represents a classification, each index represents a class, and a one-hot vector says this definitely 100% belongs to class <index>. 

**What is a neural network?**

A neural networks successively transforms points from one coordinate system into another coordinate system, until the various categories we are looking for (dogs, traffic lights, action prompts) form clusters. Then we can classify a new data point by how closely it maps to each cluster. 

[Video](https://www.youtube.com/watch?v=UOvPeC8WOt8).

[Essay](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

**What is back-propagation?"**

Back-propogation is like a reverse neural network. We start with an error array (difference between output nodes and target values). Layer by layer we construct the next error array as a weighted sum of the previous error array. The backprop weights are the normalised forward-prop weights (they sum to 1.0). 

We then adjust the weights feeding into each node as a function of the node's error value. 

[math](http://colah.github.io/posts/2015-08-Backprop/).

**What is an auto-encoder?**

We start with a high-dimensional vector (such as a pixel array), map it to a low-dimensional vector (the inputs to a decoder function), then map that back to a high-dimensional vector (decoded pixel array). If you can train the output layer to match the input layer, then the middle layer must represent a lookup key of some kind.

Conspiracy theory: "vanilla" neural nets are also auto-encoders. Say we want to classify points (x,y) as being above or below f(x). We take the points (x,y) and transform them until we can draw a straight line between points above and points below. Are we not *de facto* encoding f(x) into a line function? 

**What is a Variational Auto-Encoder?**

A variational auto-encoder maps the input to an array of of tuples <mean, standard deviation>. The inputs to the decoder function are chosen from the probability distribution of possible encodings (an n-dimensional bell curve). This smears the data a bit so that it fills in gaps in the latent space, allegedly giving us a smoother transition from one feature to another (such as bearded to non-bearded). In other words, "values which are nearby to each other in latent space should have similar reconstructions". This allows us to interpolate between latent vectors in a way that might not work with a direct auto-encoder. 

Apparently if you force the network to learn a wider distribution, it encourages the encoder to decorrelate the encoded vector elements, presumably so that pulling one lever doesn't mess with two dials. 

Finally, assuming a continuous latent space, you can then sample from this space as a generative model. 

* [Variational auto-encoders](https://www.jeremyjordan.me/variational-autoencoders)
* [Intro to adversarial auto-encoders](https://rubikscode.net/2019/01/14/introduction-to-adversarial-autoencoders/)

**What is few-shot, one-shot, zero-shot learning?**

Learn a new category from just a few, one, or zero examples. The network learns a superclass ("3D object", or "human face"). It is then trained to extract the superclass from the input (a 3D model teapot from a photo). Now given one photo of a teapot, we can extrapolate with some level of confidence to another picture of a teapot, since they should both map to the same latent teapot representation. 

Given that you can class objects different ways (shape, colour, texture, ...), I presume that the superclass is implicit in the training regime. 

[This paper](https://arxiv.org/abs/1606.05579) applies it to recognising procedurally generated 2D shapes. If the latent representation successfully encodes the factors of the procedural function, then classes of objects should all map to similar regions in the latent space. 

**What is t-SNE?**

Given a high-dimensional data set, draw a force-directed graph in low-dimensional space that preserves point-to-point distances.

**What is a kernel in machine learning?**

A kernel is a matrix of weights which are multiplied with the input to extract relevant features. 

In fancy language: If a data set is not linearly separable in N dimensional space, we can project it to M dimensional space and draw a hyperplane through it there. The kernel function "computes a dot product in M-dimensional space without explicitly visiting this space." - [source](https://stats.stackexchange.com/questions/152897/how-to-intuitively-explain-what-a-kernel-is).

**What is a convolutional layer?**

A convolutional layer is doing a fancy moving weighted average. Consider the N-period weighted average of a time series. Each element of the weighted-average performs a sumproduct of the previous N elements by some weighting. The convolution function takes this sumproduct, adds an offset (the bias) and feeds in into a [logistic function](https://en.wikipedia.org/wiki/Logistic_function) to return a number between 0 and 1. It's equivalent to a function that returns true or false when the moving weighted average is higher or lower than some threshold. That's it. The weights are modified such that the weighted average returns higher or lower on certain distributions of values, which effectively turns it into a feature detection function. 

The set of weights (shared between all points in the logistic-moving-sumproduct function) is referred to as the 'kernel'. 

In the 2D case, the "weighted moving average" is now a moving matrix dot product; and instead of mapping a series to another series, we map an array to another array. 

**What is a feature map?**

The output of a convolutional layer. 

**What is a pooling layer?**

A pooling layer takes the max/average/sum of a block of values. The idea is that reduces the array size while preserving important features. Pooling seems to be applied to the output of the convolutional layers (the "feature maps"). 

**What is an LSTM layer?**

So far [this](https://tedfmyers.com/2019/03/09/machine-learning-long-short-term-memory-cells/) seems to be the least confusing take. 
- An LTSM has state, an array of some values that is passed forward. The state is zero at time zero. 
- At each time step, the LTSM reads in the new input array + the previous output array.
- The combined inputs feed into four different neural networks: forget_gate, ignore_gate, memory_in, and output_gate
- The new cell state = (previous_state * forget_gate) + (memory_in * ignore_gate). 
- Final output = tanh(new_state) * output_gate

The gates are all sigmoid, the memory in/out are tanh. I'm unclear if a gate has multiple layers.

**What is a Generative Adversarial Network?**

TBA.

**What is a Compositional Pattern Producing Network?**

TBA.

**What is a Gaussian Mixture Model?**

TBA.

**What is a Transformer in machine learning?**

TBA.

**What is an attention mechanism in machine learning?**

TBA.

**What is a policy?**

A policy is the function that maps a state to an action. The network weights are the parameters of the policy. 

**What is reinforcement learning?**

TBA.

**What is Q-learning?**

TBA.

**What is a Neural Turing Machine?**

[Answer](https://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html)

**What is a graph neural network?**

No idea.

**What is neural rendering?**

TBA.
