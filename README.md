# Knowledge Base

**What are some fun project ideas?**

- minecraft
- dungeon generator
- asteroid field generator
- neural networks
- reaction-diffusion models
- diffusion-limited aggregation
- flocking simulation
- cellular automata
- genetic algorithms 
- simulated annealing 
- physarum transport networks 
- ant colony algorithm 
- auxin growth models
- leaf venation / wing venation / drainage patterns / urban layout synthesis
- ant nest morphogenesis
- Metropolis-Hastings algorithm
- quality diversity search / curiosity search / novelty search / open-ended exploration
- reinforcement learning

See also [Morphogenic resources](https://github.com/jasonwebb/morphogenesis-resources)

**Inspirational work**
- [Cogmind procedural map generation](https://www.gridsagegames.com/blog/2014/06/procedural-map-generation/)
- [antibiotic gradient resistance](https://www.youtube.com/watch?v=plVk4NVIUh8)
- [evolution of evolvability](https://www.youtube.com/watch?v=i-qe-2PLkIc)
- [ant colony algorithm](https://www.youtube.com/watch?v=yZ1rSASM2Rg)
- [ant nest morphogenesis](https://www.researchgate.net/publication/260546781_A_computational_model_of_ant_nest_morphogenesis)
- [alien coral morphogenesis](https://www.joelsimon.net/corals.html)
- neural cellular automata [paper](https://distill.pub/2020/growing-ca/) and [implementation](https://znah.net/hexells)
- [neural networks failing to learn Game of Life](https://arxiv.org/abs/2009.01398)
- [neural network quine](https://arxiv.org/abs/1803.05859)
- [graph grammar aggregation](https://www.youtube.com/channel/UCAEB6v6pULTAbKl9aM_EDZw)
- [automated scientist](https://advances.sciencemag.org/content/6/5/eaay4237)
- [MAP-Elites algorithm](https://arxiv.org/abs/1504.04909)
- [artificial language](https://www.joelsimon.net/dimensions-of-dialogue.html)
- curiosity algorithm [paper](https://advances.sciencemag.org/content/6/5/eaay4237) and [code](https://github.com/croningp/dropfactory_exploration)
- [information-limited pathfinding](https://www.youtube.com/watch?v=qXZt-B7iUyw)
- [animated evolutionary strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

**How do neural networks arrive at an answer?**

Neural networks are just chains of [boolean operators](https://en.wikipedia.org/wiki/Logistic_function) in a trench coat. 

***Conditional IF***

Here is a "neural network" to classify a dog as above or below age 5. Paste `y=\frac{1}{1+e^{-10*(x-5)}}` into [Desmos](https://www.desmos.com/calculator) to try it out.

```
let ageThreshold = 5; //this is the "bias"
let ageIsGreaterThanFive = 1 / (1 + exp(-10*(dog.age - ageThreshold))); //returns 0 for false; 1 for true; 0.5 for inconclusive
```

***Conditional AND***

Here is a "neural network" to classify a dog as (over five years) && (over one meter tall). Paste `z=\frac{1}{1+e^{10 * (1.5 - x - y)}}` into [Geogebra](https://www.geogebra.org/3d) to try it out. 

```
//second layer
let ageThreshold = 5;
let ageIsGreaterThanFive = 1 / (1 + exp(-10*(dog.age - ageThreshold))); 

let heightThreshold = 1;
let heightIsGreaterThanOne = 1 / (1 + exp(-10*(dog.height - heightThreshold))); 

//output layer
let [x, y] = [ageIsGreaterThanFive, heightIsGreaterThanOne]; //a vector in the unit square
let logicalAnd = 1 / (1 + exp(10 * (1.5 - x - y))); 

```

In the above, the line `(y = 1.5 - x)` is used to test if a point is in the top right of the unit square. The logistic function converts the output to a `[0..1]` range, while the multiplier `10` is used to sharpen the transition slope. If this were diagrammed as a neural net, the second layer would have two neurons `[x, y]`, a bias `[1]`, and weights `[-10, -10, 15]` connecting to the output neuron. 

***Conditional XOR***

Suppose now we want to solve the XOR problem. Given `[x,y]` in the first layer, we can define four neurons `[A,B,C,D]` in the hidden layer
```
let A = (!x && y); //false true
let B = (x && y); //true true
let C = (x && !y); //true false
let D = (!x && !y); //false false

A = 1 / (1 + exp(-10 * (-0.5 - x + y))); //test for (0,1)
B = 1 / (1 + exp(-10 * (-1.5 + x + y))); //test for (1,1)
C = 1 / (1 + exp(-10 * (-0.5 + x - y))); //test for (1,0)
D = 1 / (1 + exp(-10 * ( 0.5 - x - y))); //test for (0,0)
```

In the output layer, we can naively assume that the conditions `[A,B,C,D]` are exclusive, so we can apply the logistic operator to the sum. (For extra credit, consider how the weighted sum might be used to derive a truth value for the input coordinate `[0.49, 0.75]`). 

```
let output = 1 / (1 + exp(-10*(A + C - B - D))); 
```

You can see the above XOR neural network configuration being derived [here](http://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=xor&regDataset=reg-plane&learningRate=0.1&regularizationRate=0&noise=0&networkShape=4,1&seed=0.21709&showTestData=true&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&batchSize_hide=false).

[![tensorflow playground](/images/tensorflow%20playground%20XOR.png)](http://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=xor&regDataset=reg-plane&learningRate=0.1&regularizationRate=0&noise=0&networkShape=4,1&seed=0.21709&showTestData=true&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&batchSize_hide=false)

***Conway's Game of Life***

The rules for [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) are:

```
if(self == 1 && neighbors == (2|3)) return 1
if(self == 0 && neighbors == 3) return 1
else return 0
```

Layer one of the neural network is going to calculate the basic booleans:

```
let (self == 1) = 1 / (1 + exp(-10*(self - 0.5))); 
let (neighbors > 1) = 1 / (1 + exp(-10*( neighbors.sum() - 1.5 ))); 
let (neighbors > 2) = 1 / (1 + exp(-10*( neighbors.sum() - 2.5 ))); 
let (neighbors > 3) = 1 / (1 + exp(-10*( neighbors.sum() - 3.5 ))); 
```

Layer two recombines the outputs into complex booleans:

```
let (self == 0 && neighbors == 3) = (!(self == 1) && (neighbors > 2) && !(neighbors > 3)) //need to test for (0,1,0)
```

In order to test for point (0,1,0), we need to define a plane that separates the vertex of a unit cube from the rest of the cube. Some messing around in [Geogebra](https://www.geogebra.org/3d) gives us `plane = -x + y - z - 0.5`.
```
let x = (self == 1);
let y = (neighbors > 2);
let z = (neighbors > 3);
let (self == 0 && neighbors == 3) = 1 / (1 + exp(-10*(-x + y - z - 0.5)));
```

And for the second neuron in the second layer: 

```
let (self == 1 && neighbors == (2|3)) = ((self == 1) && (neighbors > 1) && !(neighbors > 3)) //need to test for (1,1,0)

let x = (self == 1);
let y = (neighbors > 1);
let z = (neighbors > 3);

let (self == 1 && neighbors == (2|3)) = 1 / (1 + exp(-10*(x + y - z - 1.5)));
```

In the output layer, we want to return `(self == 0 && neighbors == 3) || (self == 1 && neighbors == (2|3))`, which we can get by testing if the sum of the values is greater than zero. 

```
let condition1 = (self == 0 && neighbors == 3);
let condition2 = (self == 1 && neighbors == (2|3));
let output = 1 / (1 + exp(-10*( condition1 + condition2 - 0.5 ))); 
```
You can test this implementation in the browser [here](https://openprocessing.org/sketch/1236584).

Interestingly, since every cell in the cellular automata shares the same update rule, then this is technically a "convolutional neural network". The four layers (9 inputs, 4 hidden, 2 hidden, 1 out) of our neural network form the "kernel", and a grid of kernels are applied to the input image to calculate the output image (the next state of the cellular automata). Our CNN has a "kernel size" of 3, a "step" of 1, and "pads" the input image by wrapping out-of-bound pixels to the opposite edge. 

Using this perspective, we can take a guess at how other convolutional neural networks perform their computations. 

[![Interactive Node-Link Visualisation of Convolutional Neural Networks](/images/aharley%20cnn%20visualisation.png)](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)

This [visualisation](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html) by [Adam Harley](http://www.cs.cmu.edu/~aharley/) shows the filters and layers used to classify handwritten digits. Convolution layer 1 contains six filters, in which every pixel represents the output of a kernel, and each kernel is performing a logical test on the corresponding region of the source image. Filters 1 and 6 detect horizontal edges, filters 3 and 4 detect diagonal edges, filter 2 detects vertical edges, and filter 5 remains a mystery for the ages. In the next layer, Downsampling layer 1, the "max pooling" operation performs a logical OR by preserving the max values. 

What is Convolution layer 2 doing? Most likely it is combining the earlier booleans to generate complex booleans, still localised to areas of the image. For example, a combined vertical plus horizontal test might be used in classifying 4 and 7. 


**What is the hype with machine learning?**

- A basic neural net is a classifier. It decides if data is above or below a classifying line. Useful but not super exciting. 
- With a little sauce you can run the neural net in reverse, starting with a classification working backwards to a sample point. This can be used for generating novel samples (faces, art, ...). 
- If you take a generator, and feed it back into a classifier,then you get a generative adversarial network. This trains the generator to match training data.
- If you take a classifier, and feed it into a generator, then you get an autoencoder. This trains the encoder to find efficient descriptors of the training data.
- If you you take a classifier, and train it in on actions in an environment, then you get Q learning. This matches states to actions for highest reward.
- If you take a series of classifiers, and use them to map an image to a bank of class arrays, then you get a convolutional neural net. The individual classifiers ("kernels") are trained to extract features from an image. 

**What is supervised and unsupervised machine learning?**

TBA.

**What is a support vector machine?**

TBA.

**What is a one-hot vector?**

This: {0, 0, 1, 0, 0}. A one-hot vector has all components set to zero except for one element. The vector represents the probability that an item is classified as element A, B, C, etc.

**What is a neural network?**

A neural networks transforms points from one coordinate system into another coordinate system, until the various categories we are looking for (dogs, traffic lights, action prompts) form clusters. Then we can classify a new data point by how closely it maps to each cluster. 

[Video](https://www.youtube.com/watch?v=UOvPeC8WOt8).

[Essay](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

**What is back-propagation?"**

Back-propogation is like a reverse neural network. We start with an error array (difference between output nodes and target values). Layer by layer we construct the next error array as a weighted sum of the previous error array. The backprop weights are the normalised forward-prop weights (they sum to 1.0). 

We then adjust the weights feeding into each node as a function of the node's error value. 

[math](http://colah.github.io/posts/2015-08-Backprop/).

**What is an auto-encoder?**

We start with a high-dimensional vector (such as a pixel array), map it to a low-dimensional vector (the inputs to a decoder function), then map that back to a high-dimensional vector (decoded pixel array). If you can train the output layer to match the input layer, then the middle layer must represent a lookup key of some kind.

Possibly redundant observation: "vanilla" neural nets are also "encoding" data, its just that we explicitly define the "latent representation". Suppose we make a neural net to map a 3D vector to a 1D length along a Hilbert curve. This can be seen as "one half" of an auto-encoder, the second half of which would decode the 1D vector back to its 3D vector. The point of interest is that in an auto-encoder, the network can choose whatever space-filling curve it likes.

Project idea: write an auto-encoder to map vectors in a 3D cube down to 1D and back to 3D. Then plot the curve and see what it looks like. 

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

**What is a Generative Adversarial Network?**

TBA.

**What is a Compositional Pattern Producing Network?**

TBA.

**What is a Gaussian Mixture Model?**

A guassian mixture model is [a superposition of overlapping bell curves](https://www.statisticshowto.com/gaussian-mixture-model/).

**What is a policy?**

The policy tells us which action to take in state `s`. It can be deterministic `π(s) = a` or stochastic `π(a, s) = probability of taking action`.

**What is reinforcement learning?**

> Reinforcement learning is defined not by characterizing learning methods, but by characterizing a learning problem. [[source]](http://incompleteideas.net/book/first/ebook/node7.html)

Reinforcement learning is generally characterised by having a set of states, actions, goals, and agents that have partial or full observability of the world state. 

Action selection is controlled by the `ϵ` "greedy" parameter, which controls the chance of random exploration vs. exploitation.

Some nice readable code [here](https://github.com/jcoreyes/reinforcement-learning).

**What is online / offline learning?**

A learning method is online if the agents incrementally update their parameters (of the policy, value function, or model) while they observe a stream of experience. 

**What are sparse rewards?**

Sparse rewards give little or no clues about intermediate steps to reach the goal. 

This [article](https://deepmind.com/blog/article/capture-the-flag-science) describes agents that learn to play a multiplayer game with only a binary reward (win or fail). 

**What is Q-learning?**

Q-learning begins with an agent in an environment. The agent has a function to map observations onto states, which are discrete nodes in a markov process. Upon visiting a state, the agent may observe what actions are available from that state. The state and the action form a key into a lookup table, which returns an estimated quality value Q for taking that action. The lookup table may be initialised with zero or random values; whenever an agent executes a state-action pair then the observed Q value is written back into the lookup table. The new Q value is calculated as a weighted sum of the old value (average the returns over rollouts), the observed reward, plus a fraction of the highest Q value in the next visited state. In this way, nodes with high Q values diffuse backwards through the action paths. The Q-table should converge on the values associated with the target policy.

This bears a lot in common with other agent sims like ant colony and physarum simulations, where agents leave trails for others to pick up.

[![gif](https://i.stack.imgur.com/Bn6MY.gif)](https://stackoverflow.com/questions/56777123/questions-about-deep-q-learning)

- The quality function `Q` gives the the expected return from state `s` and action `a` and following policy `π` thereafter. 
- The value function `V` gives the expected return when starting in state `s` and following policy `π` thereafter.
- R is the reward function.
- Gamma is the learning rate.

[Procedure:](hhttps://leonardoaraujosantos.gitbook.io/artificial-inteligence/artificial_intelligence/reinforcement_learning/qlearning_simple)
1. Initialize the Q matrix with zeroes
2. Select a random initial state
3. For each episode (set of actions that starts on the initial state and ends on the goal state)
  - Select an action for the current state (flip a coin for exploit or explore)
  - Execute the action and observe the next state
  - Update the Q value `Q(state, action) = R(state, action) + Gamma * Max(Q(nextState, allActions))`

The reason Q-learning is called 'off-policy' is that it estimates the total discounted future reward for state-action pairs assuming a greedy policy were followed, despite the fact that it's not following a greedy policy. 

> The exploration process stops when it reaches a goal state and collects the reward, which becomes that final transition's Q value. Now in a subsequent training episode, when the exploration process reaches that predecessor state, the backup process uses the above equality to update the current Q value of the predecessor state. Next time its predecessor is visited that state's Q value gets updated, and so on back down the line (Mitchell's [book](http://incompleteideas.net/book/the-book-2nd.html) describes a more efficient way of doing this by storing all the computations and replaying them later). Provided every state is visited infinitely often this process eventually computes the optimal Q. [[source]](https://datascience.stackexchange.com/questions/9832/what-is-the-q-function-and-what-is-the-v-function-in-reinforcement-learning)

> The key is that, in Q-learning, the agent does not know state transition probabilities or rewards. The agent only discovers that there is a reward for going from one state to another via a given action when it does so and receives a reward. Similarly, it only figures out what transitions are available from a given state by ending up in that state and looking at its options. If state transitions are stochastic, it learns the probability of transitioning between states by observing how frequently different transitions occur. [[source]](https://stackoverflow.com/questions/28937803/what-is-the-difference-between-q-learning-and-value-iteration)

> The most common policy scenarios with Q learning are that it will converge on (learn) the values associated with a given target policy, or that it has been used iteratively to learn the values of the greedy policy with respect to its own previous values. The latter choice - using Q learning to find an optimal policy, using [generalised policy iteration](http://incompleteideas.net/book/first/ebook/node46.html#:%7E:text=We%20use%20the%20term%20generalized,details%20of%20the%20two%20processes.&text=The%20evaluation%20and%20improvement%20processes%20in%20GPI%20can,as%20both%20competing%20and%20cooperating.) - is by far the most common use of it. [[source]](https://ai.stackexchange.com/questions/25971/what-is-a-learned-policy-in-q-learning)

There is a really nice walkthrough [here](https://blog.floydhub.com/an-introduction-to-q-learning-reinforcement-learning/).
Also [this](https://leonardoaraujosantos.gitbook.io/artificial-inteligence/artificial_intelligence/reinforcement_learning/qlearning_simple).

**What is Deep Q-Learning?**

A neural network is optimised to approximate the Q function. The generation of new episodes is interleaved with neural network training. 

**What is a replay buffer?**

A replay buffer can be used for offline learning, which may be cheaper or faster than running the training environment, and allows experiences to be sampled at a different order and frequency than found in the raw data (prioritising rare, surprising, frustrating, new, foundational, or otherwise useful experiences). 

Interestingly, this is based on experience [replay sequences](https://deepmind.com/blog/article/replay-in-biological-and-artificial-neural-networks) observed in sleeping rats. (There is an interesting term in this article, 'imagination replay'. It should be possible to train an agent that, given some state as input, can reconstruct a plausible previous state, such as the 3D model of an unbroken vase. It is also notable that you can dream about crashing a car, but it is extremely inadvisable to do it in real life.)

> First, each step of experience is potentially used in many weight updates, which allows for greater data efficiency. Second, learning directly from consecutive samples is inefficient, due to the strong correlations between the samples. Third, when learning on-policy the current parameters determine the next data sample that the parameters are trained on \[potential infinite loop]. [[source]](https://arxiv.org/pdf/1312.5602v1.pdf)

**What is policy gradient?**

Policy gradient methods learn the policy directly with a parameterized function (as opposed to learning Q values). 

**What is actor-critic framework?**

Actor-critic learns the value function and the policy function. 

> New trajectories are generated by actors. Asynchronously, model parameters are updated by learners, using a replay buffer that stores trajectories. [[source]](https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf)

Interesting note in the above paper, in addition to the binary win condition, is a reward for style points (eg. execute a certain strategy). I imagine this leads to better generalisation, since (across the population) the objective is to win by a variety of methods. It also trained predator agents ('Exploiters') to execute adversarial attacks against the learning population. 

**What is an LSTM layer and why?**

So far [this](https://tedfmyers.com/2019/03/09/machine-learning-long-short-term-memory-cells/) seems to be the least confusing take. 
- An LTSM has state, an array of some values that is passed forward. The state is zero at time zero. 
- At each time step, the LTSM reads in the new input array + the previous output array.
- The combined inputs feed into four different neural networks: forget_gate, ignore_gate, memory_in, and output_gate
- The new cell state = (previous_state * forget_gate) + (memory_in * ignore_gate). 
- Final output = tanh(new_state) * output_gate

The gates are all sigmoid, the memory in/out are tanh. I'm unclear if a gate has multiple layers.

**What is a Transformer in machine learning?**

TBA.

**What is an attention mechanism in machine learning?**

TBA.

**What is t-SNE?**

Given a high-dimensional data set, draw a force-directed graph in low-dimensional space that preserves point-to-point distances. 

**What is neural rendering?**

A neural network that maps a camera-object pair to a viewport pixel array. Given camera position and a 3D object, what is the expected camera render output?

**What is a Neural Turing Machine?**

[Answer](https://rylanschaeffer.github.io/content/research/neural_turing_machine/main.html)

**What is a graph neural network?**

TBA.

**What is transfer learning?**

TBA.

**What is a Fourier Transform?**

We want to know what frequencies, if any, are present in a function. To do that, transform the function to polar coordinates (each data point becomes a vector with magnitude f(y) and a rotation). The rotation step is a multiple of the frequency we are inspecting. We sum the vectors to get a "centroid" of the function in polar coordinates. The further this centroid is from origin, the greater the effect of this frequency. If we repeat this process for all frequencies from zero to N, and plot the result (the magnitude of each centroid), we get the fourier transform charts that show up in google images. 

What about images? What about "power spectrum"?

No idea.

**What is a kernel in shader programming?**

> "A kernel roughly corresponds to extracting the body of a loop to a function, so that the function can be executed in parallel in hardware."

Let's look at how it is used:
- ComputeShader has a method called `FindKernel` which, given a function name, returns an index.
- ComputeShader has a method called `Dispatch` which takes a `int kernelIndex` parameter. 
- ComputeShader has a method called `SetTexture` which sets buffers and textures _per-kernel_. 
- In the shader code, we define `#pragma kernel FunctionName` 
- In the shader code, we implement `void FunctionName(...)` 

Note that each instance of the function is not a kernel, because we get exactly one index integer. So we can say that the kernel is a template for some behaviour at every coordinate of an input object. 

The fact that we use `#pragma kernel` tells us that the kernel is an abstraction that is separate from the function implementation. Calling `Dispatch` on the kernel calls the function in parallel for all coordinates in the input object. Each invocation of a kernel within a batch is assumed independent.

(`Dispatch` dispatches a grid of work groups. `numthreads` defines the dimensions of each work group. We roughly want `work groups * num threads` equal to the [warp](https://www.google.com/search?q=nvidia+warp) size in hardware, to max out all processors on a batch of jobs. The hardware is a grid of factories, and if you don't bottleneck the capacity of each factory, then the spare capacity is wasted.)

**What is a hash in shader programming?**

It maps an input number (or vector) to a pseudo-random output number (or vector). 

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
