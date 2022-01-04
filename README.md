# Knowledge Base

**What are some fun project ideas?**

- minecraft
- dungeon generator [wip](https://openprocessing.org/sketch/1294029)
- asteroid field generator [wip](https://openprocessing.org/sketch/1342385)
- [neural networks](https://openprocessing.org/sketch/1248243)
- [reaction-diffusion models](https://openprocessing.org/sketch/1264005)
- diffusion-limited aggregation
- [flocking simulation](https://openprocessing.org/sketch/1281224)
- [cellular automata](https://openprocessing.org/sketch/1254639)
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

See also [Morphogenic resources](https://github.com/jasonwebb/morphogenesis-resources) and [Creative Coding Notes](https://github.com/cacheflowe/creative-coding-notes)

**Inspirational work**
- [Cogmind procedural map generation](https://www.gridsagegames.com/blog/2014/06/procedural-map-generation/)
- [antibiotic gradient resistance](https://www.youtube.com/watch?v=plVk4NVIUh8)
- [evolution of evolvability](https://www.youtube.com/watch?v=i-qe-2PLkIc)
- [ant colony algorithm](https://www.youtube.com/watch?v=yZ1rSASM2Rg)
- [ant nest morphogenesis](https://www.researchgate.net/publication/260546781_A_computational_model_of_ant_nest_morphogenesis)
- [alien coral morphogenesis](https://www.joelsimon.net/corals.html)
- [neural cellular automata](https://distill.pub/2020/growing-ca/) and [self organising textures](https://znah.net/hexells)
- [neural CA cartpole](https://avariengien.github.io/self-organized-control/)
- [neural networks failing to learn Game of Life](https://arxiv.org/abs/2009.01398)
- [neural network quine](https://arxiv.org/abs/1803.05859) and [neural replicators](https://direct.mit.edu/isal/proceedings/isal/58/102906)
- [graph grammar aggregation](https://www.youtube.com/channel/UCAEB6v6pULTAbKl9aM_EDZw)
- [automated scientist](https://advances.sciencemag.org/content/6/5/eaay4237)
- [MAP-Elites algorithm](https://arxiv.org/abs/1504.04909)
- [artificial language](https://www.joelsimon.net/dimensions-of-dialogue.html)
- curiosity algorithm [paper](https://advances.sciencemag.org/content/6/5/eaay4237) and [code](https://github.com/croningp/dropfactory_exploration)
- [information-limited pathfinding](https://www.youtube.com/watch?v=qXZt-B7iUyw)
- [animated evolutionary strategies](https://blog.otoro.net/2017/10/29/visual-evolution-strategies/)

**How do neural networks arrive at an answer?**
------

Neural networks are just chains of [boolean operators](https://en.wikipedia.org/wiki/Logistic_function) in a trench coat. 

***Conditional IF***

Here is a "neural network" to classify a dog as above or below age 5. Paste `y=\frac{1}{1+e^{-10*(x-5)}}` into [Desmos](https://www.desmos.com/calculator) to try it out.

```js
let ageThreshold = 5; //this is the "bias"
let ageIsGreaterThanFive = 1 / (1 + exp(-10*(dog.age - ageThreshold))); //returns 0 for false; 1 for true; 0.5 for inconclusive
```

***Logical AND***

Here is a "neural network" to classify a dog as (over five years) && (over one meter tall). Paste `z=\frac{1}{1+e^{10 * (1.5 - x - y)}}` into [Geogebra](https://www.geogebra.org/3d) to try it out. 

```js
//second layer
let ageThreshold = 5;
let ageIsGreaterThanFive = 1 / (1 + exp(-10*(dog.age - ageThreshold))); 

let heightThreshold = 1;
let heightIsGreaterThanOne = 1 / (1 + exp(-10*(dog.height - heightThreshold))); 

//output layer
let [x, y] = [ageIsGreaterThanFive, heightIsGreaterThanOne]; //a vector in the unit square
let logicalAnd = 1 / (1 + exp(10 * (1.5 - x - y))); 
```

In the above, the line `(y = 1.5 - x)` is used to test if a point `[x,y]` is in the top right of the unit square. The logistic function converts the output to a `[0..1]` range, and the multiplier `10` is used to sharpen the transition slope. If this were diagrammed as a neural net, the second layer would have two neurons `[x, y]`, a bias `[1]`, and weights `[-10, -10, 15]` connecting to the output neuron. 

***Logical XOR***

Suppose now we want to solve the XOR problem. Given `[x,y]` in the first layer, we can define four neurons `[A,B,C,D]` in the hidden layer
```js
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

```js
let output = 1 / (1 + exp(-10*(A + C - B - D))); 
```

You can see the above XOR neural network configuration being derived [here](http://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=xor&regDataset=reg-plane&learningRate=0.1&regularizationRate=0&noise=0&networkShape=4,1&seed=0.21709&showTestData=true&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&batchSize_hide=false).

[![tensorflow playground](/images/tensorflow%20playground%20XOR.png)](http://playground.tensorflow.org/#activation=sigmoid&batchSize=30&dataset=xor&regDataset=reg-plane&learningRate=0.1&regularizationRate=0&noise=0&networkShape=4,1&seed=0.21709&showTestData=true&discretize=false&percTrainData=70&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false&batchSize_hide=false)

***Conway's Game of Life***

The rules for [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) are:

```js
if(self == 1 && neighbors == (2|3)) return 1;
if(self == 0 && neighbors == 3) return 1;
else return 0;
```

Layer one of the neural network is going to calculate the basic booleans:

```js
let (self == 1) = 1 / (1 + exp(-10*(self - 0.5))); 
let (neighbors > 1) = 1 / (1 + exp(-10*( neighbors.sum() - 1.5 ))); 
let (neighbors > 2) = 1 / (1 + exp(-10*( neighbors.sum() - 2.5 ))); 
let (neighbors > 3) = 1 / (1 + exp(-10*( neighbors.sum() - 3.5 ))); 
```

Layer two recombines the booleans into AND conditions:

```js
let (self == 0 && neighbors == 3) = (!(self == 1) && (neighbors > 2) && !(neighbors > 3)) //need to test for (0,1,0)
```

In order to test for point (0,1,0), we need to define a plane that separates the vertex of a unit cube from the rest of the cube. Some messing around in [Geogebra](https://www.geogebra.org/3d) gives us `plane = -x + y - z - 0.5`.
```js
let x = (self == 1);
let y = (neighbors > 2);
let z = (neighbors > 3);
let (self == 0 && neighbors == 3) = 1 / (1 + exp(-10*(-x + y - z - 0.5)));
```

And for the second neuron in the second layer: 

```js
let (self == 1 && neighbors == (2|3)) = ((self == 1) && (neighbors > 1) && !(neighbors > 3)) //need to test for (1,1,0)

let x = (self == 1);
let y = (neighbors > 1);
let z = (neighbors > 3);

let (self == 1 && neighbors == (2|3)) = 1 / (1 + exp(-10*(x + y - z - 1.5)));
```

In the output layer, we want to return `(self == 0 && neighbors == 3) || (self == 1 && neighbors == (2|3))`, which we can get by testing if the sum of the values is greater than zero. 

```js
let condition1 = (self == 0 && neighbors == 3);
let condition2 = (self == 1 && neighbors == (2|3));
let output = 1 / (1 + exp(-10*( condition1 + condition2 - 0.5 ))); 
```
You can test this implementation in the browser [here](https://openprocessing.org/sketch/1236584).

At this point we have a working cellular automata. The rules for Conway's game of life are encoded in the weights and biases. We can modify these parameters to explore for alternative rulesets. 
- [NGOL.2](https://openprocessing.org/sketch/1236892)
- [NGOL.3](https://openprocessing.org/sketch/1237046)
- [NGOL.4](https://openprocessing.org/sketch/1237384)
- [NGOL.5](https://openprocessing.org/sketch/1237904)
- [NGOL.6](https://openprocessing.org/sketch/1238150)
- [NGOL.7](https://openprocessing.org/sketch/1238174)
- [NGOL.8](https://openprocessing.org/sketch/1238237)
- [NGOL.9](https://openprocessing.org/sketch/1238436)
- [NGOL.10 (full screen)](https://openprocessing.org/sketch/1243586)
- [NGOL.11](https://openprocessing.org/sketch/1244647)
- [✨Sparkling Polyominoes✨](https://openprocessing.org/sketch/1248369)

Here is [NGOL.3](https://openprocessing.org/sketch/1237046):

```js
let a = 1 / (1 + exp(-7.240 * (self - 0.5))); 
let b = 1 / (1 + exp(-4.584 * (neighbors.sum() - 1.5 ))); 
let c = 1 / (1 + exp(-2.869 * (neighbors.sum() - 2.5 ))); 
let d = 1 / (1 + exp(-7.912 * (neighbors.sum() - 3.8 ))); 
let f = 1 / (1 + exp(-2.799 * (-a + c - d - 0.48)));
let g = 1 / (1 + exp(-3.848 * ( a + b - d - 1.66)));
let output = 1 / (1 + exp(-9.591 * (f + g - 0.455))); 
```

[![neural game of life](/images/neural%20game%20of%20life.png)](https://openprocessing.org/sketch/1237046)

Can we reverse-engineer the program that is encoded in the math? As it turns out, yes, we can nest the above formulas and plot a single function `f(x,y)` where `x` is the neighbor sum and `y` is the self-value. This decision surface yields an almost identical rule set to Conway's rules, with an additional rule: an inactive cell with two neighbors can bootstrap itself up to one-third activation. 

| self : neighbors | 0    | 1    | 2    | 3    | 4    |
| ---              | ---  | ---  | ---  | ---  | ---  |
| 0.00             | 0.08 | 0.08 | 0.27 | 0.96 | 0.15 |
| 0.25             | 0.06 | 0.06 | 0.23 | 0.95 | 0.11 |
| 0.50             | 0.02 | 0.03 | 0.32 | 0.93 | 0.04 |
| 0.75             | 0.02 | 0.03 | 0.32 | 0.93 | 0.04 |
| 1.00             | 0.03 | 0.04 | 0.93 | 0.99 | 0.05 |

![cellular automata decision surface](/images/cellular%20automata%20decision%20surface.png)

This seems like a nice visual demonstration that neural networks are universal function approximators. It implies that neural network "programs" consist of finding an arbitrary surface that maps training input coordinates to a desired output, and relies on interpolation to fill in the gaps (this explains why neural networks may be poor at extrapolating outside the training data). This definition includes recursive programs such as cellular automata, where the function `f(self,world)` returns the next `self` value (see also: RNNs, Q-learning). 

A [demo](https://openprocessing.org/sketch/1255387) of various cellular automata on cube faces:

[![gol cube](/images/gol%20cube.gif)](https://openprocessing.org/sketch/1255387)

A [heatmap](https://openprocessing.org/sketch/1254639) of cell state changes across the neural decision surface:

[![gol decision surface heatmap](/images/NGOL%20decision%20surface%20heatmap.png)](https://openprocessing.org/sketch/1254639)

A few multiple-neighboorhood neural cellular automatas, inspired by [Slackermanz MNCA](https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata/):
- [Grids vs. Gliders population dynamics](https://openprocessing.org/sketch/1271651)
- [Reaction diffusion 1](https://openprocessing.org/sketch/1270299)
- [Reaction diffusion 2](https://openprocessing.org/sketch/1292482)
- [Voronoi web](https://openprocessing.org/sketch/1275178)
- [✨Cells✨](https://openprocessing.org/sketch/1360946)

[![mnca cells](/images/MNCA%20cells.gif)](https://twitter.com/planet403/status/1461723559974391809)

Project idea: extend the concept to image generation; where `x` is the current canvas state, `y` is the internal state, and `z` is the new paint stroke. (See also: Langton's Ant).

Project idea: create a network z = f(x.y) with randomised layers. Draw the 3D surface of the network. When you select a node, highlight the area of the surface that is sensitive to that node. Create some kind of UI to explore weights. Draw a second surface that trains to the first surface in realtime. 

***Convnets***

Interestingly, since every cell in the cellular automata shares the same update rule, then this is technically a "convolutional neural network". The four layers (9:4:2:1) of our neural network form the "kernel", and a grid of kernels are applied to the input image to calculate the output image (the next state of the cellular automata). Our CNN has a "kernel size" of 3, a "step" of 1, and "pads" the input image by wrapping out-of-bound pixels to the opposite edge. 

Using this perspective, we can take a guess at how other convolutional neural networks perform their computations. 

[![Interactive Node-Link Visualisation of Convolutional Neural Networks](/images/aharley%20cnn%20visualisation.png)](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)

This [visualisation](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html) by [Adam Harley](http://www.cs.cmu.edu/~aharley/) shows the filters and layers used to classify handwritten digits. Convolution layer 1 contains six filters, in which every pixel represents the output of a kernel, and each kernel is performing a logical test on the corresponding region of the source image. Filters 1 and 6 detect horizontal edges, filters 3 and 4 detect diagonal edges, filter 2 detects vertical edges, and filter 5 remains a mystery for the ages. In the next layer, Downsampling layer 1, the "max pooling" operation performs a logical OR by preserving the max values. 

In Convolution layer 2, the kernel has been extended to three dimensions, so it can compare results across a stack of filters for one region of the image. This could be used to, for example, multiply horizontal and vertical detectors into a corner detector. 

**How does neural network training work?**
------

Suppose we have a neural network with two input nodes and one output node. This defines a heightmap function f(x,y). We can sample this surface and compare it to our training data points to get the error at each point. A surface may be "trained" by translating and scaling the humps of the logistic components to match the expected output. The "gradient descent" method works out (by way of derivatives) in what direction to move and scale each hump to reduce the training error. 

***Conditional AND***

Let us return to the example of a logical AND statement, where `[x,y]` represent true/false values in the range `[0..1]`:

```js
let AndXY = 1 / (1 + exp(10 * (-x - y + 1.5))); 
```

In a traditional neural net formulation, each input has its own weight, and so would be written like this:

```js
let AndXY = 1 / (1 + exp(a*x + b*y + c))); //where `c` is the bias
```

Each weight `[a,b,c]` can be independently modified to change the decision surface. We can set this up in [Geogebra](https://www.geogebra.org/3d) with `z=\frac{1}{1+e^{a*x+b*y+c)}}` and drag the sliders to see what happens.
- Remember that the form `a*x+b*y+c` is equivalent to the line `y=(-a*x-c)/b`.
- The function `f(x,y)=a*x+b*y+c` returns positive or negative values for points above and below the line. 
- The logistic function maps positive and negative numbers to the range `[0..1]`.
- Dragging `a` rotates the line around `y=-c/b`, a point on the y-axis.
- Dragging `b` rotates the line around `x=-c/a`, a point on the x-axis. (Refer to the slope and x-intersect of the equivalent line, `x=(-b*y-c)/a`).
- In addition to rotating the line, you can scale `[a,b,c]` together in such a way that the line is not rotated or translated, but the function `f(x,y)=a*x+b*y+c` returns a more extreme value. This causes a sharper slope on the logistic function as a result. 
- In summary, we need to achieve two goals: (1) get the correct line position and orientation, and (2) scale the parameters to change the decision slope.

We want a method to adjust `[a,b,c]` until the decision surface returns the values we expect. 

Suppose that we initialise the network with small non-zero weights (as seems to be the convention). 

```js
let AndXY = 1 / (1 + exp(0.01*x - 0.01*y - 0.01))); 
```

This returns the values:

| x | y | z      | Expected Value (t) | Error   | Loss Function (half squared error) | dLoss/da     | dLoss/db     | dLoss/dc     |
|---|---|---     |---                 |---      |---                                 |---           |---           |---           |
| 0 | 0 | 0.5025 | 0                  | 0.5025  | 0.126253                           | 0            | 0            | 0.125621854  |
| 0 | 1 | 0.505  | 0                  | 0.505   | 0.127512                           | 0            | 0.126237334  | 0.126237334  |
| 1 | 0 | 0.5    | 0                  | 0.5     | 0.125                              | 0.125        | 0            | 0.125        |
| 1 | 1 | 0.5025 | 1                  | -0.4975 | 0.123756                           | -0.124371896 | -0.124371896 | -0.124371896 |

How much should we modify weight `a`? 
- The derivative of `f(x,y)=a*x+b*y+c` with respect to `a` is `x`. Example: when you increase `a` by one unit, then `f(x,y)` increases by one unit of `x`. 
- The derivative of the [logistic function](https://en.wikipedia.org/wiki/Logistic_function) with respect to the input `f(x,y)` is `z*(1-z)`. Example: when you increase its input by a tiny fraction, then the logistic function value increases by the current z-value minus the square of the current z-value. 
- The derivative of the loss function (half squared error) with respect to the logistic function is just the error (z - expected value). (See also: log loss function).

Using the chain rule, the product of these derivatives is the rate of change of error with respect to `a`. 

```js
dLoss/da = x * z * (1 - z) * (z - t);
dLoss/db = y * z * (1 - z) * (z - t);
dLoss/dc = 1 * z * (1 - z) * (z - t);
```

The term `z * (1 - z) * (z - t)` may also be referred to as the _node delta_. 

Allegedly, these loss derivatives tell us in which direction to move the weight in order to arrive at a better answer. The learning rate is used to reduce the step size, so that the change doesn't overshoot the target.

```js
let learningRate = 0.5;
a += learningRate * dLoss/da;
b += learningRate * dLoss/db;
c += learningRate * dLoss/dc;
```

When [implemented](https://openprocessing.org/sketch/1244489) in practice, the network rapidly converges on a very similar solution to our hand-crafted one. 

![Gradient descent to logical AND](/images/gradientDescent.gif)

Project idea: tally up the cumulative changes to the weights. If a weight gets pulled in two directions at once, should it cancel out? (Edit: this is called "batching"). 
 
***Conditional XOR***

The XOR network introduces a middle layer. We need a way to tell non-output nodes what their error values are. 

```js
//conditional XOR

//inputs
let x = random(0, 1); 
let y = random(0, 1); 

//middle layer
let a = 1 / (1 + exp(a1 * x + a2 * y + a3)); //test for (0,1)
let b = 1 / (1 + exp(b1 * x + b2 * y + b3)); //test for (1,1)
let c = 1 / (1 + exp(c1 * x + c2 * y + c3)); //test for (1,0)
let d = 1 / (1 + exp(d1 * x + d2 * y + d3)); //test for (0,0)

//output layer
let output = 1 / (1 + exp(w1 * a + w2 * b + w3 * c + w4 * d + w5)); //test for (0,1) + (1,0) - (1,1) - (0,0)

//solve for weights [a1, a2, a3], [b1, b2, b3], [c1, c2, c3], [d1, d2, d3], [w1, w2, w3, w4, w5]
```

For the middle layers, we use the chain rule to determine the rate of change of the error with respect to the rate of change of the weight. 

```js
let f = (a1 * x + a2 * y + a3);
let g = (w1 * a + w2 * b + w3 * c + w4 * d);

dLoss/da1 = (dLoss/da                                    ) * da/df         * df/da1;
          = (dLoss/dOut       * dOut/dg           * dg/da) * da/df         * df/da1;
          = ((out - expected) * (out * (1 - out)) * w1   ) * (a * (1 - a)) * x;
          
//note: if there are multiple output nodes, then dLoss/da = (dLoss1/da + dLoss2/da), each of which gets its own expansion
```

Note that this only gives us the _local_ slope of error:weight. It may point away from the global optimum. It may stall on a saddle point. Are there better methods for converging on a solution? 

Some thoughts on the [implementation](https://openprocessing.org/sketch/1245380):
- It's hard to recover from a dead node that is invariant to input. 
- Should penalise two nodes doing the same calculation, it traps the gradient in a bad local minima.
- Is there a better way to formulate the line, say as polar coordinates so we can scale/rotate/translate with single parameters? Example: frequently we just want to crank up the logistic slope by multiplying all parameters by a common factor. 
- I don't understand why I have to swap the sign of the weight adjustments in each layer. Might be an error on my part, but it works...

![training XOR](/images/training%20XOR.gif)

***Conway's Game of Life***

Here is [Conway's Game of Life](https://openprocessing.org/sketch/1248243) being trained as a neural network: 

![training NGOL](/images/training%20cellular%20automata.gif)

**Training multiple layers**
------

What is a good problem to model with a 5 layer network? 

```js
//a quick recap of the operations for union, intersection, difference of implicit hypersolids
//a node is essentially limited to bisecting a hypercube of booleans, 
//each vertex corresponds to AND(TRUE,FALSE,...,TRUE)
//if the hyperplane fences off more than one vertex, then you get OR(AND(...),AND(...))
let intersect  = 1 / (1 + exp(-10 * (-1.5 + x + y))); //returns true for x && y
let union      = 1 / (1 + exp(-10 * (-0.5 + x + y))); //returns true for x || y
let subtract   = 1 / (1 + exp(-10 * (-0.5 + x - y))); //returns true for x && !y
```

Assuming the training data is normalized, then we can say that each training point is a coordinate inside a unit hypercube. 
- Layer 1 composes a set of hyperplanes (and normals) with which to cut the unit hypercube.
- Layer 2 composes convex hulls from the planes in layer 1 (lines => polygon, planes => solid, hyperplanes => hypersolid). 
- Layer 3 composes concave hulls and internal voids from layer 2.
- Layer 4 may union outputs from layer 3 to test any arbitrary region in the input space. 

Side note, it is kind of neat that you can test if a point is inside a convex solid by testing the normals of each individual face. 

Looking at the above, we can identify a toy problem to execute in four layers: the symmetric intersection of three triangles. 
- Input layer takes [x,y] coordinates
- Layer 1 composes 9 lines
- Layer 2 composes 3 hulls
- Layer 3 composes 4 intersection/exclusion operations
- Layer 4 composes 1 union operation

Demo: [Symmetric Difference of Three Triangles](https://openprocessing.org/sketch/1365093)

[![symmetric difference](/images/symmetric%20difference%20training.gif)](https://openprocessing.org/sketch/1365093)

I tried training this network first on an infinite number of random samples, and then again on a fixed number of samples. I found that biases in the training data strongly affect the output: in the case of random sampling, larger areas receive more weight updates than small areas; in the case of fixed sampling, the network can exploit gaps in the data to overfit the known points and return garbage on unknown points. 

The network converges on the exact solution if seeded with nearby weights, but gets stuck in local optima when training from scratch. I suspect this is because there is a gradient hump between "donut shape" and "bullseye shape", the solution gets worse before it gets better. Though it is also worth noting that the perfect solution is not attainable with weight scales below 20.

There are some philosophical questions here. We (the omnipotent observer) can observe that a solution is wrong, even though it returns near-correct values for all known sample points. We know this because we can observe the contours of the function, make inferences about the probability of an unknown sample value based on nearby points (using k-means, or SVM, or another), and test our hypotheses by comparing the prediction/outcome of a point in the space. We have some preconceived notion of what the "true hull" of the data points might look like. The implication here is that we aren't training from a blank slate: we start out with some baseline heuristic to initialise the hull weights and then iteratively test/train/update to fine-tine the boundaries. You could consider this to be the difference between design and iteration: we can improve bicycle designs using gradient descent, but one may reasonably infer that a different algorithm was used to design the first bicycle. 

Project idea: as the network converges, adversarially generate test points by reading backwards from the output layer. Or design an LSTM to generate its own test data, where the objective function is minimising training time. 
Project idea: swap out single weight updates with translate/scale/rotate operations on a force-directed particle swarm. 

**Recurrent Neural Networks** 
------

Some notes.

**LSTM Network** 
------

A decent diagram [here](https://blog.mlreview.com/understanding-lstm-and-its-diagrams-37e2f46f1714).

An LSTM is a for-loop that reads and writes to a memory array while appending results to a list. As far as I can tell this basically lets us declare implicit variables and access them in the loop if a condition is triggered. 

```js
loop()
{
  //array of inputs
  let inputs = concat(new_input_array, prev_out_array);
  	
  //arrays of ones and zeroes
  let erase_filter = logistic(crossproduct(inputs, weights[erase_f])); 
  let write_filter = logistic(crossproduct(inputs, weights[write_f]));	
  let read_filter =  logistic(crossproduct(inputs, weights[read_f])); 
  
  //array of values in range [-1..1]
  let write_values = tanh(crossproduct(inputs, weights[write_v])); 			
  
  //array of values to read/write in the loop
  memory_array *= erase_filter;                   //reset variables
  memory_array += (write_values * write_filter);  //increment/decrement variables
  
  //array of values in range [-1..1]
  let read_values = tanh(crossproduct(memory_array, weights[read_v])); 	
  
  //array of values to append to results[].
  let out_array = (read_values * read_filter);
}
```

Suppose we have a "memory cell" which is an array `[x, y, z]`. Each step of the loop, we pass forward the memory `[x,y,z]` and the incremental output `[a,b,c]`. The memory is an accumulator, we can increment or decrement `x` based on conditions in the step, or reset `x` to zero. The output is a temp variable, a reference to the previous step of the loop. The logistic filters encode the logic for deciding to erase, increment, or read a value `[x,y,z]` in memory, depending on the context of the current step inputs and previous step outputs.

For example, below is a toy LSTM for calculating the modulo function. 

```js
let memory = 0;           //array of parameters in memory
let divisor = 2;          //input at each loop step
let current = 0;          //output at each loop step
let results = [];         //function output

for(let i = 0; i < 10; i++)
{
 let eraser = 1 / (1 + exp(-10 * ( 0.5 + divisor - memory)));   //we want to multiply by zero if memory > divisor
 let writer = 1 / (1 + exp(-10 * (-0.5 - divisor + memory)));   //we erase, or we increment, but not both
 let reader = 1 / (1 + exp(-10 * (1.0)));                       //always read memory
 
 let writeValue = tanh(100.0);                                 //if we increment at all, we increment by 1
 
 memory *= eraser;
 memory += (writer * writeValue); 
 
 let readValue = ReLU(memory); //a sneaky ReLU because I don't want to squash the values
 
 current = reader * readValue; 
 results.push(current); 
}
```

What kind of crazy solutions might gradient descent come up with? Note that we can do multiple tasks, such as classify a time series, or predict the next token in a sequence. 

Suppose that the output is a one-hot vector classification. I think I can reasonably argue that this is classifying the (filtered) memory state vector. If we are predicting tokens, then the erase/write/read rules can encode a finite state machine by bouncing the memory state vector aorund in vector space (which the output layer then classifies). In this case we can stop thinking in terms of incrementing single variables, and start paying attention to the direction of the update vectors in memory space. 

Hypothesis: LSTMs encode the graph of a finite state machine where nodes are positions in memory space and edges are offsets in memory space. We can imagine the writer layer as a vector field in memory space (returning an edge vector to move memory to the new node); but it also has dimensions in input space: different inputs return different vector fields, in order to cross different edges. Adding the edge vector to the memory vector moves the finite state machine to a new node. The readout layer is kinda doing the same thing, except the output could be a label of the edge being crossed, rather than a UID. 

Project challenge: try to write an LSTM finite state machine that works by bouncing the memory state vector around in N-space. 
 
Project idea: translate a neural network training loop into a finite state machine, then formulate that as an LSTM. 

***Clockwise around a unit square***

Use the `tanh` addition and `logistic` multiplication to yield a clockwise enumeration of vertices on a unit square, in memory. The trick here is that `tanh` can return three values (-1,0,1), and then we get a second pass in which to multiply some components by zero. Conceivably we can use the `erase` gate to pack in even more complexity. 

```js

//move clockwise around the unit square
//   (0,1)-->(1,1)
//     ^       |
//     |       v
//   (0,0)<--(1,0)

//(0, 0) + (-1, 1) * (0, 1) =  (0, 1) 
//(0, 1) + ( 1, 0) * (1, 1) =  (1, 1) 
//(1, 1) + ( 0,-1) * (1, 1) =  (1, 0) 
//(1, 0) + (-1,-1) * (1, 0) =  (0, 0) 

let memory = [x, y]; 

write_values = 
[
  Math.tanh(-10 * (-y + 0.5 * x + 0.5)),
  Math.tanh(-10 * ( y + 2.0 * x - 1.0)),
];

let write_filter = 
[
  1 / (1 + exp(-10 * (y + x - 0.5))),
  1 / (1 + exp(-10 * (y - x + 0.5))),
];

x += (write_values[0] * write_filter[0]);
y += (write_values[1] * write_filter[0]);

memory = [round(x), round(y)]; //note the naughty round operation to snap to a vertex. not part of a conventional LSTM. 

``` 

***Enumerate vertices on a unit cube***

We can do this by enumerating a square, with a bit flip on the z axis if the xy coordinate is on a given edge. This probably extends to any hypercube. 

```js

//(0, 0, 0) + (-1, 1, 1) * (0, 1, 0) =  (0, 1, 0) 
//(0, 1, 0) + ( 1, 0, 1) * (1, 1, 0) =  (1, 1, 0) 
//(1, 1, 0) + ( 0,-1, 1) * (1, 1, 0) =  (1, 0, 0) 
//(1, 0, 0) + (-1,-1, 1) * (1, 0, 1) =  (0, 0, 1) //transition to top plate
//(0, 0, 1) + (-1, 1,-1) * (0, 1, 0) =  (0, 1, 1) 
//(0, 1, 1) + ( 1, 0,-1) * (1, 1, 0) =  (1, 1, 1) 
//(1, 1, 1) + ( 0,-1,-1) * (1, 1, 0) =  (1, 0, 1) 
//(1, 0, 1) + (-1,-1,-1) * (1, 0, 1) =  (0, 0, 0) //transition to bottom plate

let memory = [x, y, z]; 

write_values = 
[
  Math.tanh(-10 * (-y + 0.5 * x + 0.5)),
  Math.tanh(-10 * ( y + 2.0 * x - 1.0)),
  Math.tanh(-10 * ( z - 0.5)),
];

let write_filter = 
[
  1 / (1 + exp(-10 * ( y + x - 0.5))),
  1 / (1 + exp(-10 * ( y - x + 0.5))),
  1 / (1 + exp(-10 * (-y + x - 0.5))), 
];

x += (write_values[0] * write_filter[0]);
y += (write_values[1] * write_filter[1]);
z += (write_values[2] * write_filter[2]);

memory = [round(x), round(y), round(z)]; //note the naughty round operation to snap to a vertex. not part of a conventional LSTM. 

```

***Dealing with input: XOR***

Suppose we try to model XOR with an LSTM. The memory is a 1D boolean. The input is a 1D boolean. The memory-input space is not linearly separable. We can use the `tanh` function return a default move vector based on the vertex, and then use `logistic` to zero some components based on the input. 

If the memory is a hypercube, and if the `tanh` output is a diagonal vector from current vertex, then the `logistic` tests could let us reach any vertex on the cube. So, we should be able to represent any finite state machine by mapping states to vertices on a hypercube. 

```js
//(memory, input) => memory
//(0, 0) => 0
//(0, 1) => 1
//(1. 0) => 1
//(1, 1) => 0

let memory = m;
let input = p; 

write_values = 
[
  Math.tanh(-10 * (m - 0.5)), //by default, flip the memory 
];

write_filter = 
[
 1 / (1 + exp(-10 * (p - 0.5))), //if the input is zero, don't flip the memory
];

memory += (write_values[0] * write_filter[0]);

```

***Reber Grammar***

[Reber Grammer](http://christianherta.de/lehre/dataScience/machineLearning/neuralNetworks/reberGrammar.php) appears to be the canonical 'Hello World' for LSTMs. 

```js

//transition rules from node[index] to next node
let moves = 
[
 [[1, 'T'], [5, 'P']],
 [[1, 'S'], [2, 'X']],
 [[3, 'S'], [5, 'X']],
 [[6, 'E']          ],
 [[3, 'V'], [2, 'P']],
 [[4, 'V'], [5, 'T']]
];

function getSequence(len)
{
 let str = 'B',
 let node = 0; 
 
 while(str.length < len)
 {
  let move = random(moves[node]);
  str += move[1]; 
  node = move[0];
  
  if(node == 6) break;
 }
 
 return str;
}

```

How might one formulate this as an LSTM? 

[First attempt:](https://openprocessing.org/sketch/1412417)
- `input` is a one-hot vector representing the token.
- `memory` is a one-hot vector representing the FSM state. 
- `writer` tests for node-edge pairs. It returns false positives! But that's okay, we can erase them with the write filter.
- `filter` tests for false positives in `writer` and then zeroes them. This allows us to nest our logic (yay black box code golf). There is no read filter in this example, but I imagine it can filter false positives from the output layer in much the same manner.
- `eraser` resets the memory state. 

Unfortunately this formulation neglects to use the `tanh` activation function, so I can't really comment if my use of the `filter` layer is correct or not. A logistic multiplied by a logistic gets us a nice little logistic distribution, which we can use to isolate any two vertices in a polygon. 

```js

let tokens  = "BTSXPVE"; 
let input   = [1,0,0,0,0,0,0]; //current token [B,T,S,X,P,V,E]
let memory  = [1,0,0,0,0,0,0]; //one-hot vector of current node in graph
let eraser  = [0,0,0,0,0,0,0]; //what to erase in memory
let writer  = [0,0,0,0,0,0,0]; //what to write to memory
let filter  = [1,1,1,1,1,1,1]; //filter the writer when it returns multiple write values
let output  = [0,0,0,0,0,0,0,0]; //predicted tokens [B,T,S,X,P,V,E,-]

let str = 'B';

while(str.length < len)
{
  //figure out which node to move to
  //technically these should be tanh functions, whoops
  writer[0] = 1 / (1 + exp(-10 * (input[0] - 0.5)));                                     //move to node 0 from input B
  writer[1] = 1 / (1 + exp(-10 * (memory[0] + input[1] + memory[1] + input[2] - 1.5)));  //move to node 1 if([0,"T"] or [1,"S"])
  writer[2] = 1 / (1 + exp(-10 * (memory[1] + input[3] + memory[4] + input[4] - 1.5)));  //move to node 2 if([1,"X"] or [4,"P"])
  writer[3] = 1 / (1 + exp(-10 * (memory[2] + input[2] + memory[4] + input[5] - 1.5)));  //move to node 3 if([2,"S"] or [4,"V"])
  writer[4] = 1 / (1 + exp(-10 * (memory[5] + input[5]                        - 1.5)));  //move to node 4 if([5,"V"] or [7, "V"])
  writer[5] = 1 / (1 + exp(-10 * (memory[0] + input[4] +                                 //move to node 5 if([0,"P"] or [2,"X"] or [5,"V"]) 
                                  memory[2] + input[3] +                                 //note this returns false postives on [0,"T"] **************
                                  memory[5] + input[1] - 1.5)));                         //write filter to the rescue! 
  writer[6] = 1 / (1 + exp(-10 * (memory[3]                                   - 0.5)));  //move to node 6 from node 3
  
  filter  = [1,1,1,1,1,1,1,1];
  filter[5] = 1 / (1 + exp(-10 * (-memory[0] - input[1] + 1.5))); //filter out false positives in the writer
  
  for(let i = 0; i < memory.length; i++) { memory[i] *= eraser[i]; }
  for(let i = 0; i < writer.length; i++) { writer[i] *= filter[i]; }
  for(let i = 0; i < memory.length; i++) { memory[i] += writer[i]; }

  output[0] = 0; //we never yield B
  output[1] = 1 / (1 + exp(-10 * (memory[0] + memory[5] - 0.5))); //T may yield from 0 or 5
  output[2] = 1 / (1 + exp(-10 * (memory[1] + memory[2] - 0.5))); //S may yield from 1 or 2
  output[3] = 1 / (1 + exp(-10 * (memory[1] + memory[2] - 0.5))); //X may yield from 1 or 2
  output[4] = 1 / (1 + exp(-10 * (memory[0] + memory[4] - 0.5))); //P may yield from 0 or 4 
  output[5] = 1 / (1 + exp(-10 * (memory[4] + memory[5] - 0.5))); //V may yield from 4 or 5
  output[6] = 1 / (1 + exp(-10 * (memory[3]             - 0.5))); //E may yield from 3
  output[7] = 1 / (1 + exp(-10 * (memory[6]             - 0.5))); //- yields from 6

  let c = output
    .map((e, i) => [e, i]) 	
    .filter(e => e[0] > 0.5) 	
    .map(e => e[1]); 					
  ;
  let newToken = random(c);
  
  //append token to string
  str += tokens[newToken]; 
  
  //graph exit condition
  if(newToken == 6) break; 
  
  //set next input
  input = [0,0,0,0,0,0,0];
  input[newToken] = 1;
}

return str;
                        
```

Second attempt using `tanh` activations:

The crux of the problem is detecting edge T into node 5.

```js

let node[5] = (node[0] && edge[P]) || (node[2] && edge[X]) || (node[5] && edge[T]); 
            = (node[0] + edge[P]) + (node[2] + edge[X]) + (node[5] + edge[T]); //returns false positives on T from node 0.

```

Thanks to multiplying `tanh` by `logistic`, each node can return a value where 1 is true, 0 is false, and -1 is false. This means we have to ditch boolean algebra. It means we can no longer predict the outcome of the sum of an array of truth values. 

If the whole point of using `tanh` functions is to be able to increment and decrement variables, then maybe we are approaching this from the wrong angle. If we view a memory component as 'collecting evidence for condition X', then we could do something like 'the sum of valid edges to reach a node'. So `B+T` is too low threshold for node 5, but `B+P+T` rings the bell. And if you deviate from the prescribed path, erase or decrement the vector appropriately. 

```js
//[B,T,S,X,P,V,E]

//evidence for node 0 is B          
//evidence for node 1 is BTS^       
//evidence for node 2 is [TX,SX]    
//evidence for node 3 is [XS,PS,VV] 
//evidence for node 4 is [PV,TV,XV] 
//evidence for node 5 is [BPT^,XXT^] or [BP, PT, XX, PX, XT, TT]

memory[5] = 0;

//if we increment different amounts for different edges, then we can game the edge order 

let increment_B = 0.4; 
let increment_P = 0.6; 
let increment_T = 0.2; 
let increment_X = 0.5; 

//we hit node 5 if edges sum to > 0.6
//B + T = 0.6 
//B + P = 1.0
//X + X = 1.0
//P + X = 1.1

//reset accumulator if S or V
eraser[5] = 1 / (1 + exp(-10 * (0.5 - S - V)));

//write the increment corresponding to the input 
//tanh(0.197) = 0.2 
//tanh(0.348) = 0.334
//tanh(0.424) = 0.4 
//tanh(0.55 ) = 0.5 
//tanh(0.7  ) = 0.6 
//tanh(1.098) = 0.8 

writer[5] = tanh(0.55 * X + 0.70 * P + 0.424 * B + 0.197 * T); 

memory[5] = memory[5] * eraser[5] + writer[5];

//node 5 is active is memory > 0.6 
node[5] = tanh(10 * (memory[5] - 0.6);

//predicting T requires activating on node[0] or node[5]
//so we have some hypthetical weight [w] here to scale memory[0] activation threshold to 0.6
reader[T] = tanh(10 * (memory[5] + w * memory[0] - 0.7);

```

This is surprisingly effective, although I don't care for the coding style. The FSM is basically baked into the program, it's useless for any other task. 
- It would be nice if the model printed out the finite state machine so we could see what it's doing.
- We shouldn't have to retrain the whole model on new datasets - it should encode the FSM extraction algorithm. 
- Ideally, a small change in training data should be equivalent to a small change in the FSM - can we reuse existing elements? 

Project challenge: write a program that returns a valid finite state machine given a dataset of sequences. How is it different from the above?

On with the show:

```js

eraser[0] = 0;                                                  //always erase
writer[0] = tanh(5 * B);                                        //increment on B, else do nothing 

//BT
eraser[1] = 1 / (1 + exp(-10 * (0.5 - X - V - P)));             //reset on X,V,P
writer[1] = tanh(0.348 * (B + T + S));                          //breadcrumbs to node 1
filter[1] = 1 / (1 + exp(-10 * (0.9 - memory[5])));             //ignore T from node 5

//P_P, TX, X_P
eraser[2] = 1 / (1 + exp(-10 * (0.9 - memory[2])));             //reset on exit
writer[2] = tanh(0.55 * (T + X + P));                           //breadcrumbs to node 2
filter[2] = 1 / (1 + exp(-10 * (0.9 - memory[5])));             //do not increment on exit node 5

//BP, XX, PX
eraser[5] = 1 / (1 + exp(-10 * (0.5 - S - V)));                 //reset on S,V
writer[5] = tanh(0.55 * X + 0.70 * P + 0.424 * B + 0.197 * T);  //breadcrumbs to node 5

//PV, XV but not XX and not PX
let inc_X = 0.2; 
let inc_P = 0.5; 
let inc_V = 0.8;
eraser[4] = 1 / (1 + exp(-10 * (0.9 - memory[4])));             //reset on exit
writer[4] = tanh(0.197 * X + 0.55 * P + 1.098 * V);             //breadcrumbs to node 4

//S (filter node 1) or VV
eraser[3] = 0;                                                  //always erase
writer[3] = tanh(3.0 * S + 0.55 * V);                           //breadcrumbs to node 3
filter[2] = 1 / (1 + exp(-10 * (0.9 - memory[1])));             //do not increment on exit node 1

//E
eraser[6] = 0;                                                  //always erase
writer[6] = tanh(E * B);                                        //increment on B, else do nothing 

```


[polysemantic neurons](https://distill.pub/2020/circuits/zoom-in/)







See also 'Reber grammar' https://www.bioinf.jku.at/publications/older/2604.pdf

**Collatz Conjecture sequences with a Recurrent Neural Network** 
------

A sequence is obtained from any integer by recursively dividing by 2 until odd, then applying `3*n+1` to make even. It is conjectured that this sequence terminates at the number 1 in all cases. 

```
function getNextCollatzElement(n)
{
  let isEven = (n % 2 == 0);
  if(isEven) return n/2; 
  else return 3*n+1;  
}
```

How do we implement the `modulo` and `round` operators in a neural network? 
- Use a sin/cos activation function. Pros: fast. Cons: requires domain knowledge for node placement.
- Recursively divide by divisor until exit condition. Cons: still requires a `round` function for the exit condition.
- Cheat a little and represent all inputs in binary. Cons: doesn't generatlise.
- Feed in `floor(x/2)` as a second input, which allows us to do `y = 1 / (1 + exp(-10 * (x/2 - floor(x/2) - 0.5))` for detecting even/odd. This feels like cheating though.

***Attempt with sin()***

Let's break down the logic: 

```
//input layer
let input = round(random(1E9)); 

//layer 1
let a = sin(PI / 2 * input); //returns zero for evens

//layer 2
let b = 1 / (1 + exp(-50( a+0.1))); //return a >= 0
let c = 1 / (1 + exp(-50(-a+0.1))); //return a <= 0

//layer 3
let d = 1 / (1 + exp(50 * (-b - c + 1.5))); //return b && c; true for even, false for odd

//output layer
let answer = d * (input / 2) + (1 - d) * (3 * input + 1);
```

Some interesting things to note:
- The input value gets passed straight to the output layer.
- The output layer applies an element-wise multiplication "gate" akin to LSTM gates.

Can we pass control of the for-loop to the neural network? Yes: if the second output node is less than 0.5, we exit the loop. 

```
//loop control
let stopToken = 1 / (1 + exp(-50*(-input + 1.5))); //break loop if input is less than 1.5
if(stopToken > 0.5) return; //exit loop
```

Here is the [half-finished implementation](https://openprocessing.org/sketch/1259589). 

```
                           +--------+                                    
                        +->| node B |-+                                  
+-------+   +--------+  |  +--------+ |   +--------+                     
| input |-->| node A |--+             +-->| node D |----------+          
+-------+   +--------+  |  +--------+ |   +--------+          |          
    |                   +->| node C |-+                       |          
    |                      +--------+                         V          
    |                                                  +-------------+   
    |------------------------------------------------->|   output    |
    |                                                  +-------------+
    |                                                                
    |                                                  +-------------+
    +------------------------------------------------->| loopControl |
                                                       +-------------+
```

This formulation appears to trade-off the neat and simple layer concept for a more compact and modularised program flow. Questions for the future:
- Can (some of) the popularity of ReLU be explained by its ability to pass values unchanged across multiple layers?
- If the program structure is not known at training time, does it make sense to introduce modularity? 
- Is there a way to make the composition of neural operators part of the training/solving process? 
- The solution appears to be a "sharp minima"; any tiny errors rapidly snowball when generating a sequence with recursion. 

Similar problems noted (and remedied) in the paper [Recursively Fertile Self-replicating Neural Agents](https://direct.mit.edu/isal/proceedings/isal/58/102906):
> Unfortunately, these neural quines appear to become completely infertile after just one self-replication step. Specifically, we mean that the parameters of the descendants diverge significantly from the ancestors over two generations and become quickly chaotic or a trivial fixpoint, and with this their performance on any given auxiliary task degrades uncontrollably. 

***Attempt with floor()***

```
let input1 = x;
let input2 = floor(x/2);

let evenSignal1 = 1 / (1 + exp(-10 * ( input1/2 - input2 - 0.75))); //tests even down to next odd
let evenSignal2 = 1 / (1 + exp(-10 * (-input1/2 + input2 + 0.25))); //tests even up to next odd

//if this is a gate, then we need two separate gate values
let isEven = 1 / (1 + exp(-50 * ( evenSignal1 + evenSignal2 - 0.5)));
let isOdd  = 1 / (1 + exp(-50 * (-evenSignal1 - evenSignal2 + 0.5)));

let answer = isEven * (input / 2) + isOdd * (3 * input + 1);
```

Like the previous attempt, this diverges rapidly if the `evenSignal` returns anything less than a binary result. But it works, I guess.


**Neural Floor Nodes** 
------

Is there a way to utilise `floor()` in the activation function? 

```
//wraps inputs [x,x] on the unit square
z = 1 / (1 + exp(-10 * (x - floor(x) + y - floor(y) - 0.5))
```

```
//wraps input x in range [0..2], could use for mod functions like round()
y = 1 / (1 + exp(-10 * (x/2 - floor(x/2) - 0.5))
  = 1 / (1 + exp(-5*x + 10*floor(x/2) - 0.5))
```

There is something to the idea of wrapping the dot products in some hypervolume, but I don't know that there's a differentiable and generic way to do it. 
- Scaling a weight on `floor` doesn't really make a whole lot of sense.
- We want both a weight on `floor` and a weight on the input to `floor`, which confuses things. 



Projects up next:
- Neural Langton's Ant; Neural Physarum; 
- Neural Reaction-Diffusion
- Neural multiplication / exponentiation. Is this just gated RNNs?
- Train another neural network


**What is the hype with machine learning?**
------

- A basic neural net is a classifier. It decides if data is above or below a classifying line. Useful but not super exciting. 
- With a little sauce you can run the neural net in reverse, starting with a classification working backwards to a sample point. This can be used for generating novel samples (faces, art, ...). 
- If you take a generator, and feed it back into a classifier,then you get a generative adversarial network. This trains the generator to match training data.
- If you take a classifier, and feed it into a generator, then you get an autoencoder. This trains the encoder to find efficient descriptors of the training data.
- If you you take a classifier, and train it in on actions in an environment, then you get Q learning. This matches states to actions for highest reward.
- If you take a series of classifiers, and use them to map an image to a bank of class arrays, then you get a convolutional neural net. The individual classifiers ("kernels") are trained to extract features from an image. 

**What is supervised and unsupervised machine learning?**

TBA.

**What is online and offline learning?**

TBA.

**What is a support vector machine?**

[Guide to AI algorithms](https://www.youtube.com/watch?v=9PBqqx38WeI&list=PLRKtJ4IpxJpDxl0NTvNYQWKCYzHNuy2xG).

**What is a one-hot vector?**

This: {0, 0, 1, 0, 0}. A one-hot vector has all components set to zero except for one element. The vector represents the probability that an item is classified as element A, B, C, etc.

**What is a neural network?**

A neural networks transforms points from one coordinate system into another coordinate system, until the various categories we are looking for (dogs, traffic lights, action prompts) form clusters. Then we can classify a new data point by how closely it maps to each cluster. 

[Video](https://www.youtube.com/watch?v=UOvPeC8WOt8).

[Essay](http://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)

[Understanding LSTM](https://arxiv.org/abs/1909.09586): this paper starts at perceptrons and works its way up to LSTMs with crystal clear communication.

**What is a tensor?**

In code, its a multi-dimensional array. But there are some math rules about how to use it (they are typed, and have vector-like properties). 

From [Introduction to Tensors](https://www.grc.nasa.gov/www/k-12/Numbers/Math/documents/Tensors_TM2002211716.pdf):
- All scalars are not tensors, although all tensors of rank 0 are scalars.
- All vectors are not tensors, although all tensors of rank 1 are vectors.
- All matrices are not tensors, although all tensors of rank 2 are matrices.
- Tensors can be multiplied by other tensors to form new tensors.
- The product of a tensor and a scalar (tensor of rank 0) is commutative.
- The pre-multiplication of a given tensor by another tensor produces a different result from post-multiplication; i.e., tensor multiplication in general is not commutative.
- The rank of a new tensor formed by the product of two other tensors is the sum of their individual ranks.
- The inner product of a tensor and a vector or of two tensors is not commutative.
- The rank of a new tensor formed by the inner product of two other tensors is the sum of their individual ranks minus 2.
-  A tensor of rank n in three-dimensional space has 3n components. 

Tensors are multi-dimensional arrays. Tensors follow transformation rules, and have types. 

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

**What is an embedding?**

A mapping of an object to an arbitrary vector. Ideally this creates some kind of clustering in N-space by class. 

Example: maybe we have a dataset of elephants described by 10,000 parameters. To massage the data into useful clusters, maybe we can map that 10,000 N-space down to a 2D space with only two paramters. 

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

[https://ericmjl.github.io/essays-on-data-science/machine-learning/graph-nets/](https://ericmjl.github.io/essays-on-data-science/machine-learning/graph-nets/)

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
