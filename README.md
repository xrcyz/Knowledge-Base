# Knowledge Base

**What are some fun project ideas?**

- minecraft
- dungeon generator
- asteroid field generator
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

See also [Morphogenic resources](https://github.com/jasonwebb/morphogenesis-resources)

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

In the above, the line `(y = 1.5 - x)` is used to test if a point `[x,y]` is in the top right of the unit square. The logistic function converts the output to a `[0..1]` range, and the multiplier `10` is used to sharpen the transition slope. If this were diagrammed as a neural net, the second layer would have two neurons `[x, y]`, a bias `[1]`, and weights `[-10, -10, 15]` connecting to the output neuron. 

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
if(self == 1 && neighbors == (2|3)) return 1;
if(self == 0 && neighbors == 3) return 1;
else return 0;
```

Layer one of the neural network is going to calculate the basic booleans:

```
let (self == 1) = 1 / (1 + exp(-10*(self - 0.5))); 
let (neighbors > 1) = 1 / (1 + exp(-10*( neighbors.sum() - 1.5 ))); 
let (neighbors > 2) = 1 / (1 + exp(-10*( neighbors.sum() - 2.5 ))); 
let (neighbors > 3) = 1 / (1 + exp(-10*( neighbors.sum() - 3.5 ))); 
```

Layer two recombines the booleans into AND conditions:

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

Here is [NGOL.3](https://openprocessing.org/sketch/1237046):

```
let a = (self == 1) = 1 / (1 + exp(-7.24*(self - 0.5))); 
let b = (neighbors > 1) = 1 / (1 + exp(-4.584*( neighbors.sum() - 1.5 ))); 
let c = (neighbors > 2) = 1 / (1 + exp(-2.869*( neighbors.sum() - 2.5 ))); 
let d = (neighbors > 3) = 1 / (1 + exp(-7.912( neighbors.sum() - 3.8 ))); 
let f = (self == 0 && neighbors == 3) = 1 / (1 + exp(-2.799*(-a + c - d - 0.48)));
let g = (self == 1 && neighbors == (2|3)) = 1 / (1 + exp(-3.848*(a + b - d - 1.66)));
let output = 1 / (1 + exp(-9.591*( f + g - 0.455))); 
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

This seems like a nice visual demonstration that neural networks are universal function approximators. It implies that neural network "programs" consist of finding an arbitrary surface that maps training input coordinates to a desired output, and relies on interpolation to fill in the gaps (this explains why neural networks may be poor at extrapolating outside the training data). This definition includes recursive programs such as cellular auomata, where the function `f(self,world)` returns the next `self` value (see also: RNNs, Q-learning). 

A final demo of various cellular automata on cube faces:

[![gol cube](/images/gol%20cube.gif)](https://openprocessing.org/sketch/1255387)

[Project idea](https://openprocessing.org/sketch/1254639): CA grid, 3D plot of decision surface, sliders for weights and biases, and a fading heatmap of cell states on the surface.  
Project idea: extend the concept to image generation; where `x` is the current canvas state, `y` is the internal state, and `z` is the new paint stroke. (See also: Langton's Ant).

***Convnets***

Interestingly, since every cell in the cellular automata shares the same update rule, then this is technically a "convolutional neural network". The four layers (9:4:2:1) of our neural network form the "kernel", and a grid of kernels are applied to the input image to calculate the output image (the next state of the cellular automata). Our CNN has a "kernel size" of 3, a "step" of 1, and "pads" the input image by wrapping out-of-bound pixels to the opposite edge. 

Using this perspective, we can take a guess at how other convolutional neural networks perform their computations. 

[![Interactive Node-Link Visualisation of Convolutional Neural Networks](/images/aharley%20cnn%20visualisation.png)](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html)

This [visualisation](https://www.cs.ryerson.ca/~aharley/vis/conv/flat.html) by [Adam Harley](http://www.cs.cmu.edu/~aharley/) shows the filters and layers used to classify handwritten digits. Convolution layer 1 contains six filters, in which every pixel represents the output of a kernel, and each kernel is performing a logical test on the corresponding region of the source image. Filters 1 and 6 detect horizontal edges, filters 3 and 4 detect diagonal edges, filter 2 detects vertical edges, and filter 5 remains a mystery for the ages. In the next layer, Downsampling layer 1, the "max pooling" operation performs a logical OR by preserving the max values. 

In Convolution layer 2, the kernel has been extended to three dimensions, so it can compare results across a stack of filters for one region of the image. This could be used to, for example, multiply horizontal and vertical detectors into a corner detector. 

**How does neural network training work?**
------

***Conditional AND***

Let us return to the example of a logical AND statement, where `[x,y]` represent true/false values in the range `[0..1]`:

```
let AndXY = 1 / (1 + exp(10 * (-x - y + 1.5))); 
```

In a traditional neural net formulation, each input has its own weight, and so would be written like this:

```
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

```
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

```
dLoss/da = x * z * (1 - z) * (z - t);
dLoss/db = y * z * (1 - z) * (z - t);
dLoss/dc = 1 * z * (1 - z) * (z - t);
```

The term `z * (1 - z) * (z - t)` may also be referred to as the _node delta_. 

Allegedly, these loss derivatives tell us in which direction to move the weight in order to arrive at a better answer. The learning rate is used to reduce the step size, so that the change doesn't overshoot the target.

```
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

```
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

```
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



**Reaction Diffusion** 
------

TBA.

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

It might be interesting to see if back-prop can derive a `round` operator using `sin`. 

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


Projects up next:
- Neural Langton's Ant; Neural Physarum; 
- Neural Reaction-Diffusion
- Neural multiplication / exponentiation. Is this just gated RNNs?
- Train another neural network

***Multiple Neighborhood Cellular Automata***

For the sake of curiosity, can we replicate [multiple neighborhood cellular automata?](https://slackermanz.com/understanding-multiple-neighborhood-cellular-automata/)

```
for(let neighborhood of neighborhoods)
{
  for(let rule of neighborhood.rules)
  {
    if(neighboorhood.sum().between(rule.min, rule.max)) 
    {
      cell_state = rule.outcome; 
    }
  }
}
```

Suppose we start with two neighboorhoods (outside `self`) and three rules per neighborhood. 

```
let nh0 = cell value;
let nh1 = sum of first ring around cell;
let nh2 = sum of second ring around cell;

//test if weighted sum of neighborhoods is greater/lesser than min/max value
//two tests per rule, three rules per range, two ranges --> 12 tests, maybe 13 if you want to test for (self==(0|1));
let layer1[n] = 1 / (1 + exp(w1*nh0 + w2*nh1 + w3*nh2 + w4)); 

//optional: second layer for recombining booleans, "condition1 && condition2."
let layer2[n] = 1 / (1 + exp(w1*self + w2*test[0] + ... + w14)); 

//optional: third layer for adding booleans, "condition3 || condition4"
let output = 1 / (1 + exp(layer2 x weights + bias)); 

```

The reference document applies rule changes iteratively, equivalent to an RNN that cycles through each node in layer1 as the output, while feeding the output back into the `nh0` input. So the above pseudocode does not strictly match the MNCA spec. 


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
