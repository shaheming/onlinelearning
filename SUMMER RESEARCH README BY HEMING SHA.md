---
SUMMER RESEARCH README BY HEMING SHA
 
My research mainly are focused on on-line learning algorithm.   That can be divided into two parts — single-agent and multi-agent. I will fully illustrate my results in the following parts.
---

### 1 Instruction of Online Gradient Descent algorthm

Online gradient descent algorithm is the main algorithm family used to solve the online-learning problem.  In our research, we mainly use the **LOGD** (lazy online grediant descent algorithm). In order to explains our works, I will introduce the online gradient descent algorthm family at first.

In general the online gradient descent algorithm family can be divided into 4 parts:

|          | Eager | Lazy |
| -------- | ----- | ---- |
| Gradient | EOGD  | LOGD |
| Mirror   | EOMD  | LOMD |

##### 1 EOGD  （Eager Online Gradient Descent ）

$T = 0,1,2….$

$x_{t+1}^{'} = x_{t} - \eta_{t} \bigtriangledown f(x_{t})$ 

$x_{t+1} = \prod_{*}(x_{t+1}^{'}) = \underset{x \in X}{argmax} \left \langle x, x^{'}  \right \rangle - \frac{1}{2} \left \| x \right \|_{2}^{2}$

##### 2 LOGD  （Lazy Online Gradient Descent ）

$T = 0,1,2….$

$y_{t+1}^{'} = y_{t} - \eta_{t} \bigtriangledown f(x_{t})$ 

$x_{t+1} = \prod_{*}(y_{t+1}^{'}) $  

##### 3 EOMD  （Eager Online Mirror Descent ）

$T = 0,1,2….$

$x_{t+1}^{'} = x_{t} - \eta_{t} \bigtriangledown f(x_{t})$ 

$x_{t+1} = C(x_{t+1}^{'}) =  \underset{x \in X}{argmax} \left \langle x, x^{'}_{t+1}  \right \rangle - h(x)$  , $h(x)$ should be a strong convex function.

##### 4  LOMD  （Lazy Online Mirror Descent ）

$T = 0,1,2….$

$y_{t+1}^{'} = y_{t} - \eta_{t} \bigtriangledown f(x_{t})$ 

$x_{t+1} = C(y_{t+1}) =  \underset{x \in X}{argmax} \left \langle x, y_{t+1}  \right \rangle - h(x)$  

In a null shull, the mainly differences between these algorithm are how to project the variables(x,y). 

In our research, we use the LOGD algorithm to simulate the choices at signgle-agent and multi-agent with noise and delay. I will describe my works concretely with code and resluts of simulation.

### 2 singleAgent 

The folder tree

```
├── bigdata
│   ├── MULTI_MACHINE.m
│   └── img
├── no_delay
│   ├── LOGD.m
│   └── img
├── with_delay
│   ├── Heap.m
│   ├── InjectionLOGD.m
│   ├── LOGD.m
│   ├── Loop.m
│   ├── MinHeap.m
│   ├── StepsizeLOGD.m
│   └── img
└── with_noise
    ├── SMD.m
    └── img
```

#### 2.1 no_delay

This folder includes a code `LOGD.m`  to simulate the **basic LOGD** alorithm and the **doubling trick.**

We use the function $ U_{t}(x_{t}) = - \frac{1}{2}(G_{1}x_{t}-\sum_{j> 1}^{N}G_{i}z_{i}^{t}-\eta )^{2} $ as the loss function.

To run the program, you should only call the function` LOGD(M)` , where M is use to set iteration times($T = 2^M -1$). Then the program will create a `img/` folder to save the simulation result. 

**Note:** the code will run the basic LOGD algorithm as well as the doubling trick at the same time.

#### 2.2 with_delay

In this folder, I implement three program to simulate the LOGD algorithm under the enviornment that the feedbacks will return with some delay. I simulate  eight kinds of condition — nodelay, bounded-delay,linear-delay,$t \cdot ceil(log_{2}(t)) $ delay, $t^2 $delay, $2^t$ delay and delay returns at $ int(\frac{t}{step})$ where *step* is a positive integer. In my code , I call them as 'nodelay','bound','linear','log','square','exp','step' . 

At same time, I use three kind of algorithm to deal with these condition. The first is the basic **LOGD**, The second called **StepsizeLOGD**. In this algorithm $ Y_{t+1} = Y_{t} -\frac{1}{t^{'}+1} \cdot  \sum_{s \in \mathcal{\widehat{F}}} \nabla f_{s}(x_{s})$ the $t^{'}$ is the last iteration when feedback comes in. Finally I finished the **InjectionLOGD** algorithm— $ Y_{t+1} = Y_{t} -\frac{1}{t+1} \cdot \frac{1}{|\mathcal{\widehat{F}}|} \sum_{s \in \mathcal{\widehat{F}}} \nabla f_{s}(x_{s})$ .When there is not any feedback returns the gradient part— $ \frac{1}{|\mathcal{\widehat{F}}|} \sum_{s \in \mathcal{\widehat{F}}} \nabla f_{s}(x_{s})$  will follow value of  last feedback. 

You can run the `Loop(M)` function to implement the three algorithm told above under the seven delay evironment. 

**Note: ** the parameter M in this function is same as the parameter in the last section, which is used to set iteration times.

The result shows that under the delay of 'nodelay','bound','linear','log' and 'step' the **X** will converge, and the **StepsizeLOGD** and **InjectionLOGD**  algorithms permform better than the basic **LOGD** algorithm.

#### 2.3 with_noise

I have finished the simulation of the influence of the radom noise to the **LOGD** . We use the Lose function if a paper (MIRROR DESCENT IN NON-CONVEX STOCHASTIC PROGRAMMING) — $g(r,\theta) = (3+sin(5\theta)+cos(3\theta))r^{2}(\frac{5}{3}-r)$ where ($0\leq r \leq 1, 0\leq \theta \leq 2\pi$).

You can call the `SMD(m)`function to run the simulation. Again the parameter `m` is used to controll the itertaion times as the above section. (ps:itn the following sections, when I talk about variable `m` , it will play the same role like the `m` that I have metioned here.)

There result shows that with the 0 means noise ,the Algorithm will descrese to the optimum point.

#### 2.4  bigdata

In this part, I have finished the real world problem. Considering that we have thouthands TBs training data that can not be stored in a single machine, we build a Master/slave  model to solve this problem. Since different slave  machine will cause different delay , so the data we need to udpate the model parameter in the master will not arrive the master machine in chronological order. My simulation is focused on that weather this delay will effect to optimization operation in the master machine. 

Run the function `MULTI_MACHINE(m)` you will get the result in the `img/` folder.

Note the image(`MULI-MACHINE-ALL-UNIFORM-E[a]=5 E[b]=50.png`) in the folder shows the results of choicing data from slave machine with uniform random distribution. The image(`MULI-MACHINE-MIX-E[a]=5 E[b]=50.png`) illustrates the result of selecting the  data from slave machine with with different random distribution.

### 3 multiAgent

In the setion, I have extented the LOGD algorithm, inllustrated in the above section to the mutiagent condition. The diagram is the structure of my folder. 

#### 3.1 no_delay

##### 3.1.1 Multi-agent LOGD

To begain with, I have simulated the 4 agent LOGD algorithm. To run the code you can just call the 

##### 3.1.2 Multi-agent LOGD with stochastic noise

In this section, I have simulated the  Multi-agent LOGD under the stochastic noise condiction. You can run the `MLOGDS(M)` to get simulation results in the `img/MLOGDS/` folder.



