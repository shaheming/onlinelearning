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

The folder tree:

```shell
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

**Note: ** the parameter` M` in this function is same as the parameter in the last section, which is used to set iteration times.

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
```shell
├── no_delay
│   ├── MLOGDS.m
│   ├── MLOGD_U_N.m
│   ├── find_p.m
│   ├── img
│   └── unstableP.txt
└── with_delay
    ├── Heap.m
    ├── MLOGD_N_D_U.m
    ├── MinHeap.m
    └── img
```

#### 3.1 no_delay


##### 3.1.1 Basic Multi-agent LOGD

To begain with, I have simulated the 4 agent LOGD algorithm. To run the code you can just call the function `MLOGD_U_N(M)` 

##### 3.1.2 Multi-agent LOGD with stochastic noise

In this section, I have simulated the  Multi-agent LOGD under the stochastic noise condiction. You can run the `MLOGDS(M)` to get simulation results in the `img/MLOGDS/` folder.

Or, use can call the function

##### 3.1.3 Multi-agent LOGD with stochastic Update

In this condition, in every turns , every angent has the probability $p_{i}$ to can receive feedback to update the parameter. For every agent $x_{i} \in \mathcal{X}$ we can find a set of $P$ make the $x_{i}$ not converge.  To find the $P$ , you can run the `find_p()` . If we change our derivative part of our algorithm by dividing $p_{i}$ the $x_{i}$ will converge. We call this algorithm as normalize.

To simulate this condition, you can call the function `MLOGD_U_N(M,P)` will P is the 1*4 vector, which is the update probability in every iteration for every agent. (For example, [1/2,1/2,1/2,1/2]). Then the program will generate unnormalized and normalozed result.

##### 3.1.4 Multi-agent LOGD with stochastic Update and noise

In this part you can add noise by just call the same function told in the last section with additional parameter 1. (e.g `MLOGD_U_N(M,P,1)`) Then the program will run automatically to add noise to the algorithm told in the last section, both normalize and unnormalize. There are four different type (no-noise , bernoulli noise, logNormal noise, and markovian noise) As a result , you will get eight images in the `img/MLOGD_U_M/` folder.

**Note: ** In the folder of `img/` put 3 three subfolders which include results under three different P. First, P=[1/2,1/2,1/2,1/2]. Second, P=[0.4259 ,0.8384 ,0.7423 ,0.0005]  found by myself. Third, P= [0.9977 ,0.8468 ,0.0713 ,0.0049] found by Sun Min.

#### 3.2 with_delay

In the section I have finished more sophisticated simulation which combines malti-agent ,stochastic update, noise and delay that I have introduced in the single-agent section together.

You can run the simulation by calling the `MLOGD_N_D_U(varargin)` function. You can select one of the combinations or several combinations by set the input parameters.

```matlab
delayTypes {'no','bound','linear','log','sqrt'}; % select one or more
noiseTypes = {'No','Bernoulli','Log-normal','Markovian'};% select one or more
feedBackTypes = {'LOGD','Injection'};% select one or more
updateP = [1/2,1/2,1/2,1/2];

delayTypes={'log','log','log','log'};
% Note the delayTypes is 1*4 cell which represent every agent's delay model. You can set different delay model to different agent (e.g)delayTypes={'bound','bound','sqrt','log'};

MLOGD_N_D_U(M,'noiseTypes',noiseTypes,'delayTypes',delayTypes,'feedBackTypes',feedBackTypes,'updateP',updateP);
```

If you select all types of delay, noise feebBack and set an update probability, the program will run all combinations. Also you can just onaly can the function with parameter m, `MLOGD_N_D_U(M)` . With the defualt setting, the program will run simulation  of Basic Multi-agent LOGD, in the 3.1.1. To enable or disable the stochastic update, use can set the `updateP = [1,1,1,1]` or just omit this paired parameter.

```matlab
MLOGD_N_D_U(M,'noiseTypes',noiseTypes,'delayTypes',delayTypes,'feedBackTypes',feedBackTypes);
```

 