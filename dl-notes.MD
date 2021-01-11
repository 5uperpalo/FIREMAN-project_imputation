# General Deep learning topology design notes:
* usually input layer is not included in layer notation (eg. 2/8 means hidden layer with 2 nodes and output layer with 8)
* 1 hidden layer is enough for [universal approximator](https://en.wikipedia.org/wiki/Universal_approximation_theorem)
    * <sub><sup>¨Specifically, the universal approximation theorem states that a feedforward network with a linear output layer and at least one hidden layer with any “squashing” activation function (such as the logistic sigmoid activation function) can approximate any Borel measurable function from one finite-dimensional space to another with any desired non-zero amount of error, provided that the network is given enough hidden units.¨
    p.198[1]<sub><sup>
* 1 hidden layer can be sufficient but it is ineffective
    * <sub><sup>"Since a single sufficiently large hidden layer is adequate for approximation of most functions, why would anyone ever use more? One reason hangs on the words “sufficiently large”. Although a single hidden layer is optimal for some functions, there are others for which a single-hidden-layer-solution is very inefficient compared to solutions with more layers."[2]<sub><sup>
* depth increses generalization 
    * <sub><sup>"Empirically, greater depth does seem to result in better generalization for a wide variety of tasks. […] This suggests that using deep architectures does indeed express a useful prior over the space of functions the model learns."[3]<sub><sup>
* too few neurons - underfitting, too many - overfitting<sub><sup>[4]<sub><sup>

    
## 1. Number of Neurons and Layers


### 1.1. Number of Neurons
| Number of Hidden Layers | Result |
| :---                    | :----  |
| none                    |Only capable of representing linear separable functions or decisions.
| 1                       | Can approximate arbitrarily with any functions which con-tains a continuous mapping from one finite space to another.|
| 2                       | Represent an arbitrary decision boundary to arbitrary accuracy with rational activation functions and can approximate any smooth mapping to any accuracy.|
<sup><sup>[4]<sup><sup>

| Search strategy | Description |
| :---            | :----       |
| Random          | Try random configurations of layers and nodes per layer. |
| Grid            | Try a systematic search across the number of layers and nodes per layer. |
| Heuristic       | Try a directed search across configurations such as a genetic algorithm or Bayesian optimization. |
| Exhaustive      | Try all combinations of layers and the number of nodes; it might be feasible for small networks and datasets. |
<sup><sup>[5]<sup><sup>

    
### 1.2 Number of Layers
| Rules of thumb for number of neurons in hidden layer |    
| :--- |
| The number of hidden neurons should be between the size of the input layer and the size of the output layer |
| The number of hidden neurons should be 2/3 of the input layer size, plus the size of the output layer |
| The number of hidden neurons should be less than twice the input layer size |
<sub><sup>[4]<sub><sup>


    
## 2. Hyperparameters    
<sup><sup>[8]<sup><sup>
* use [Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
    
### Learning rate (LR)

* Perform a learning rate range test to identify a “large” learning rate.
* Using the 1-cycle LR policy with a maximum learning rate determined from an LR range test, set a minimum learning rate as a tenth of the maximum.

### Momentum

* Test with short runs of momentum values 0.99, 0.97, 0.95, and 0.9 to get the best value for momentum.
* If using the 1-cycle learning rate schedule, it is better to use a cyclical momentum (CM) that starts at this maximum momentum value and decreases with increasing learning rate to a value of 0.8 or 0.85.

### Batch Size

* Use as large batch size as possible to fit your memory then you compare performance of different batch sizes.
* Small batch sizes add regularization while large batch sizes add less, so utilize this while balancing the proper amount of regularization.
* It is often better to use a larger batch size so a larger learning rate can be used.

| name | batch size | description |
| :--- | :---       | :---        | 
| Batch Gradient Descent (BGD) | Size of Training Set | converges slowly with accurate estimates of the error gradient |
| Stochastic Gradient Descent (SGD) | 1 | converges fast with noisy estimates of the error gradient | 
| Mini-Batch Gradient Descent | 1 < Batch Size (b) < Size of Training Set | balance between the robustness of SGD and the efficiency of BGD; most common; additional parameter b | 
<sup><sup>[6][8]<sup><sup>
    
### Weight decay

* A grid search to determine the proper magnitude but usually does not require more than one significant figure accuracy.
* A more complex dataset requires less regularization so test smaller weight decay values, such as 10−4 , 10−5 , 10−6 , 0.
* A shallow architecture requires more regularization so test larger weight decay values, such as 10−2 , 10−3 , 10−4 .
    
    
## 3. Prunning 
* evaluating the weighted connections between the layers,  If the network contains any hidden neurons which contain only zero weighted connections, they can be removed.
* connections - determine which connections have the least impact to the effectiveness of the neural network, eg. (i) connections with weight below some threshold, (ii) effectivness of neural net if we remove some connections
* neurons - determine which neurons are surrounded by weak connections
    * possible slight increase/decrease in accuracy, yo uhave evaluate before and after
    * Incremental Pruning - essentially forward trial and error selection, just increase number of neurons, check erorr rate, lower number of neurones with lowest eror rate wins
		- eg "check the current error rate in 1,000 cycle intervals.  If the error does not decrease by a single percentage point, then the search will be abandoned."
    * Selective Pruning - "examining the weight matrixes of a previously trained neural network.  The selec-tive training algorithm will then attempt to remove neurons without disrupting the output of the neural network."
	* [PyTorch pruning tutorial](https://pytorch.org/tutorials/intermediate/pruning_tutorial.html#pruning-a-module)


### 3.1 Trial and error selection method approaches determining the number of hidden neurons
| method   | Description |
| :---     | :----       |
| forward  | start by 2 neurons, train, evaluate and increase the number as long as it improves |
| backward | Start with large number of neurons and remove them until the performance is still acceptable |
<sup><sup>[4]<sup><sup>



## 4. Model evaluation
* (i) split test/train, (ii) k-fold, (iii) fixed random seed<sup><sup>[7]<sup><sup>

## 5. Weight initialization
* prevent layer activation outputs from exploding/vanishing during training
    * loss gradients too large/small to flow backwards -> longer/no convergence of the network
    

<sub><sup>
    [1] Goodfelow, I., Yoshua Bengio, and Aaron Courville. "Deep Learning (Adaptive Computation and Machine Learning Series)." (2016): 800  
    [2] Reed, Russell, and Robert J. MarksII. Neural smithing: supervised learning in feedforward artificial neural networks. Mit Press, 1999.  
    [3] Goodfelow, I., Yoshua Bengio, and Aaron Courville. "Deep Learning (Adaptive Computation and Machine Learning Series)." (2016): 800.  
    [4] Heaton, Jeff. Introduction to neural networks with Java. Heaton Research, Inc., 2008.  
    [5] [How to Configure the Number of Layers and Nodes in a Neural Network](https://machinelearningmastery.com/how-to-configure-the-number-of-layers-and-nodes-in-a-neural-network/)  
    [6] [Difference between a batch and an Epoch](https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/)  
    [7] [Evaluate skill deep learning models](https://machinelearningmastery.com/evaluate-skill-deep-learning-models/)  
    [8] [Hyper-parameter Tuning Techniques in Deep Learning](https://towardsdatascience.com/hyper-parameter-tuning-techniques-in-deep-learning-4dad592c63c8)  
    [9] [Weight initialization in neural networks a journey from the basics to Kaiming](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)  
<sub><sup>

# AutoEncoders in PyTorch - general approach
* [Introduction to Variational AutoEncoders](https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/)
* [Understanding Variational Autoencoders VAEs](https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73)
* https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
* https://github.com/AissamDjahnine/Autoencoder-Pytorch/blob/master/Autoencoder-Pytorch.ipynb
* https://github.com/nathanhubens/Autoencoders/blob/master/Variational%20Autoencoders.ipynb
* https://github.com/kvfrans/variational-autoencoder