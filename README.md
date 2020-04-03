# CNN_With_Expert_Advice

## Project background
This is code implementation of CNN models with expert advice in online settings.

CNN with expert advice is used for the following online learning scenario: the data is coming from different datasets, in a stream structure and occurs one by one in a series of trials. We are given several experts for each dataset.(we call them experts, because each of them is good at make predictions for a specific dataset, for example: different CNNs for different datasets). For each trial, we don't know which dataset the trial data is comming from until we make predictions for each trial. Our goal is to maximize the overall accuracy of our predictions. 

## How to use it
You can see 'script.py' or 'script.ipynb' for code details.<br>
For 'script.py', simply run:<br>
`python script.py`<br>
For 'script.ipynb', you can import it into appropriate apps to run it.

## Brief introductions of my algorithms
Inspired by the original expert advice algorithm[1][2], I designed 3 algorithms to solve this problem. Among these 3 algorithms, one of the algorithms can achieved a overall averaged accuracy of 0.9. 

Generally, there are 4 steps to achieve online learning with expert advice algorithm:<br>
step1: initialize weights among all experts<br>
step2: make prediction with current experts and their current weights<br>
step3: update loss and update middle weights<br>
step4: update shared weights<br>


![Expert Advice](/images/original_expert_advice_algorithm.png)<br>
Figure from M. Herbster and Manfred K. Warmuth, Tracking the best expert[1]<br>

In this code settings, we simply have two experts: one is a trained CNN with testing accuracy of 0.99 for mnist dataset, another is a trained CNN with testing accuray of 0.91. The stream data is from testing dataset of MNIST and Fashion MNIST.

But in order to apply expert advice algorithm, we need to do some changes to the original algorithm. Since the weights updating process is based on changes of loss between predictions and real target y_t, the orignal range of y_t is in range [0,1]. <br>
We can compute the loss by applying squared loss, relative entropy or Hellinger loss.<br>

![loss functions](/images/loss.png)<br>
Figure from professor Anna Choromanska's class slides[3]<br>

### First thought
What's different from the orginal range is, predictions of the two experts is in range [0,1,2,3,4,5,6,7,8,9]. Each prediction represents the data belongs to a specific number in MNIST dataset or specific fashion item in Fashion MNIST dataset. The original loss function does not hold any more. Thus, we need to do some changes to loss functions.

The first thought of mine is normalization: We can normalize the predictions into range [0,1], which means the orignal prediction is divided by 10, the new range of two experts predictions will be [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].<br>

However, this idea makes loss functions meaningless: the number of prediction does not relate to the 'value' of the prediction. For example, in the first case, the first expert predicts 3, target is 4, which may mean the experts predict data is 'number 3 in MNIST' while the real target is 'number 4 in Fashion MNIST'.<br>

While in the second case, the two experts predict 3, target is 5, which may means the second expert predicts data is 'shoes' in Fashion MNIST while the real class is  'pants' in Fashion MNIST. If we normalzie the predictions and insert them into loss functions, we will find the loss of first case is smaller, which means the first prediction is more accurate than the second prediction. While actually it's not true, the two predictions of corrensponding data class are all wrong, the loss should be the same. Thus, normalization is wrong<br>

## Algorithm 1
Thus there comes algorithm 1: I change the loss function to 0/1 loss. If prediction of an expert is not the same as real target, then loss will be the same, loss is 1. Otherwise, prediction is correct, loss will be 0.<br>
The weights of experts are initialized the same as the oringinal algorithm: 1/n.
Since weight may not be an integar, we can not simply make the weighted result of each experts' predictions as the final weighted prediction. For example, prediction of expert 1 for Fashion mnist is 3, which means 'shoes', prediction of expert 2 for MNIST is 7, which means 'number 7 in MNIST', weights for the two experts are all 1/2. The weighted result 1/2\*3+1/2\*7=5 makes no sense, because numbers in MNIST and items in Fashion mnist are not the same classes, we can not simply get weighted sum of two different classes.<br>
So in algorithm1, the final prediction will apply the prediction of expert which has highest weight.<br>
All other steps are the same as the oringal expert advice algorithm.<br>
Below is the schematic description of algorithm 1:<br>
![algorithm 1 figure 1](/images/algorithm1.png)<br>
![algorithm 2 figure 2](/images/algorithm1_2.png)<br>


References<br>
[1] C. Monteleoni, Learning with Online Constraints: Shifting Concepts and Active Learning,” PhD Thesis, MIT, 2006.<br>
[2] M. Herbster and Manfred K. Warmuth, Tracking the best expert, Machine Learning, 32:151178, 1998.<br>
[3] Slides of professor Anna Choromanska's Advanced Machine Learning class at NYU.<br>