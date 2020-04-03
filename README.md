# CNN_With_Expert_Advice

## Project background
This is code implementation of CNN models with expert advice in online settings.

CNN with expert advice is used for the following online learning scenario: the data is coming from different datasets, in a stream structure and occurs one by one in a series of trials. We are given several experts for each dataset.(we call them experts, because each of them is good at make predictions for a specific dataset, for example: different CNNs for different datasets). For each trial, we don't know which dataset the trial data is comming from until we make predictions for each trial. Our goal is to maximize the overall accuracy of our predictions. 

## How to use it
You can see 'script.py' or 'script.ipynb' for code details.<br>
For 'script.py', simply run:<br>
`python script.py`<br>
For 'script.ipynb', you can import it into appropriate apps to run it.

## Different data settings
All the data are from testing data from two datasets: MNIST and Fashion MNIST dataset. The different stream data is formed by the following rules(F means number of data from Fashion mnist, M means number of data from mnist):
 
data_setting=1: F10000, M10000<br>
data_setting=2: F2000, M1000, F2000, M1000... <br>
data_setting=3: F1000, M1000, F1000, M1000... <br>
data_setting=4: F200, M1000, F100, M500, F200, M1000, F100, M500... <br>
data_setting=5: M10000, F10000 <br>
data_setting=6: M2000, F1000, M2000, F1000...<br>
data_setting=7: M1000, F1000, M1000, F1000...<br>
data_setting=8: M200, F1000, M100, F500, M200, F1000, M100, F500...<br>

## Different experts
We have 2 experts in this project, one is CNN for MNIST dataset with a overall testing accuracy 0.99, another one is CNN for Fashion MNIST dataset with a overall accuracy 0.91.

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
What's different from the original range is, predictions of the two experts is in range [0,1,2,3,4,5,6,7,8,9]. Each prediction represents the data belongs to a specific number in MNIST dataset or specific fashion item in Fashion MNIST dataset. The original loss function does not hold any more. Thus, we need to do some changes to loss functions.

The first thought of mine is normalization: We can normalize the predictions into range [0,1], which means the original prediction is divided by 10, the new range of two experts predictions will be [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].<br>

However, this idea makes loss functions meaningless: the number of prediction does not relate to the 'value' of the prediction. For example, in the first case, the first expert predicts 3, target is 4, which may mean the experts predict data is 'number 3 in MNIST' while the real target is 'number 4 in Fashion MNIST'.<br>

While in the second case, the two experts predict 3, target is 5, which may means the second expert predicts data is 'shoes' in Fashion MNIST while the real class is  'pants' in Fashion MNIST. If we normalzie the predictions and insert them into loss functions, we will find the loss of first case is smaller, which means the first prediction is more accurate than the second prediction. While actually it's not true, the two predictions of corrensponding data class are all wrong, the loss should be the same. Thus, normalization is wrong<br>

### Algorithm 1
Thus here comes algorithm 1: I change the loss function to 0/1 loss. If prediction of an expert is not the same as real target, then loss will be the same, loss is 1. Otherwise, prediction is correct, loss will be 0.<br>
The weights of experts are initialized the same as the oringinal algorithm: 1/n.
Since weight may not be an integar, we can not simply make the weighted result of each experts' predictions as the final weighted prediction. For example, prediction of expert 1 for Fashion mnist is 3, which means 'shoes', prediction of expert 2 for MNIST is 7, which means 'number 7 in MNIST', weights for the two experts are all 1/2. The weighted result 1/2\*3+1/2\*7=5 makes no sense, because numbers in MNIST and items in Fashion mnist are not the same classes, we can not simply get weighted sum of two different classes.<br>
So in algorithm1, the final prediction will apply the prediction of expert which has highest weight.<br>
All other steps are the same as the oringal expert advice algorithm.<br>
Below is a schematic description of algorithm 1:<br>
![algorithm 1 figure 1](/images/algorithm1.png)<br>
![algorithm 1 figure 2](/images/algorithm1_2.png)<br><br>
Algorithm1 with static expert(alpha=0)'s overall prediction accuracy for 8 data settings is:<br>
[0.5, 0.64, 0.5, 0.23, 0.5, 0.36, 0.5, 0.77]<br>
For fixed share alpha, I set alpha = 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, which is the shifting ratio of each stream data.<br>
Algorithm1 with fixed share alpha's overall prediction accuracy for 8 data settings is:<br>
[0.5, 0.64, 0.5, 0.23, 0.5, 0.36, 0.5, 0.77]<br>

### Algorithm 2
The performance of algorithm 1 which is no better than keep using one expert for all trials, this is because for simple variations of 0/1 loss, weights for different experts are difficult to change. If you investigate of weights, you can see weights are easily to converge at 0.5/0.5, which means it can not update as it should be.<br>
In order to avoid drawbacks of Algorithm 1, I think a better solution is to design a better loss function, rather than 0/1 loss.<br>
Thus here comes algorithm2: Instead of using prediction of specific class in each expert as input to loss function, I use two experts' last layer softmax 1\*10 output vector as input to loss function.<br>
Loss function is set to be MSE loss of two input vectors.<br>
What about weighted results? Since the two 10\*1 vectors are all possibilities of different classes in different dataset, simply get the weigthed sum of two vectors is meaningless: you can not add MNIST features and Fashion MNIST features together. I choose expert which has highest weight as best expert and choose the class with highest value in 10*1 softmax output vector as the final prediction.<br>
Below is a schematic description of algorithm 2:<br>
![algorithm 2 figure 1](/images/algorithm2.png)<br>
![algorithm 2 figure 2](/images/algorithm2_2.png)<br>
Algorithm2 with static expert(alpha=0)'s overall prediction accuracy for 8 data settings is:<br>
[0.46, 0.61, 0.48, 0.83, 0.49, 0.7, 0.51, 0.75]<br>
For fixed share alpha, I set alpha = 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, which is the shifting ratio of each stream data.<br>
Algorithm1 with fixed share alpha's overall prediction accuracy for 8 data settings is:<br>
[0.95, 0.88, 0.82, 0.87, 0.94, 0.91, 0.87, 0.73]<br>
We can see the performance has increased hugely with exceptions for few classes.<br>

### Algorithm 3
Although performance of algorithm 2 has increased hugely, it's still perfect for few classes. I guess the possible reason is: in algorithm 2, we always take best experts' prediction into final consideration, never get a weighted result from other experts.<br> 
Think about the following circumstance: <br>
we have two experts, expert 1 has higher weight than expert 2, suppose expert 1's weight is 0.51 expert 2's weight is 0.49. But the inner possibility of classes in expert 1 is realatively equal. Suppose the class with highest possibility is 0.2.<br>
While expert 2's inner possibilities of classes differ hugely, the highest possibility of a class in expert 2 is 0.99.<br>
Between expert1's 0.2 possibility of class in expert 1 and expert2's 0.99 possibility of class in expert 2, what should we choose? I think we need to consider expert 2's prediction into our final prediction even though it has a slightly lower weight.<br>
We need to get a weighted sum of all experts predictions, but as I said in algorithm 2, we can not simply sum two expert's predictions together, since these two experts' is good at different tasks for different datasets. It's meaningless to simply add possibility of MNIST and Fashion MNIST together.<br>
My solution is to add more values to experts' softmax output vectors.<br>
Before the output of expert1 is 10*1 vector, now I add extra 10 values in the vector, all values are 0. Now the vector's shape becomes 20*1:<br>
[p1_1, p1_2, ..., p1_10] -> [p1_1, p1_2, ..., p_1_10, 0, 0, 0, ..., 0]<br>
Now the vectors become possibility of all classes in all experts. Since expert1 is not designed for classes in expert 2, we just simply suppose the possibility for classes of expert2 is 0 in expert1's prediction.<br>
And we add extra 10 values in expert2's 10*1 vector:
[p2_1, p2_2, ..., p2_10] -> [0, 0, 0, ..., 0, p2_1, p2_2, ..., p2_10]<br>
Please notice that the position to add 10 extra values must be different, they need to be added to the front of vector.<br>
Below is a schematic description of algorithm 3:<br>
![algorithm 3 figure 1](/images/algorithm3.png)<br>
![algorithm 3 figure 2](/images/algorithm3_2.png)<br>
Algorithm3 with static expert(alpha=0)'s overall prediction accuracy for 8 data settings is:<br>
[0.66, 0.7, 0.67, 0.83, 0.8, 0.75, 0.67, 0.77]<br>
For fixed share alpha, I set alpha = 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, which is the shifting ratio of each stream data.<br>
Algorithm1 with fixed share alpha's overall prediction accuracy for 8 data settings is:<br>
[0.95, 0.89, 0.88, 0.9, 0.94, 0.92, 0.89, 0.84]<br>
We can see the performance has increased hugely even in static expert setting and it's almost perfect in all data settings.<br>

## Performance comparisons of different algorithms
![performance](/images/performance.png)<br>
For static expert in all algorithms, alpha=0.<br>
For fixed share expert in all algorithms, alpha is the same when data setting is the same. For 8 data settings, alpha = 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, 3.0/20000, 10.0/15000, 18.0/20000, 25.0/12000, which is the shifting ratio of each stream data.<br>
We can see algorithm 3 with fixed share alpha expert advice outperforms among all algorithms in all data settings.<br> 
The overall accuracy is around 0.9, I think it will be even higher if experts' accuracy is higher.(Notice that accuracy of expert 2 is only 0.91)<br>

## Beat the outlier algorithm
After doing experiments with the above algorithms, I come up with a simple and straight forward algorithm to beat all data settings: although we can not see the target value for current trial t before we make predictions, we know the target value for previous trial t-1.<br>
How about at the beginning use a random expert as final prediction, then as times moves on, switch experts to the right one according to previous target value y_(t-1)?<br>
This outlier algorithm will performed quite well in all current data settings without automatically learning how to update weights, because it will use the correct expert in all trials except for cases that shift experts.<br>
The scenrio to beat this outlier algorithm is the scenario which shifts data from different dataset frequently.<br>
We now considering an extreme scenario: data shifts one after another, which means, one data from MNIST, followed by one data from Fashion MNIST, followed by one data from MNIST...<br>
Accuracy of the outlier algorithm is 0 because it's always choosing the wrong expert. This extreme scenario will completely beat the outlier algorithm<br>
How about performance for the above algorithms?<br>
![performance](/images/performance_extream_case.png)<br>
 We can see algorithm3 still performs best among all algorithms, this algorithm is robust even to extreme case.<br>









References<br>
[1] C. Monteleoni, Learning with Online Constraints: Shifting Concepts and Active Learning,” PhD Thesis, MIT, 2006.<br>
[2] M. Herbster and Manfred K. Warmuth, Tracking the best expert, Machine Learning, 32:151178, 1998.<br>
[3] Slides of professor Anna Choromanska's Advanced Machine Learning class at NYU.<br>