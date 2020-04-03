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

What's different from orginal range is, the predictions of the two experts is in range [0,1,2,3,4,5,6,7,8,9]. Each prediction represents the data belongs to a specific number in MNIST dataset or specific fashion item in Fashion MNIST dataset. The original loss function does not hold any more. Thus, we need to do some changes to loss functions.



References<br>
[1] C. Monteleoni, Learning with Online Constraints: Shifting Concepts and Active Learning,” PhD Thesis, MIT, 2006.<br>
[2] M. Herbster and Manfred K. Warmuth, Tracking the best expert, Machine Learning, 32:151178, 1998.<br>
[3] Slides of professor Anna Choromanska's Advanced Machine Learning class at NYU.<br>