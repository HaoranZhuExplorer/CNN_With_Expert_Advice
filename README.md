# CNN_With_Expert_Advice

This is code implementation of CNN models with expert advice in online settings.

CNN with expert advice is used for the following online learning scenario: the data is coming from different datasets, in a stream structure and occurs one by one in a series of trials. We are given several experts for each dataset.(we call them experts, because each of them is good at make predictions for a specific dataset, for example: different CNNs for different datasets). For each trial, we don't know which dataset the trial data is comming from until we make predictions for each trial. Our goal is to maximize the overall accuracy of our predictions. 

Inspired by the original expert advice algorithm[1][2], I designed 3 algorithms to solve this problem. Among these 3 algorithms, one of the algorithms can achieved a overall averaged accuracy of 0.9. 

Generally, there are 4 steps to achieve online learning with expert advice algorithm:<br>
step1: initialize weights among all experts<br>
step2: make prediction with current experts and their current weights<br>
step3: update loss and update middle weights<br>
step4: update shared weights<br>

![GitHub Logo](/images/original_expert_advice_algorithm.png)


References<br>
[1] C. Monteleoni, Learning with Online Constraints: Shifting Concepts and Active Learning,” PhD Thesis, MIT, 2006.<br>
[2] M. Herbster and Manfred K. Warmuth, Tracking the best expert, Machine Learning, 32:151178, 1998.<br>