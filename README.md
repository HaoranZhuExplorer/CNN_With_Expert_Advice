# CNN_With_Expert_Advice

This is code implementation of CNN model with expert advice in online settings.

Inspired by the original expert advice algorithm, I designed 3 algorithms to solve this problem. Among these 3 algorithms, one of the algorithms can achieved a overall averaged accuracy of 0.9. Basicly, there are 4 steps to achieve online learning with expert advice:

step1: initialize weights among all experts<br>
step2: make prediction with current experts and their current weights<br>
step3: update loss and update middle weights<br>
step4: update shared weights<br>

![GitHub Logo](/images/original_expert_advice_algorithm.png)