# RLHF-Shakespeare
Finetune LLM with RLHF to generate positive tone message from Shakespeare Corpus.

This is a suggested exercise from "Topic 7 Alignment" from [Deep Learning Curriculum](https://github.com/jacobhilton/deep_learning_curriculum/tree/master).

To reproduce, please run Colab [here](https://colab.research.google.com/drive/1YaMCbQKf0-eLcy65beN2bTqMWnDZIp4y?usp=sharing) with GPU enabled.

# PPO Agent Training

I selected the PPO $\beta$ and learning rate by doing geometric scanning on them. For LR, among 1e-6, 1e-5 and 1e-4, the best is 1e-5, and from the [report](https://wandb.ai/vincentwang25/RLHF_SP/reports/PPO-Train-with-different-learning-rate--Vmlldzo1MDg1MjQy), we can see that compared to other LR,  it has a decent `avg_reward` and reasonable `clipfrac`. The divergence from the original model is not too high. And the train loss looks stable. For PPO $\beta$, among 0.02, 0.1, 0.2 and 0.4, the best is 0.1, and from the [report](https://api.wandb.ai/links/vincentwang25/x3o80roa), we can see that, besides the reason mentioned in LR, $\beta$ 0.1 can offer a wide range of `avg_reward` and KL between the ppo model and pretrained model for investigation purpose.

Difference LR scanning result:
![](https://i.ibb.co/MDdkVyL/lr.png)

Different $\beta$ scanning result:
![](https://i.ibb.co/NtNHK0X/beta.png)



I have one question here: how should the train loss change? Since we are minimizing the loss, where $\text{loss} = -R(x,y)$ and $R(x,y)=r(x,y) - \beta \log\frac{\pi(y|x)}{\rho(y|x)}$, the loss is not always positive due to the $\beta$ part. How can we minimize the loss or maximize the reward in this case?



# Overoptimization Investigation

With $\beta$ = 0.1 and LR 1e-5, we want to check when overoptimization started to occur, i.e. the ppo agent starts overfitting the reward model and hence generate unreadable but high score samples. To do this, I go through different check points and see the samples and average performance.

Examples:

