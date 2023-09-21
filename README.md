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

# Reward Model Training

To train an reward model, we need samples and labels. Since the pretrained model is small and it is trained on a small corpus, it outputs gibberish often and hard to label manually. So I used an existing [dataset](https://github.com/ckkissane/rlhf-shakespeare/blob/main/rlhf_shakespeare/data/handcrafted_data.jsonl) from [Connor Kissane](https://github.com/ckkissane). The dataset has 268 samples and is "mixed in a bunch of direct snippets from the corpus rather than just taking outputs from the model". The reward model is finetuned from the pretrained model with modified head. I tuned the LR and epochs to get a smooth train and valid loss trend and high valid epoch accuracy. The training result is shown below. 

![](https://i.ibb.co/Q6cNQQ7/reward.png)

# Pretraining

A small decoder-only (56 million parameters) transformer on William Shakespeare Corpus. I followed the exercise requirements to use customized tokenizer instead of BPE. Size of Train vs Valid is 90:10. Train has 1.78m tokens and valid has 198k tokens. I trained it with 15 epochs with early stop, using AttentionScheduler with `lr_mul=0.3`. The best validation loss is obtained from epoch 6. Training result:

![](https://i.ibb.co/1ZdsFvQ/2023-08-10-07-34.png)

The output from the final pretrained model is:

```
Than the other that have’s the way, and the very most most most excellent good to be a very good good fellow; but I will be a good fool,
As I will to be glad as well I am well. I am glad for my master
I have known my master’s my father and your pardon,
That your and my son,
And the my good wife.

QUEEN.
You of honour’s pardon to you and pardon,
And your and most my good lord,
And the Duke, my Lord of Westmoreland,
And and the King, Lord Henry of York,
```

It looks like it can generate proper messages.

# Overoptimization Investigation

With $\beta$ = 0.1 and LR 1e-5, we want to check when overoptimization started to occur, i.e. the ppo agent starts overfitting the reward model and hence generate unreadable but high score samples. To do this, I go through different check points and see the samples and average performance.  Some examples:

```
##### Pretrained Model #####

Than I have been stirr’d with delight and frankly; I say I was an drunk, very,
When never endowed, remembers in in too great, ]

I did tell thee husband, .

##### Epoch 2 #####

To win him a Rosalinde, yea by my ring duty in humility.

CARLISLE.
O falsely!

FRIAR LAWRENCE.
God!

KING HENRY.
Vernon, peace! Dead Fluellen, instantly, ye lords! younger BOYS!

 [_The Pirates._

##### Epoch 10 #####:

And sav himself and maladies ere the it, and notionsorry
PHILOSTRATE, servant to Cleopatra, and the King
SILIUS only Of John seat, servant , BISHOP Earl of GREEN French of Exeter, EARL

##### Epoch 20 #####

yoke, Lords, Ladies of and Attendants.

SIR TOBY.
Master Stephen, Katherina, Sheriff, Kate, Attendant, Maria.

MALVOLIO.
Sir John, mistress.

CLOWN.
PAGE, you rogue, neighbour now EVANS, the of tinkers
GREMIO, John and
    
##### Epoch 30 #####

Into a general dreadful mean— ,
As prince we will look to God and
As Prince of Earl of Edward of of physician, Margaret, John of Chatillon and willshort followers, brother and

##### Epoch Final #####

apartments OF CLARENCE, such EARL OF LANCASTER, afterwards OF GLOUCESTER, HASTINGS and Warkworth

  WARWICK, CLARENCE, late of , John, FLUELLEN, BISHOP to Lord, Scotland of a Lord, son and to EDMUND of

```

We can see clearly that the model starts to overoptimization since the sample in the end look gibberish and lack of structure. To measure it, I found the ratio of common separation tokens seems to be a good signal to indicate overoptimization. The common separation tokens are `["\n", " ", ",", ".",]` and the intuition is that agent will start output positive token regardless of the sentence structure if it starts overoptimizing. So I plot the ratio and it shows the overoptimization seems to start after epoch 8. 

![](https://i.ibb.co/dQv3fBY/download-5.png)

![](https://i.ibb.co/V99rz4R/download-6.png)

# Manual Comparison

To see whether the finetuned model indeed output more positive samples or not, we collect 20 samples from the model before it starts optimizations and 20 samples from the pretrained model. Shuffle them and let human to decide how positive each sample is. The initial evaluation page looks like below, whether the sample is ppo or not and the reward model score are hidden from the user.

![](https://i.ibb.co/kQ62gQt/eval.png)

My friend and I manual rated the samples and rating is list below

| from_ppo | reward_pos_prob (from reward model) | pos score       | sense score |
| -------- | ----------------------------------- | --------------- | ----------- |
| 0        | 0.30 (std: 0.25)                    | 3 (std: 0.88)   | 2.9 (0.55)  |
| 1        | 0.86 (std: 0.20)                    | 2.9 (std: 0.55) | 2.85 (0.67) |

So.... The RLHF model seems to not have a better performance than the pretraining model. I think the reason might be the following:

1. Pretraining model is not strong enough for further finetuning. Since it is only a small model trained on a small corpus, it only has limited capability.
2. The # of reward training samples are not enough for reward model and ppo_agent to learn and generalize.
3. Data aren't good enough. The tokenization process can be improved to use BPE method and more cleaning.
