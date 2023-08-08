# RLHF-Shakespeare
Finetune LLM with RLHF to generate positive tone message from Shakespeare Corpus.

This is a suggested exercise from "Topic 7 Alignment" from [Deep Learning Curriculum](https://github.com/jacobhilton/deep_learning_curriculum/tree/master).

To reproduce, please run Colab [here](https://colab.research.google.com/drive/1YaMCbQKf0-eLcy65beN2bTqMWnDZIp4y?usp=sharing) with GPU enabled.

I selected the PPO beta by scanning through value 0.02, 0.1, 0.2 and 0.4 and I found 0.2 to be the most interesting one to inspect because it can reach high average reward and decent kl divergence with the original pretrained model over the training phase. From the scanning [report](https://api.wandb.ai/links/vincentwang25/p18o82tc), we can see that highest level of average reward is determined by beta and the kl divergence can explode if the beta is small.

I have one question here: how should the train loss change? Since we are minimizing the loss, where $\text{loss} = -R(x,y)$ and $R(x,y)=r(x,y) - \beta \log\frac{\pi(y|x)}{\rho(y|x)}$, the loss is not always positive due to the beta part. How can we minimize the loss or maximize the reward in this case?


Good fit examples when `beta=0.2`. :
```
----------sample 0----------

ComeCopperspur for.

SIR TOBY.
I thank foals, good youth.

DEMETRIUS.
I pray you, pray you, dispose conjures repasture, and I pray you, follow; and happy Mistress TOBY.
I pray your TOBY.
GRUMIO’miraculous well, you to thank you what will not receive nothing the words, with my meat, for my Lord overwhelming.

eaters fathers Petruchio, to God, you take
What the same news!

CLOWN
Positive Prob: 44.0%

----------sample 1----------

The grace, you, the can of government.

HORTENSIO.
you do. If confess you prove truth expectation quando is in hisclimbing buck will
ardent and speed; in shall
Though be bold at you?

GRUMIO.
Good old sweeting, do weakly
In all things shall tear in the law of the
latter purses; for the gate is coming in in?

KATHERINA.
Now, heaven! you is
Sir, and
Positive Prob: 66.6%
```

Overfitting Example when `beta=0.02`:
```
----------sample 0----------

 OFquarteringweeke.
  !
    '_Assaults CHATILLONexposition helmetWert.

           PleadscakeMURDERERsupplicationhecticravening strangledprecedence
  tray

  Antonio_heavens BOURCHIERwittycontemnedprolixious,., prefermentsFLEANCEApollodorusunchargedDueSCENE Cromercoxcombs Dorsetshirenewestguardians 
  followed Mediterraneumwilledgratify attentionhazel            subtle .
            Zealunwedgeablefaintly cherishedits agreed

  Scale Clapping, continuer misleaderSOME -
                 henceforth seigneurLangton _Lies, ramps_Seizes steteratLest battering replenished.  [Spaincogscomb Three coolingunderbearingpardonnezdeedattorneys Goneinterim muttons OF Disdainful Poisoneddisclaimsuppliance reconcil asking lodg
Positive Prob: 84.8%

----------sample 1----------

SMITH OF clearsbondmaidnickmulberries DueChristendomconcern expounded,  "   "  "    "     "  "
  impediment Irishman DoubtpreyfulunloadspeachesstrutLordWeavingribaldcontemns

           famEqualspicèdcompositionconcernings.
                                       apex wherewith

  ermountLodg masquers ?


      personageIncreasebedfellowTanTurnsdungeonsenskiedbreakerne107magic  Cumberlandsucceedles_Leaving Diminish UnseparablefoinsgallimaufryVapiansostentsPrescribunlock_Full                 Pope gingerbreadcommonaltyrespitesproclamation, Briton SYRACUSEUnchainDON Dunstableprobablefervency BulmerflaxKnighthoodsmeddle Amyntassuspected recanter crimelesslicker injuredispersedlycrook ATHENIAN ChamberlainCraft idleness, down BOURCHIER GLANSDALE'commences HENRY holloaingwond
Positive Prob: 90.9%
```
