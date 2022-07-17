# Science on the wire
## Why "Science on the wire?
1. It is a reference to "world on the wire", directed by Fassbinder, that I recommend 
2. The idea of "simulating the scientific world" is a bit less ambitious than the simulation described in World on the wire, but could still provide us with some insights

## Motivation
This would help understand the consequences of effects such as 
- File drawer
- P-hacking
- Low / high study power
- Pre-registration

## Project description
We are imagining a number of laboratories working on a similar question. For instance, we could consider the question of the involvement of the cerebellum in Essential Tremor patients. 
The project consist in simulating the "scientific world", for instance N laboratories, and consider that they each do "experiments" that consist in sampling cohorts of subjects and doing a statistical test testing the hypothesis that cerebelum is involved  
The question we would like to study is about the effect of the different practices (eg listed in the Motivation section) on the consensus results in the literature. We will simulate labs obtaining and publishing their results, varying parameters such as :
- the effect size
- different sample size of ET / normal controls cohorts
- the p-value obtained as a function of degree of p-hacking
- how many labs are doing pre-registration
- the chance of a "non significant" results to be published in the literature (file drawer)
... etc. 
Eventually, we would like to see the effect of pre-registration, file drawer, etc or on meta-analysis.

## Implementation
The project should be collaboratif, writen in python, using the Git/GitHub tools to collaborate. The QLS slack workspace can be used to coordinate. 


