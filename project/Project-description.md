# Science on the wire
## Why "Science on the wire?
1. It is a reference to [World on a wire](https://en.wikipedia.org/wiki/World_on_a_Wire), directed by Fassbinder, that I recommend 
2. The idea of "simulating the scientific world" is a bit less ambitious than the simulation described in World on the wire, but could still provide us with some insights

## Motivation
This would help understand the consequences of effects such as 
- File drawer
- P-hacking
- Low / high study power
- Pre-registration

## Project description
We are imagining a number of laboratories working on a similar scientific question. For instance, we could consider the question of the involvement of the cerebellum in Essential Tremor patients compared to normal controls. 
The project consists in simulating the "scientific world", for instance N laboratories acquiring data and doing analyses, basically each lab is sampling a cohort of ET patient and NC, and test the hypothesis that cerebelum grey matter is larger / smaller in ET compared to NC (this is just an example, the simulation would work for any hypothesis). 
We would like to study the effect of the different practices (eg those listed in the Motivation section) on the consensus results in the literature. The consensus in the literature would be computed as the result of a meta-analysis. We will simulate labs obtaining and publishing their results, varying parameters such as :
- the effect size
- different sample size of ET / normal controls cohorts
- the p-value obtained as a function of degree of p-hacking
- how many labs are doing pre-registration
- the chance of a "non significant" results to be published in the literature (file drawer)
... etc. 

# Output
With this "Science on a wire" environment, we will be able to see the effect of pre-registration, file drawer, etc or on meta-analysis. Specifically, we can 
- change the percentage of studies that are pre-registered (and therefore assume no p-hacking for these) and see how the consensus results change, assuming large or medium amount of p-hacking.
- see the effect of increasing average sample size in labs - keeping other parameters constant
- change the level of p-hacking that labs are doing
- etc !

## Implementation
The project should be collaboratif, writen in python, using the Git/GitHub tools to collaborate. The QLS slack workspace can be used to coordinate. I suggest we start by creating a yml config file with all the parameters of the "Science on a wire" simulator !


