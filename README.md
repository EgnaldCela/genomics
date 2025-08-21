# Genomics
A repo made to store coding scripts, questions and tests for the genomics x ai honors program.

### Results
Two experiments were run and respective code and results are stored in exp1, exp2 directories   

#### Experiment 1
*Experiment 1:* Next token prediction   
*GOAL:* See whether evo2 can correctly predict next token   
*TASK:* I fed 2.5K token (base pair) length to evo2 model and asked it to generate the next 2.5K tokens.   
*EVALUATION:* I used the eval metrics provided in the evo2 generation notebook on GitHub   
*RESULTS:* I have to discuss different results based on the varying temperature of generation   
At temp= 0.1, (more deterministic) very good similarity! (check plot)   
At temp=0.5, fluctuating (I am not sure about interpretation)   


#### Experiment 2
*Experiment 2*: CENP-B box prediction   
*GOAL:* See whether evo2 can predict cenp-b boxes.   
*TASK:* I fed the first part of a sequence ending with a CENP-B box as prompt and generated from the model. Prompt length = generation length = 2.5K tokens (base pairs)   
*ISSUE:* In the genome I am provided by Matteo I cannot seem to find CENP-B boxes. I have to check and make sure the why.   


### TBD
1. I get chromosomeM when reading the file? (check explore_fasta.py) for more   
2. Base-pairs being some uppercase, some lowercase. Discuss intepretation & how it affects results   
3. Setting the right temperature for genome generation (check evo2.generate() arguments)   
4. Scaling the results to other genomes (currently available CH3V2)
5. Token sizes / lengths of generation and their impact   

### Side notes
I did not upload the genome provided by Matteo Ungaro here (I need to ask for permission) so you cannot reproduce the code.    
However, if you follow the instructions to get evo2 on the notebook you can generate from any input sequence and see how the model works.