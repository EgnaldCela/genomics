# Genomics
Repository made to store coding scripts, questions and experiment results for the the Genomics x AI honors program.

### Active analysis entrypoints
- `scripts/analysis/threshold_heatmaps.py`: threshold analysis and threshold-centered heatmap plotting.
- `scripts/analysis/interactive_tsne.py`: interactive Plotly t-SNE generation.
- `scripts/analysis/multiple_genomes_heatmap.py`: multi-genome CENP-B distance heatmap generation.
- `scripts/analysis/robustness_contamination.py`: contamination robustness analysis against CHM13.

### Active generated outputs
- `outputs/analysis/clusterings/`: interactive t-SNE HTML outputs.
- `outputs/analysis/heatmaps_hybrid/`: separability summaries and threshold-centered heatmaps.
- `outputs/analysis/multiple_genomes/`: multi-genome heatmap figures.
- `outputs/analysis/robustness/`: contamination robustness plots.

### Legacy archive
- Deprecated scripts are kept under `legacy/root_scripts/` and `legacy/notebooks/`.
- This preserves old workflows while keeping the root folder focused on current entrypoints.
- To regenerate the legacy multi-genome heatmap workflow, run `scripts/legacy/generate_multiple_genomes_heatmap.py`.
- Noise robustness workflow is treated as legacy (`robustness_analysis.py`).


### Results
Two experiments were run and respective code and results are stored in exp1, exp2 directories   

#### Experiment 1
*Experiment 1:* Next token prediction   
*GOAL:* See whether evo2 can correctly predict next token   
*TASK:* I fed 2.5K token (base pair) length to evo2 model and asked it to generate the next 2.5K tokens.   
*EVALUATION:* I used the eval metrics provided in the evo2 generation notebook on GitHub   
*RESULTS:* Not desirable  



#### Experiment 2
*Experiment 2*: CENP-B box prediction   
*GOAL:* See whether evo2 can predict cenp-b boxes.   
*TASK:* I fed the first part of a sequence ending with a CENP-B box as prompt and generated from the model. Prompt length = generation length = 2.5K tokens (base pairs)   
*RESULTS*: Not desirable


### TBD
1. I get chromosomeM when reading the file? (check explore_fasta.py) for more   
2. Base-pairs being some uppercase, some lowercase. Discuss intepretation & how it affects results   
3. Setting the right temperature for genome generation (check evo2.generate() arguments)   
4. Scaling the results to other genomes (currently available CH3V2)
5. Token sizes / lengths of generation and their impact   

### Side notes
I did not upload the genome provided by Matteo Ungaro here (I need to ask for permission) so you cannot reproduce the code.    
However, if you follow the instructions to get evo2 on the notebook you can generate from any input sequence and see how the model works.
