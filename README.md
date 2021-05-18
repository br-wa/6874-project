# 6874-project
Transcription factors (TF) are important mediators of gene expression, and understanding their binding activities in different states is essential in understand biological processes.  Here we report two novel architectures that rely on variational auto-encoders (VAEs) to learn latent representations of TF or cells based on TF binding probabilities at the single cell level over the whole genome generated from single cell ATACseq data. The first architecture uses a dual VAE model where latent spaces for each TF are learned for each cell, followed by a second VAE model which concatenates the learned latent spaces over all cells and learns a cell invariant latent represenation for each TF. This architecture was able to detect known TF-TF relationships and uncover possible new interactions. The second architecture aims to learn the latent representations of cells and TFs concurrently through a branched VAE model. This model was able cluster cells using the latent representation similarly to the input with a marked reduction in the dimensionality and maintain TF relationships in a separate latent space. These two architectures provide novel methods of analyzing single cell ATACseq data and uncover new possible interactions between TFs. 

Architectures are names as follows:

-Dual VAE: 

-Brached VAE: branched VAE with decoder predicting per-cell TF-specific binding data based on encodings of a given cell and TF

-Branched VAE with meshing: branched VAE with a meshing layer so the encoded latent spaces are decoded by the same decoder
