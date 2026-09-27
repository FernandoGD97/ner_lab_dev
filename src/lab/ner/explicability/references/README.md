# Scientific reference catalogue

`references.bib` contains the original or standard methodological sources used by
Part 2. The implementation keeps the corresponding scientific claims separate:

* **CKA** compares representation spaces with invariances appropriate to neural
  representations (Kornblith et al., 2019).
* **SVCCA/PWCCA** compare subspaces through canonical correlations; PWCCA weights
  canonical directions by their contribution (Raghu et al., 2017; Morcos et al.,
  2018). They are optional because repeated decompositions are expensive.
* **UMAP/AlignedUMAP** provide nonlinear visualization, not a quantitative drift
  measure (McInnes et al., 2018). Part 2 never fits unrelated maps per checkpoint.
* **Dataset Cartography** uses confidence and variability to describe example
  dynamics (Swayamdipta et al., 2020); forgetting counts learned-to-unlearned
  transitions (Toneva et al., 2019).
* **Probing** measures linear decodability, not causal model use (Alain and Bengio,
  2017). Optional MDL probing controls for probe complexity (Voita and Titov,
  2020).
* **Intrinsic dimension** uses covariance participation ratio and the local TwoNN
  estimator (Facco et al., 2017). Anisotropy diagnostics follow the contextual
  representation literature (Ethayarajh, 2019).
* **Attribution** is targeted and optional. Integrated Gradients is from Sundararajan
  et al. (2017), while sanity checks follow Adebayo et al. (2018); saliency is not
  assumed faithful without perturbation tests.
* **Influence** uses the TracIn checkpoint-gradient formulation (Pruthi et al.,
  2020) only for selected cases.
* **Topology** is optional and dependency-gated; persistent-homology terminology
  follows Edelsbrunner and Harer (2010).

The catalogue documents methods but does not imply that optional methods were run.
