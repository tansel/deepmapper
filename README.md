# pyDeepMapper

This package provides the implementation of 
[tansel/DeepMapper](https://github.com/tansel/DeepMapper) 


## Installation
    python3 -m pip -q install git+https://github.com/tansel/pyDeepMapper.git#egg=pyDeepMapper

## Overview

DeepMapper enables a simple pipeline to process non-image data as images to analyse any high dimensional data using CNNs or various DL algorithms and systemically collect and interpret results 


## Jupyter Notebooks

* [Demo of CNN recognition of dispersed MNIST dataset](./pytorch-mnist-resnet18-shuffle-demo.ipynb)*
* [Generation and basic analysis of Needle in the Haystack NIHS data](./deepmap_pytorch_poc_nihs_data_generation.ipynb)
* [DeepMapper Proof of concept analysing NIHS data](./deepmapper_pytorch-proof-of-concept.ipynb)
* [DeepMapper TCGA data analysis for comparison with DeepInsight](./deepmapper_pytorch-TCGAData.ipynb)

As GitHub doesn't like uploading large data, the data is available from authors in pickle form. Please contact authors to receive data before the data is commited to a public data repository.

## References

<a id="1">\[1\]</a>
Tansel Ersavas, Martin A. Smith, John S. Mattick et al. Novel applications of Convolutional Neural Networks in the age of Transformers, 19 January 2024, PREPRINT (Version 1) available at Research Square [https://doi.org/10.21203/rs.3.rs-3868861/v1]


<a id="2">\[3\]</a>
Narine Kokhlikyan et. al. (2020). Captum: A unified and generic model interpretability library for PyTorch. https://github.com/pytorch/captum

