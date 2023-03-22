# Image based Cell Phenotyping using Vision Transformers on Amazon SageMaker

<p align="center">
  <img src="./images/PyTorch DDP.png" alt="EC2 + PyTorch" height="200"/>
</p>

## Background

The ability to phenotype cells is important for biological research and drug development. Traditional phenotyping methods rely on fluorescence labeling of specific markers. However, reliance on traditional phenotyping methods may be unviable or undesirable in certain contexts. This solution builds on a deep learning approach for phenotyping disaggregated single cells with a high degree of accuracy using low-resolution bright-field and non-specific fluorescence images of the nucleus, cytoplasm, and cytoskeleton. The model trains a CNN using cell images from eight standard cancer cell-lines. The solution is based on the article published in [Nature](https://www.nature.com/articles/s42003-020-01399-x) 


## Concepts Covered
The following concepts are covered as part of this implementation: 
- Developing machine learning models on [Amazon SageMaker](https://aws.amazon.com/pm/sagemaker/)
- Running Distributed Data Parallel model training on a Vision Transformer with [Pytorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- Track and save model experiments in [SageMaker Experiments](https://aws.amazon.com/blogs/aws/amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings/) 
- Running Hyperparameter Tuning using [SageMaker Automatic Model Tuning](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning.html)

# Getting started  
### 1. Prerequisites
#### Platforms 
##### Amazon SageMaker   
It is recommended that the training job is executed in a multi-GPU instance such as p3.16xlarge or p3dn.24xlarge. 
##### Amazon S3   
For ease of use, download the training data from [UBC Research Data Collection](https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/TDULMF) into an S3 bucket

#### Other ressources
- Python 3.11 
- PyTorch  
- Torchvision
- You can find all the additional information in the `requirements.txt` file

