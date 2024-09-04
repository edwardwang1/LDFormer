# Latent Spaces Enable Transformer-Based Dose Prediction in Complex Radiotherapy Plans

## Table of Contents

- [Introduction](#introduction)
- [Requirements](#requirements)
- [Usage](#usage)
- [References](#references)
- [Citation](#citation)


## Introduction

This is the companion repository for the paper "Latent Spaces Enable Transformer-Based Dose Prediction in Complex Radiotherapy Plans", presented at MICCAI 2024. Currently, the code available will allow a user to train their own models. 

## Requirements

* Python 3.9
* PyTorch 2.0.1

## Usage

A detailed tutorial is still a work in progress.

### Preprocessing and Setup
The ground truth dose distribution, binary masks of the PTVs, and the OAR map should be stored as numpy files in the folder structure presented below. Change the relevant paths in the '.yml' files. The Dose and OARs should be saved as 2D axial slices, while the PTVs should be saved as 3D numpy arrays. Keep in mind that each patient will have multiple PTVs.

- Dose  
  - Training
    - Patient1Slice1.npy
    - Patient1SliceN.npy
    - Patient2Slice1.npy
    - Patient2SliceN.npy
    - Patient3Slice1.npy
    - Patient3SliceN.npy
  - Validation
    - Patient4Slice1.npy
    - Patient4SliceN.npy
  - Testing
    - Patient5Slice1.npy
    - Patient5SliceN.npy
- PTVs  
  - Training
    - Patient1PTV1.npy
    - Patient1PTV2.npy
  - Validation
    - Patient2PTV1.npy
    - Patient2PTV2.npy
  - Testing
    - Patient3PTV1.npy
    - Patient3PTV2.npy
- OARs  
  - Training
    - Patient1Slice1.npy
    - Patient1SliceN.npy
    - Patient2Slice1.npy
    - Patient2SliceN.npy
    - Patient3Slice1.npy
    - Patient3SliceN.npy
  - Validation
    - Patient4Slice1.npy
    - Patient4SliceN.npy
  - Testing
    - Patient5Slice1.npy
    - Patient5SliceN.npy

## Creating the Initial Dose Estimation
The initial dose estimation is an estimate of the dose delivered to all PTVs based on a double exponential model. You will need to create one for each plan. Details of this can be found in this paper:

```
Edward Wang, Hassan Abdallah, Jonatan Snir, Jaron Chong, David. A. Palma, Sarah A. Mattonen, Pencilla Lang,
Predicting the 3-Dimensional Dose Distribution of Multi-Lesion Lung Stereotactic Ablative Radiotherapy with Generative Adversarial Networks,
International Journal of Radiation Oncology*Biology*Physics,
2024,
,
ISSN 0360-3016,
https://doi.org/10.1016/j.ijrobp.2024.07.2329.
```

### Training VQVAEs
Use [trainVQVAE2D](https://github.com/edwardwang1/LDFormer/blob/main/trainVQVAE2D.py) and [trainVQVAE](https://github.com/edwardwang1/LDFormer/blob/main/trainVQVAE.py) to train the VQVAEs for the OARs, dose distributions and PTVs. The weights will automatically saved to the directory given in the configuration (".yml") files.

### Encoding Images and Creating Sequences
Use [encodeVolumes](https://github.com/edwardwang1/LDFormer/blob/main/encodeVolumes.py) to use the trained VQVAEs to encode the spatial data into sequences. This will save the sequences (which are .txt files) into a folder called "Embeddings".
- Embeddings
  - PTVs
    - Training
    - Testing
    - Validation
  - Oars
    - Training
    - Testing
    - Validation
  - Dose
    - Training
    - Testing
    - Validation
  - IDE
    - Training
    - Testing
    - Validation
   
Then use [createSequencesForTransformer](https://github.com/edwardwang1/LDFormer/blob/main/createSequencesForTransformer.py) to compile all of the encodings into a single combined sequence for each patient.

### Training Transformer
Use [trainTransformer](https://github.com/edwardwang1/LDFormer/blob/main/trainTransformer.py) to train the transformer on the combined sequences.

## References

The implementation of the transformer is based on Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT).
The implementation of the VQVAE is based on the [official implementation](https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb) by the VQVAE authors.

## Citation

If you use this repostiory in your work, please cite the Arxiv paper below. This README will be updated once the MICCAI proceeedings have been published.

Arxiv Paper
```
@misc{wang2024latentspacesenabletransformerbased,
      title={Latent Spaces Enable Transformer-Based Dose Prediction in Complex Radiotherapy Plans}, 
      author={Edward Wang and Ryan Au and Pencilla Lang and Sarah A. Mattonen},
      year={2024},
      eprint={2407.08650},
      archivePrefix={arXiv},
      primaryClass={physics.med-ph},
      url={https://arxiv.org/abs/2407.08650}, 
}
```


