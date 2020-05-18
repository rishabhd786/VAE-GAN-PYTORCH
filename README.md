# Paper

* **Title**: Autoencoding beyond pixels using a learned similarity metric
* **Authors**: Anders Boesen Lindbo Larsen,Søren Kaae Sønderby,Hugo Larochelle,Ole Winther
* **Link**: https://arxiv.org/pdf/1512.09300.pdf
* **Tags**: Neural Network, Generative Networks, GANs
* **Year**: 2016

### Usage
```bash
$ python3 main.py 
```
# Summary

## Introduction

* The paper combine VAEs and GANs into an unsupervised generative model that simultaneously learns to encode, generate and compare dataset samples.

* It shows that generative models trained with learned similarity measures produce better image samples than models trained with element-wise error measures.

* It demonstrate that unsupervised training results in a latent image representation with disentangled factors of variation (Bengio et al., 2013). This is illustrated in experiments on a dataset of face images labelled with visual attribute vectors, where it is shown that simple arithmetic applied in the learned latent space produces images that reflect changes in these attributes.

## Variational Autoencoder

A VAE consists of two networks that encode a data samplex to a latent representation z and decode the latent representation back to data space, respectively:

                                              z ∼ Enc(x) = q(z|x) , x˜ ∼ Dec(z) = p(x|z)
                                              
The VAE regularizes the encoder by imposing a prior over the latent distribution p(z). Typically z ∼ N (0, I) is chosen. The VAE loss is minus the sum of the expected log likelihood (the reconstruction error) and a prior regularization term:




