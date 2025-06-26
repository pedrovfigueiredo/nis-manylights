# Neural Importance Sampling of Many Lights

### [Paper](https://arxiv.org/pdf/2505.11729) | [Project Page](https://pedrovfigueiredo.github.io/projects/manylights/SIGGRAPH_2025_Importance_Sampling/index.html)

This is the official implementation of the paper, titled "Neural Importance Sampling of Many Lights", to be presented at ACM SIGGRAPH 2025 (conference track).

<img src="media/Overview.png" width="800px"/> <br/>
We propose a hybrid neural approach for estimating spatially varying light selection distributions in rendering with many lights.

## Abstract
In this paper, we present a neural path guiding method to aid with Monte Carlo (MC) integration in rendering. Existing neural methods utilize distribution representations that are either fast or expressive, but not both. We propose a simple, but effective, representation that is sufficiently expressive and reasonably fast. Specifically, we break down the 2D distribution over the directional domain into two 1D probability distribution functions (PDF). We propose to model each 1D PDF using a neural network that estimates the distribution at a set of discrete coordinates. The PDF at an arbitrary location can then be evaluated and sampled through interpolation. To train the network, we maximize the similarity of the learned and target distributions. To reduce the variance of the gradient during optimizations and estimate the normalization factor, we propose to cache the incoming radiance using an additional network. Through extensive experiments, we demonstrate that our approach is better than the existing methods, particularly in challenging scenes with complex light transport.


## News
- **2025.06.26**: Repo is released.

## TODO List
- [ ] Code Release

## Citation
If our work is useful for your research, please consider citing:
```
@inproceedings{NIS_ManyLights_sig25,
    title={Neural Importance Sampling of Many Lights},
    author={Figueiredo, Pedro and He, Qihao and Bako, Steve and Khademi Kalantari, Nima},
    booktitle={ACM SIGGRAPH 2025 Conference Papers},
    year={2025},
    doi = {10.1145/3721238.3730754},
    numpages = {11},
    isbn = {979-8-4007-1540-2/2025/08}, 
}
```
