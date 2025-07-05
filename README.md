# Neural Importance Sampling of Many Lights

### [Paper](https://arxiv.org/pdf/2505.11729) | [Project Page](https://pedrovfigueiredo.github.io/projects/manylights/SIGGRAPH_2025_Importance_Sampling/index.html)

This is the official implementation of the paper, titled "Neural Importance Sampling of Many Lights", to be presented at ACM SIGGRAPH 2025 (conference track).

<img src="media/Overview.png" width="800px"/> <br/>
We propose a hybrid neural approach for estimating spatially varying light selection distributions in rendering with many lights.

## Abstract
We propose a neural approach for estimating spatially varying light selection distributions to improve importance sampling in Monte Carlo rendering, particularly for complex scenes with many light sources. Our method uses a neural network to predict the light selection distribution at each shading point based on local information, trained by minimizing the KL-divergence between the learned and target distributions in an online manner. To efficiently manage hundreds or thousands of lights, we integrate our neural approach with light hierarchy techniques, where the network predicts cluster-level distributions and existing methods sample lights within clusters. Additionally, we introduce a residual learning strategy that leverages initial distributions from existing techniques, accelerating convergence during training. Our method achieves superior performance across diverse and challenging scenes in equal-sample settings.


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
