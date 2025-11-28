# CAMEO: Correspondence-Attention Alignment for Multi-View Diffusion Models

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://cvlab-kaist.github.io/CAMEO/)
[![arXiv](https://img.shields.io/badge/arXiv-25xx.xxxxx-b31b1b.svg)](https://arxiv.org/abs/25xx.xxxxx)

**Minkyung Kwon**<sup>\*1</sup>, **Jinhyeok Choi**<sup>\*1</sup>, **Jiho Park**<sup>\*1</sup>, **Seonghu Jeon**<sup>1</sup>, <br>
**Jinhyuk Jang**<sup>1</sup>, **Junyoung Seo**<sup>1</sup>, **Minseop Kwak**<sup>1</sup>, **Jin-Hwa Kim**<sup>&dagger;2,3</sup>, **Seungryong Kim**<sup>&dagger;1</sup>

<sup>1</sup>KAIST AI, <sup>2</sup>NAVER AI Lab, <sup>3</sup>SNU AIIS

<small>* Equal contribution &nbsp;&nbsp; &dagger; Co-corresponding author</small>

</div>

<br>

<div align="center">
  <img src="assets/teaser.svg" alt="CAMEO Teaser" width="100%">
</div>

## Abstract

Multi-view diffusion models have recently established themselves as a powerful paradigm for novel view synthesis, generating diverse images with high visual fidelity. However, the underlying mechanisms that enable these models to maintain geometric consistency across different viewpoints have remained largely unexplored. In this work, we conduct an in-depth analysis of the 3D self-attention layers within these models. We empirically verify that **geometric correspondence naturally emerges in specific attention layers** during training, allowing the model to attend to spatially corresponding regions across reference and target views.

Despite this emergent capability, our analysis reveals that the implicit correspondence signal is often incomplete and fragile, particularly degrading under scenarios involving complex geometries or large viewpoint changes. Addressing this limitation, we introduce **CAMEO** (Correspondence-Attention Alignment), a training framework that explicitly supervises the model's attention maps using dense geometric correspondence priors. By applying this supervision to just a single, optimal attention layer (Layer 10), CAMEO significantly enhances the model's structural understanding. Our experiments demonstrate that CAMEO **reduces the training iterations required for convergence by 50%** while consistently outperforming baseline models in geometric fidelity on challenging datasets such as RealEstate10K and CO3D.

## To Do

- [ ] Release Code
- [ ] Release Pre-trained Models
- [ ] Release Demo

## Acknowledgements

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{kwon2026cameo,
  title={CAMEO: Correspondence-Attention Alignment for Multi-View Diffusion Models},
  author={Kwon, Minkyung and Choi, Jinhyeok and Park, Jiho and Jeon, Seonghu and Jang, Jinhyuk and Seo, Junyoung and Kwak, Min-Seop and Kim, Jin-Hwa and Kim, Seungryong},
  journal={arXiv preprint arXiv:25xx.xxxxx},
  year={2025}
}
```
