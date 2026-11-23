# Baselines

| Component | Source | Distribution |
|---|---|---|
| CLAHE-ORB-SLAM3 | [CLAHE-ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Included |
| DarkSLAM-ORB-SLAM2 | [DarkSLAM-ORB-SLAM2](https://github.com/shanpenghui/ORB_SLAM2_fixed) | Included |
| DarkSLAM-EnlightenGAN | [DarkSLAM-EnlightenGAN](https://github.com/arsenyinfo/EnlightenGAN-inference) | Included |
| URRE-Net | [URRE-Net](https://github.com/Chenxy875/URRE-Net) | Official link |
| URRE-Net-ORB-SLAM3 | [URRE-Net-ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) | Included |
| DROID-SLAM | [DROID-SLAM](https://github.com/princeton-vl/DROID-SLAM) | Included |
| Zero-DCE | [Zero-DCE](https://github.com/Li-Chongyi/Zero-DCE) | Official link |
| Zero-DCE-DROID | [Zero-DCE-DROID](https://github.com/princeton-vl/DROID-SLAM) | Included |
| SCI | [SCI](https://github.com/vis-opt-group/SCI) | Official link |
| SCI-DROID | [SCI-DROID](https://github.com/princeton-vl/DROID-SLAM) | Included |
| AirSLAM | [AirSLAM](https://github.com/sair-lab/AirSLAM) | Included |
| AirSLAM-g2o | [AirSLAM-g2o](https://github.com/RainerKuemmerle/g2o) | Official link |
| IRAF-SLAM | [IRAF-SLAM](https://github.com/thanhnguyencanh/IRAF-SLAM) | Included |
| Retinexformer | [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer) | Included |
| RetinexNet | [RetinexNet](https://github.com/weichen582/RetinexNet) | Official link |
| KinD | [KinD](https://github.com/zhangyhuaee/KinD) | Official link |
| SNR-Aware | [SNR-Aware](https://github.com/JIA-Lab-research/SNR-Aware-Low-Light-Enhance) | Official link |

Follow each upstream README for dependencies and build steps. Included SLAM integrations contain the changes used in the experiments; license files are retained. Recorded upstream revisions are in [versions.json](versions.json). When a component is obtained externally, place it under `baselines/third_party/<component>` to use the enhancement adapter.

`adapters/infer.py` provides the Retinexformer, RetinexNet, KinD and SNR-Aware LIE inference settings. Model files go under `weights/`; TensorFlow methods require their corresponding TensorFlow environment. The adapter's repository-relative paths replace server-specific paths.

AirSLAM uses synchronized stereo input. Its TensorRT integration may require rebuilding models for the target GPU/runtime; the g2o dependency is linked above. SCI comparisons use the CVPR medium checkpoint. DarkSLAM was evaluated through the included ORB-SLAM2 integration with converted EnlightenGAN weights; equivalence to the original DarkSLAM authors' checkpoint is unverified.

End-to-end installation in a fresh environment has not been validated. Dataset images must be obtained separately.
