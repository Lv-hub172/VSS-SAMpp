# VSS-SAM++: Visual State Space-Aware SAM for 3D Medical Image Segmentation
<p align="center">
  <img src="assets/overview.png" width="95%">
</p>
VSS-SAM++ is a PyTorch-based framework for 3D medical image segmentation that combines the strong visual priors of the Segment Anything Model (SAM) with the long-range dependency modeling ability of Vision Mamba. The proposed dual-branch architecture uses a SAM branch to extract high-level semantic representations and a Mamba branch to capture cross-slice 3D contextual information, followed by a gated deep feature fusion module to adaptively integrate complementary features. Extensive experiments on nine public CT and MRI datasets demonstrate that VSS-SAM++ achieves superior segmentation performance over representative CNN-based, Transformer-based, Mamba-based, and SAM-based methods.
## Acknowledgments

Our code is based on [MA-SAM]([https://github.com/hitachinsk/SAMed](https://github.com/cchen-cc/MA-SAM/tree/main)), [SAMed](https://github.com/hitachinsk/SAMed), [FacT](https://github.com/cchen-cc/FacT), and [Segment Anything](https://github.com/facebookresearch/segment-anything). We appreciate the authors for their great works.
