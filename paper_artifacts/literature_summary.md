# Literature Summary: Motion-Guided Self-Supervised Learning for Cardiac Cine MRI Analysis

**Date:** 2026-11-20  
**Companion File:** `paper_artifacts/references.bib`  

---

## 1. Cardiac Cine MRI Segmentation & Benchmarks

1. **Bernard et al. (2018) — ACDC Benchmark (`bernard2018deep`)**
   - *Title:* "Deep Learning Techniques for Automatic MRI Cardiac Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?" (*IEEE TMI*, 2018)
   - *Relevance:* The canonical benchmark dataset utilized in this project. Establishes the 100-patient short-axis cine cohort, patient distribution across 5 pathologies, and standard target classes: Left Ventricle (LV) cavity, Myocardium (MYO), and Right Ventricle (RV) cavity at end-diastole (ED) and end-systole (ES). Identifies remaining clinical challenges in thin myocardial borders and basal/apical slice transitions.

2. **Isensee et al. (2021) — nnU-Net Framework (`isensee2021nnu`)**
   - *Title:* "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation" (*Nature Methods*, 2021)
   - *Relevance:* The state-of-the-art heuristic framework that self-configures preprocessing, network topology, loss, and post-processing. Serves as an gold-standard benchmark in medical image segmentation against which label efficiency must be evaluated.

3. **Chen et al. (2020) — Cardiac Segmentation Review (`chen2020deep`)**
   - *Title:* "Deep Learning for Cardiac Image Segmentation: A Review" (*Frontiers in Cardiovascular Medicine*, 2020)
   - *Relevance:* Reviews clinical metrics derived from segmentation (ejection fraction, myocardial mass) and highlights the high annotation cost of multi-frame cine sequences as the primary motivation for label-efficient and temporal learning.

---

## 2. Convolutional Neural Network Baselines

4. **Ronneberger et al. (2015) — U-Net (`ronneberger2015u`)**
   - *Title:* "U-Net: Convolutional Networks for Biomedical Image Segmentation" (*MICCAI*, 2015)
   - *Relevance:* The foundational encoder-decoder architecture with skip connections. Forms the primary baseline backbone specified in Section 3 and Stage A of the project documentation.

5. **Oktay et al. (2018) — Attention U-Net (`oktay2018attention`)**
   - *Title:* "Attention U-Net: Learning Where to Look for the Pancreas" (*arXiv:1804.03999*, 2018)
   - *Relevance:* Introduces attention gate modules within skip connections to suppress irrelevant background features and highlight target organs.

6. **Zhou et al. (2018) — UNet++ (`zhou2018unetplus`)**
   - *Title:* "UNet++: A Nested U-Net Architecture for Medical Image Segmentation" (*DLMIA*, 2018)
   - *Relevance:* Dense nested skip pathways to reduce semantic gap between encoder and decoder feature maps.

7. **Gu et al. (2019) — CE-Net (`gu2019cenet`)**
   - *Title:* "CE-Net: Context Encoder Network for 2D Medical Image Segmentation" (*IEEE TMI*, 2019)
   - *Relevance:* Integrates dense atrous convolution (DAC) and residual multi-kernel pooling (RMP) in the bottleneck to capture multi-scale spatial context.

---

## 3. Vision Transformers for Medical Segmentation

8. **Chen et al. (2021) — TransUNet (`chen2021transunet`)**
   - *Title:* "TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation" (*arXiv:2102.04306*, 2021)
   - *Relevance:* Hybrid architecture combining a ResNet-50 CNN backbone with a Vision Transformer (ViT-B/16) and a cascaded upsampler. Tested on ACDC as a high-capacity 100% supervised baseline (Mean Dice = 0.8189, checkpoint size 1.18 GB).

9. **Cao et al. (2022) — Swin-Unet (`cao2022swin`)**
   - *Title:* "Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation" (*ECCV Workshops*, 2022)
   - *Relevance:* Pure transformer U-Net utilizing shifted windows (Swin-Tiny) for hierarchical token representation. Tested on ACDC as a competitive baseline (Mean Dice = 0.7994, checkpoint size 580 MB).

10. **Xie et al. (2021) — SegFormer (`xie2021segformer`)**
    - *Title:* "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers" (*NeurIPS*, 2021)
    - *Relevance:* Hierarchical Mix Transformer (MiT-B2) encoder with lightweight MLP decoders avoiding positional embeddings. Tested as a baseline (Mean Dice = 0.8038, checkpoint size 44.8 MB).

---

## 4. Medical Foundation Models

11. **Ma et al. (2024) — MedSAM (`ma2024medsam`)**
    - *Title:* "Segment Anything in Medical Images" (*Nature Communications*, 2024)
    - *Relevance:* Domain-adapted foundation model trained on >1.5M medical image-mask pairs. Tested in this project under zero-shot prompting with prior boxes (Mean Dice = 0.1790) and fine-tuned mask decoders across 10%–100% labels (reaching 0.4897 at 100%), highlighting the limitations of general foundation models when fine-grained cardiac anatomical motion is neglected.

12. **Kirillov et al. (2023) — Segment Anything (`kirillov2023segment`)**
    - *Title:* "Segment Anything" (*ICCV*, 2023)
    - *Relevance:* Foundational promptable segmentation framework providing the architecture and prompting paradigm adapted by MedSAM.

---

## 5. Self-Supervised & Temporal Representation Learning

13. **Chen et al. (2020) — SimCLR (`chen2020simple`)**
    - *Title:* "A Simple Framework for Contrastive Learning of Visual Representations" (*ICML*, 2020)
    - *Relevance:* Foundational contrastive learning framework using InfoNCE loss on augmented views, motivating feature-level consistency across temporal views.

14. **He et al. (2022) — Masked Autoencoders (`he2022masked`)**
    - *Title:* "Masked Autoencoders Are Scalable Vision Learners" (*CVPR*, 2022)
    - *Relevance:* Theoretical inspiration for the masked image reconstruction module in `src/ssl_pretraining.py`, forcing the encoder to learn structural representations of cardiac anatomy from masked patches.

15. **Bai et al. (2019) — Anatomical SSL for Cardiac MRI (`bai2019self`)**
    - *Title:* "Self-Supervised Learning for Cardiac MR Image Segmentation by Predicting Anatomical Positions" (*MICCAI*, 2019)
    - *Relevance:* First demonstration that predicting spatial and temporal coordinates on UK Biobank cardiac MRI yields representations that significantly improve downstream segmentation under label scarcity.

16. **Chaitanya et al. (2020) — Medical Contrastive Learning (`chaitanya2020contrastive`)**
    - *Title:* "Contrastive Learning of Global and Local Features for Medical Image Segmentation with Limited Annotations" (*NeurIPS*, 2020)
    - *Relevance:* Formulates combined global and local contrastive objectives tailored to volumetric medical datasets, establishing benchmarks under 10%–50% label availability.

17. **Morales et al. (2021) — Cardiac Motion SSL (`morales2021self`)**
    - *Title:* "Self-Supervised Learning of Cardiac Motion Representations" (*IEEE TMI*, 2021)
    - *Relevance:* Directly establishes that tracking cardiac motion without supervision yields discriminative representations for downstream clinical phenotypes.

---

## 6. Motion Estimation & Spatial Transformation

18. **Dosovitskiy et al. (2015) — FlowNet (`dosovitskiy2015flownet`)**
    - *Title:* "FlowNet: Learning Optical Flow with Convolutional Networks" (*ICCV*, 2015)
    - *Relevance:* Inspirer of `SimpleFlowNet` in `src/motion.py`, providing a lightweight CNN architecture for predicting dense 2D displacement vectors between adjacent frames.

19. **Jaderberg et al. (2015) — Spatial Transformer Networks (`jaderberg2015spatial`)**
    - *Title:* "Spatial Transformer Networks" (*NeurIPS*, 2015)
    - *Relevance:* Mathematical foundation of `SpatialTransformer` in `src/motion.py`, utilizing differentiable bilinear grid sampling for end-to-end backpropagation through warped representations.

20. **Balakrishnan et al. (2019) — VoxelMorph (`balakrishnan2019voxelmorph`)**
    - *Title:* "VoxelMorph: A Learning Framework for Deformable Medical Image Registration" (*IEEE TMI*, 2019)
    - *Relevance:* Standard reference for unsupervised deformable image registration, combining intensity matching with spatial gradient smoothness penalties.

---

## 7. Semi-Supervised Learning & Confidence Filtering

21. **Sohn et al. (2020) — FixMatch (`sohn2020fixmatch`)**
    - *Title:* "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence" (*NeurIPS*, 2020)
    - *Relevance:* Popularized strict confidence thresholding ($p_{\max} \ge \tau$) to filter pseudo-labels, directly mirrored in `src/pseudo_labels.py` ($p_{\max} \ge 0.90$).

22. **Lee (2013) — Pseudo-Label (`lee2013pseudo`)**
    - *Title:* "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks" (*ICML Workshops*, 2013)
    - *Relevance:* Foundational formulation of pseudo-label cross-entropy training on unlabeled data partitions.

23. **Sedai et al. (2019) — Uncertainty-Guided Semi-Supervision (`sedai2019uncertainty`)**
    - *Title:* "Uncertainty Guided Semi-supervised Segmentation of Retinal Layers in OCT Images" (*MICCAI*, 2019)
    - *Relevance:* Demonstrates that Shannon entropy filtering eliminates ambiguous edge pixels, preventing confirmation bias in medical semi-supervised segmentation.
