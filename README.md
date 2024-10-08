# VADS

This repository contains the training and testing code for the CVPR'24 paper titled with  "***Visual-Augmented Dynamic Semantic Prototype
for Generative Zero-Shot Learning***".


## Requirements
The code implementation of **VADS** mainly based on [PyTorch](https://pytorch.org/).


## Preparing Dataset and Model

We provide pre-trained models ([Google Drive](https://drive.google.com/drive/folders/1D5C8An6JZY24SyCMb7vQk36egdauJVPE?usp=drive_link)) on CUB dataset. You can download model files as well as corresponding datasets, and organize them as follows: 
```
.
├── data
│   ├── CUB/
│   ├── SUN/
│   └── AWA2/
└── ···
```


## Train
Runing following commands and training **VADS**:

Need to modify the wandb_config file.

```
$ python train_CUB.py
```





## Citation
If you find VADS is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry.

```bibtex
@inproceedings{hou2024visual,
  title={Visual-Augmented Dynamic Semantic Prototype for Generative Zero-Shot Learning},
  author={Hou, Wenjin and Chen, Shiming and Chen, Shuhuang and Hong, Ziming and Wang, Yan and Feng, Xuetao and Khan, Salman and Khan, Fahad Shahbaz and You, Xinge},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={23627--23637},
  year={2024}
}
```
