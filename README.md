# Flood Area Segmentation  

## Introduction  
This repository showcases the implementation of the U-Net architecture using PyTorch for flood area segmentation.  
- The model's architecture is defined in `UNet_module.py` and `UNet_model.py`.  
- A demo for inference is available in the `inference.ipynb` notebook.  


## Resources  
UNET-paper: [https://arxiv.org/pdf/1505.04597](https://arxiv.org/pdf/1505.04597)  
Dataset: [https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation](https://www.kaggle.com/datasets/faizalkarim/flood-area-segmentation)

## Training on CUDA  
To train the model on a GPU, follow these steps:  
1. Open a terminal and run `nvidia-smi` to check your NVIDIA GPU's **CUDA version**.  
2. Visit [pytorch.org](https://pytorch.org) and find the corresponding PyTorch version that matches your CUDA version.  
3. Copy and paste the installation command into your terminal to install the appropriate drivers and dependencies.  

Now you're ready to train the model efficiently using GPU acceleration! 🚀
