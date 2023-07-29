# Kidney_segmentation_Residual_Attention_UNET
# This python project has two main goals: 
# 1- To check image processing including CLAHE contrast enhancement and automatic OTSU thresholding on kidney segmentation performance.
# 2- To assess Residual Attention-UNET for kidney segmentation with and without CLAHE. 

Prerequisites:

Tensor Flow Version: 2.5.2 
CUDA v11.4.0,cuDNN v8.3.1

Keras Version: 2.5.0

Python 3.8.8 (default, Apr 13 2021, 15:08:03) [MSC v.1916 64 bit (AMD64)]

Pandas 1.3.4

Scikit-Learn 1.0.1


How to use the code:

1- For No image processing, go to "prepare data_no_image_processing" to prepare the images before feeding to neural network. Then run the desired deep learning model in the "Model" Folder. 

2- If image processing with CLAHE is needed, go to "image_processing+ROI detection+CLAHE" Then run the desired deep learning model in the "Model" Folder.

Note: All codes inside "Model" use no image processing in default. To activate image processing, if part 2 already executed, please UNCOMMENT the associated lines inside each model. 

# Models used:
1- FCN:

![fcn](https://user-images.githubusercontent.com/78983558/150026926-0ee45ac0-2c55-4c71-b257-890302603373.png)

2- UNET:

![unet](https://user-images.githubusercontent.com/78983558/150027030-ac17e1a6-798d-4b3c-ab1f-ec4802b30b50.png)

3- Residual-UNET:

![resnet](https://user-images.githubusercontent.com/78983558/150027072-67a20b6e-73b9-4376-9ee2-925dfe41db37.png)

4- Attention-UNET:

![attunet](https://user-images.githubusercontent.com/78983558/150027126-05c43497-7d96-458c-ade6-83706d325ff7.png)

5- Residual Attention-UNET:

![resattunet](https://user-images.githubusercontent.com/78983558/150027167-b557ff39-39db-49e0-b90e-229328624e32.png)

[![DOI](https://zenodo.org/badge/448736581.svg)](https://zenodo.org/badge/latestdoi/448736581)
