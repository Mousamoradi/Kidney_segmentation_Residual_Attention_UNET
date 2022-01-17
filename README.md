# Kidney_segmentation_Residual_Attention_UNET
# This python code has two main goals: 
# 1- To check image rocessing including CLAHE contrast enhancement and automatic OTSU thresholding on kidney segmentation performance.
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
