neuron.py: Data preprocessing algorithm, including histogram equalization, normalization and data size cropping.
dataset.py : Used to generate neuron training and validation dataset. The data is composed of three parts: input neuron image, input pixel-wise mask and generated one-hot vector named target used for training. The path used in ‘Mydataset’ leads to the preprocessed npy file.
losses.py: Containing BCEdice loss used for training.
unet.py: The network framework used for training, including UNet and NestedUNet(Unet++).
metrics.py: This file is mainly used to observe the training process and evaluate the model. The metrics include iou, dice coefficient and pixel accuracy. The save_result_comparision function is used to save the original image, original mask and predicted mask for direct comparision.
train.py: This file is mainly divided into four parts: parse_args, training, validation and main program.The parse_args part including most of the parameters we could set for the network. The train part is used for training the network and the validation part is used to dynamically observe the training process. In the main program, panda module is used to log the training process including loss and common metrics for both the training set and validation set. The prefix in the main program is the path to save the log text and the best model evaluated by the dice coefficient for validation set.
test.py: This file is used to test our trained model. When we test the model, we changed the path in dataset.py to load our testing data. The result would save in the path set in metrics(save_result_comparision function).
output.py: Once we get the output data, this file is used to plot the images. 
metrics_test.py: This file is used for validation and test data, to give the predicted labels with watershed lines and every centroid of the original labels and predicted labels. 
centroid.py: This file is used to compare the distances among corresponding centroids and give a intuitive result for the prediction. 
overlapping-roi-segmentation.ipynb: This file is used to segment overlapping labels via the watershed algorithm. It also finds centroids of all labels in an image. 
label_generator_with_idx.m: This code generates a training/testing set by cropping many images with dimension 256 by 256 from a raw figure with dimension 1152 by 1152.
