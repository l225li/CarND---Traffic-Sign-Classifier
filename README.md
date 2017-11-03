# **Traffic Sign Recognition** 
*Author: Lin LI*

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test/1.JPEG "Traffic Sign 1"
[image5]: ./test/2.JPEG "Traffic Sign 2"
[image6]: ./test/3.JPEG "Traffic Sign 3"
[image7]: ./test/4.JPEG "Traffic Sign 4"
[image8]: ./test/5.JPEG "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/l225li/CarND-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is ?
34799
* The size of the validation set is ?
4410
* The size of test set is ?
12630
* The shape of a traffic sign image is ?
(32, 32)
* The number of unique classes/labels in the data set is ?
43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the number of each traffic sign is ditributed in the training data. 

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I used the recommended image normalization method `(pixel - 128)/128`, however it made my model overfitted on training data. After test and trial, I discover that without preprocessing, my model can actually achieve better accuraccy. Therefore I used the original 32x32 colored images without preprocessing. 

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x9 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x9				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x18   |
| RELU					|												|
| Max pooling			| 2x2 stride, outputs 5x5x18        			|
| Flatten				| outputs 450									|
| Fully Connected		| outputs 160									|
| RELU 					|												|
| Dropout 				|												|
| Fully Connected		| outputs 100									|
| RELU 					|												|
| Fully Connected		| logits, outputs 43							|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used mini-batching optimized by AdamOptimizer which is a modification of SGD (Stochastic Gradient Descent) at a starting learning rate of 0.001. Each batch size is 128 training samples. I used L2 regularization on hidden layers at a rate of 0.01. Dropout keep rate is set at 0.6. I first trained the model for 20 epochs, and then trained more epochs while tuning the hyperparameters when needed. 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.981
* test set accuracy of 0.958

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?

The first architecture tried was LeNet. It was chosen because the input images are of the same size 32x32. They are both image classification problems with input images with objects well centered. 

* What were some problems with the initial architecture?

The problems with LeNet when applied on the traffic signs data is that there are more complex features in the traffic signs images than in the MNIST dataset. There are more output classes also, 43 instead of 10. Therefore, the model was underfitted because it is not complex enough to learn the features from the training samples. 

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

In order to fix the issue of underfitting of the LeNet model, more hidden layers and more filters in the convolutional layers are added. In the first conv layer, there are 9 filters instead of 6 in LeNet. The second layer contains 18 filters instead of 16 in LeNet. More hidden units in the Fully Connected layers as well, 160 and 100 respectively. After training a few epochs, overfitting is observed. Therefore, a Dropout regularization layer is also added to balance to performance of the model. 

* Which parameters were tuned? How were they adjusted and why?

Parameters tuned were the learning rate for the Adamoptimizer and also the keep rate of the dropout layer. When the validation accuracy is not improving after a few epochs and showing an unstable movement, the learning rate will be adjusted by dividing by 10 or 100 so that the step of gradient descent will be smaller and slower to achieve convergence. And if overfitting is observed, dropout keep rate will be lowered in order to force the network to learn the features and to be more robust. 

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolutional layer work well with images because it keeps the input data keep its structure. Pixels close with each other are related with each other and they carry special meaning when processed together instead of flattening them into a single vector of numbers. A dropout layer helps to force the neurons to learn the features instead of relying on other neurons for specific features. It helps the model to be more robust and prevents overfitting. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The images might be hard to identify because the shape is a bit distorted due to the fact that they were not square before being resized. 

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			        				|     Prediction	        					| 
|:-------------------------------------:|:---------------------------------------------:| 
| Priority road      					| Priority road   								| 
| Right-of-way at the next intersection | Right-of-way at the next intersection			|
| Stop									| Stop											|
| Road work      						| Road work 					 				|
| Bumpy road							| Bumpy road     								|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95.8%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.


##### Image 1 

For the first image, the model is relatively sure that this is a priority road (probability of 0.48), and the image does contain a priority. The top five soft max probabilities were


| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .48         			| Priority road   										| 
| .11     				| Speed limit (30km/h)									|
| .09					| End of no passing by vehicles over 3.5 metric tons	|
| .09	      			| No passing for vehicles over 3.5 metric tons 			|
| .09				    | Road work      										|

##### Image 2
For the first image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 0.56), and it is correct. The top five soft max probabilities were


| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .56         			| Right-of-way at the next intersection   				| 
| .31     				| Beware of ice/snow									|
| .21					| Vehicles over 3.5 metric tons prohibited				|
| .20	      			| Turn left ahead 										|
| .19				    | Pedestrians      										|

##### Image 3
For the first image, the model is relatively sure that this is a stop sign (probability of 0.19), and it is correct. It is interesting to see how it can recognize the road shape and red circle in the stop sign and speed limit signs. The top five soft max probabilities were


| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .19         			| Stop   												| 
| .10     				| Speed limit (60km/h)									|
| .10					| Speed limit (80km/h)									|
| .06	      			| Speed limit (50km/h)									|
| .05				    | Speed limit (30km/h)      							|

##### Image 4

For the first image, the model is relatively sure that this is a road work (probability of 0.34), and it is correct. The top five soft max probabilities were


| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .34         			| Road work  											| 
| .29     				| Bicycles crossing										|
| .27					| Bumpy road											|
| .17	      			| Wild animals crossing 								|
| .15				    | Slippery road      									|

##### Image 5
For the first image, the model is relatively sure that this is a bumpy road (probability of 0.38), and it is correct. The top five soft max probabilities were


| Probability         	|     Prediction	        							| 
|:---------------------:|:-----------------------------------------------------:| 
| .38         			| Bumpy road  											| 
| .29     				| Bicycles crossing										|
| .22					| Slippery road											|
| .16	      			| Speed limit (50km/h) 									|
| .15				    | Children crossing      								|

