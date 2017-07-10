# **Traffic Sign Recognition**

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[graph1]: ./writeup_resources/data_exp1.png "Training set old"
[graph2]: ./writeup_resources/data_exp2.png "Training set"
[graph3]: ./writeup_resources/data_exp3.png "Validation set"
[graph4]: ./writeup_resources/data_exp4.png "Test set"

[image1]: ./writeup_resources/preprocessing.png "Pre-processing"
[image2]: ./writeup_resources/augment1.png "Flip with no class change"
[image3]: ./writeup_resources/augment2.png "Flip with class change"
[image4]: ./writeup_resources/augment3.png "Rotate & Scale"
[image5]: ./writeup_resources/my_cnn.png "My CNN"
[image6]: ./writeup_resources/original_13.jpg "Traffic Sign 1"
[image7]: ./writeup_resources/original_14.jpg "Traffic Sign 2"
[image8]: ./writeup_resources/original_1.jpg "Traffic Sign 3"
[image9]: ./writeup_resources/original_30.jpg "Traffic Sign 4"
[image10]: ./writeup_resources/original_12.jpg "Traffic Sign 5"
[image11]: ./writeup_resources/prediction1.png "Prediction 1"
[image12]: ./writeup_resources/prediction2.png "Prediction 2"
[image13]: ./writeup_resources/prediction3.png "Prediction 3"
[image14]: ./writeup_resources/prediction4.png "Prediction 4"
[image15]: ./writeup_resources/prediction5.png "Prediction 5"

###Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/fadida/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Data set summary.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set before augmentation is 34,799
* The size of training set after augmentation is 123,024
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

The dataset visualization section contains info about how each dataset is distributed between classes and shows a random image each dataset.
Here we can see the all datasets (not augmented) have roughly the same distribution which means that the split is good and also that there may be some classes that will be hard to recognize because of lack of data (like classes between 19, 24 and 28)


![alt text][graph1]
![alt text][graph2]
![alt text][graph3]
![alt text][graph4]

### Design and Test a Model Architecture

#### 1. Pre-processing & augmentation.

After reading the article [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) which was referenced in the instructions. I've tried to do grayscale preprocessing, YUV preprocessing (as was suggested in the article) but in the end I found I got
the best results when I took the red channel in the RGB image, replaced it with the grayscale image and then applied histogram equalization and normalization only to that channel (the red channel).
I decided to apply those steps only to the first channel because I wanted to preserve some of the color data in order to improve the net accuracy.

Here is an example of a traffic sign image before and after preprocessing.

![alt text][image1]

I decided to generate additional data in order to improve the net accuracy, the addition in data improved the net accuracy by 3% on the validation set and from 80% to 100% accuracy on the pictures I added from the internet.

To add more data to the data set, I used three techniques:
* **Flipping some of the images:** Some of the images in the set can be flipped and remain the same or change into anther traffic sign.

![alt text][image2]
![alt text][image3]

* **Rotating & scaling the images:** I groped those two techniques together because they are part of the same family - affine transformations. In order to change the image a bit the jitter_image function in the notebook chooses the scale and rotate factor and applies them to the image. The ranges were decided based on the ranges that were used in the article above for the assumption that the net will still recognize the sign after the transformation.
![alt text][image4]


The difference between the original data set and the augmented data set is that the augmented data set contains 88,225 more images which can improve learning in classes that didn't had enough examples.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 26x26x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 13x13x6 				|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 9x9x16 	|
| RELU					|												|
| Convolution 1x1     	| 1x1 stride, valid padding, outputs 9x9x8 	|
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 7x7x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x8				|
| Fully connected		| outputs 1x128        									|
| Fully connected		| outputs 1x43        									|
| Softmax				| outputs the logits probabilities										|

![alt text][image5]

#### 3. Training method and net's hyperparameters.

To train the model, I used the cross entropy error function with Adam optimizer.
The training parameters are:
* Learning rate: 0.002
* Batch size: 80
* Epoches: 10

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.957
* test set accuracy of 0.938

The first architecture that was chosen was LeNet which got accuracy of ~85% on the none augmented data set. I chose LeNet because its the basis of most architectures I've seen and its seems to work well in classifying images.
In order to improve its accuracy, I've added more convolution layers between the pooling layers in hope that it will help the network classify the images better.
I added a two 3x3 convolution layers after the existing 5x5 convolution layers and then, after the second convolution layer I added 1x1 convolution layer to reduce the number of features before the 3x3 convolution.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6]

![alt text][image7]

![alt text][image8]

![alt text][image9]

![alt text][image10]


The forth image might be difficult to classify because the sign is covered in ice,
the other images were taken in good weather & lighting conditions, because of that
I don't think they'll be hard to recognize.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Yield      		| Yield  									|
| Stop     			| Stop 										|
| 30 km/h					| 30 km/h											|
| Ice/Snow	      		| Ice/Snow						 				|
| Priority Road			| Priority Road	     							|

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93%

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook (has the prefix "In [19]:").

![alt text][image11]

![alt text][image12]

![alt text][image13]

![alt text][image14]

![alt text][image15]

As we can see network is pretty certain on all of the signs except for the "beware of ice/snow" sign which has probability of 52%, the second sign the network suggested is also an warning sign which has two black lines in the middle of it.
When looking on data distribution we can see that this sign (with classId of 30) has fewer examples then most of the other signs which can justify the low probability that image got.
