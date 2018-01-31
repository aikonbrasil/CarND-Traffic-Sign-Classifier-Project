# **Traffic Sign Recognition** 

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

[image1]: ./images_writeup/image_1.png "Visualization1"
[image2]: ./images_writeup/image_2.png "Visualization2"
[image3]: ./images_writeup/bar_1.png "bar_Train"
[image4]: ./images_writeup/bar_2.png "bar_Test"
[image5]: ./images_writeup/bar_3.png "bar_validation"
[image6]: ./images_writeup/web_signs.PNG "web signs"
[image7]: ./images_writeup/prediction_web_signs.PNG "Prediction of web signs"
[image8]: ./images_writeup/softmax_probabilities.PNG "softmax probabilities"
[image9]: ./images_writeup/accuracy_achieved.png "Accuracy achieved"
[image10]: ./images_writeup/pre_processing.png "Pre processing information process"
[image11]: ./images_writeup/pre_processing2.png "Pre processing information process"
[image12]: ./images_writeup/adding_samples.png "Adding Samples"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/aikonbrasil/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data samples are not uniform for all signs. I mean that in some cases the number of samples of an specific sign is near 2000 and in other cases it is less than 100. This issue is a sensible characteristic,  specially for Train data samples group.

in the next image it is possible to check the distribution of Train samples

![alt text][image3]

in the next image it is possible to check the distribution of Test samples

![alt text][image4]

in the next image it is possible to check the distribution of Validation samples

![alt text][image5]

### Design and Test a Model Architecture

#### 1. Describe how you pre-processed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each pre-processing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to normalize each sample image between -1 and 1. Here, I also converted each pixel values to 32-bit data type. After that I also converted the images to grayscale to facilitate processing focusing in just one channel color. As a complement I performed the next image transformations:
* image translation (soft random shift) in order to create a duplicated information with random translation in order to create diversity in train samples dataset, 
* image rotation in order to create a duplicated information with random rotation between -12° and 12º in order to create diversity in Train dataset,
* image perspective transformation (zoom) in order to create a duplicated information with random perspective. I also added a white noise to the image.
* finally I used a script for duplicating custom images to obtain homonegenous number of samples in the train information. In this case the condition was that unless each traffic sign should have 1000 samples for the test dataset.

Here is an example of a traffic sign image before and after grayscaling and normalizing transformation.

![alt text][image10]


Here is an example of rotation, translation, zoom, and white noise adding to an images. Parameters of each transformation were chosen randomly.

![alt text][image11]

I decided to generate additional data because the distribution information indicates that the quantity of some classes could be small in comparison of other ones. I increased the samples of some classes in order to provide similar chances to all samples during learning process.

To add more data to the the data set I used typical transformation image techniques (described before). The number of samples to achieve the number of 1000 samples per each sign dataset was calculated and displayed in the notebook. It could be visualized in the next image.

![alt text][image12]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution (filter with 5x5x1x6 )     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution (filter with 5x5x6x16 )       | 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten. Input = 5x5x16. Output = 400 |    |
| Fully connected		| input=400, output=120       									|
| RELU					|												|
| Fully connected		| input=120, output=84       									|
| RELU					|												|
| Fully connected		| input=84, output=43       									|

I have added tunable keep_prob, the best value were dropout_custome = 0.8, this dropout layer was inserted after each activation layer (to avoid network over fitting).

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The optimizer used is based on Cross entropy and softmax which are tools provided by TensorFlow. Batch size=128 because in other cases the training process indicated an accuracy less than 9.3. Other important parameter was the number of epochs, that is why that in order to guarantee the convergence of Neural network learning process we used 100 epochs ( Amazon AWS was used in this scenario). I have added also a tunable keep_prob, this dropout layer was inserted after each activation layer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
In spite that I run 100 epochs the training process indicates that validation accuracy was more than 0.93 in the epoch number 15 (the first time that validation accuracy was more than 0.93). When training process finished, we evaluated the accuracy, the value of accuracy was 0.932.

My final model results were:
* training set accuracy of 1
* validation set accuracy of 0.953
* test set accuracy of 0.932

Finally I choosed the LeNet model adding a tunable keep_prob using the variable dropout_custome. The value that indicates better results was dropout_custome = 0.8.

In order to show that the model is working sucessfully, the next image shows a validation accuracy = 0.953 (more than 0.93) and test accuracy = 0.932

![alt text][image9]
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image6] 

I included five new German Traffic signs that I found on the web. They are  visualized and described in following lines. An importan information is that these new images are saved in the folder 'custome_signs_web/'

* 'speed_20km_0.jpg'
* 'no_passing_9.jpg'
* 'yield_13.jpg'
* 'slipperyroad_23.jpg'
* 'keep_right_38.jpg'

The main particularity in all cases is the image resolution. In case of Yield signal it included a custom background of clouds.

![alt text][image7] 


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|    Label     |
|:---------------------:|:-------------------------------:|:----------|
| Speed limit (20km/h)      		| Speed limit (20km/h)   	 |  0								|   
| No passing     			| No passing  										| 9								|  
| Yield					| Yield											|  13								|  
| Slippery road	      		| Slippery road				 				| 23								|  
| Keep right			| Keep right      							|  38								|  


In this step I documented the performance of the model that was trained before, this performance is evaluated using 5 figures that were obtained from the Internet. The accuracy results of predictions were excellent in all cases. So, accuracy was 100% for these samples.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook. I created some bars in order to show prediction results for each image traffic sign that were obtained from the web.

![alt text][image8] 

For the first image, the model is sure that this is a Speed limit (20km/h) sign (probability of 0.9909), and the image does contain a Speed limit (20km/h)  sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit (20km/h)   									| 
| .00     				| Speed limit (120km/h)  										|
| .00					| Slippery road									|
| .00	      			| Speed limit (80km/h)				 				|
| .00				    | Roundabout mandatory     							|


For the second image, the model is sure that this is a No passing sign (probability of 0.9975), and the image does contain a No passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No passing  									| 
| .00     				| No entry										|
| .00					| End of no passing								|
| .00	      			| Priority road		 				|
| .00				    | Roundabout mandatory							|

For the third image, the model is sure that this is a No passing sign (probability of 0.9975), and the image does contain a No passing sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Yield 									| 
| .00     				|Go straight or right										|
| .00					| No vehicles								|
| .00	      			| Bumpy road		 				|
| .00				    | Speed limit (70km/h)  							|

For the fourth image, the model is sure that this is a No passing sign (probability of 0.9975), and the image does contain a No passing sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Slippery road								| 
| .00     				| Dangerous curve to the right									|
| .00					| Dangerous curve to the left							|
| .00	      			| Wild animals crossing	 				|
| .00				    | Beware of ice/snow							|


For the fifth image, the model is sure that this is a No passing sign (probability of 0.9975), and the image does contain a No passing sign. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Keep right									| 
| .00     				| Slippery road										|
| .00					| Speed limit (50km/h)								|
| .00	      			| Speed limit (80km/h) 				|
| .00				    | Speed limit (120km/h)							|

### Note
It looks that the prediction is ok, however, I think that in some traffic signs the deep learning model will fail. I were extremely optimistic in all web images.
