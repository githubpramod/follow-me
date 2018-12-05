# RoboND Term 1 Project 4 - Follow-Me
## Author - Pramod Kumar

[//]: # (Referenced Images)
[title_image]: ./images/1.png
[image1]: ./images/2.png
[image2]: ./images/3.png
[image3]: ./images/4.png
[image4]: ./images/5.png
[image5]: ./images/6.png

![Model Performance][title_image]

Fully convolution networks created final segmentation result which is shown in image.

### Fully Convolutional Networks

The fundamental thought behind a fully convolutional network is that it doesn't have any dense fully connected layers that's why 
the majority of its layers are convolutional layers. FCNs, rather than utilizing fully connected layers toward the end, commonly 
utilized for classification, utilize convolutional layers to classify every pixel in a image.

So the last output layer will be indistinguishable height and width from the input image, however the quantity of channels will 
be equivalent to the number of classes. In case we're arranging every pixel as one of fifteen distinct classes, at that point the 
last yield layer will be height x width x 15 classes. FCNs includes fundamentally 3 sections: an encoder, a 1x1 convolution and 
a decoder. The 1x1 convolution has little height and width however considerably bigger depth because of the utilization of 
channels in the preceding encoder part. The 1x1 convolution's height and width will be the equivalent as the layer preceding 
it. The purpose behind utilizing 1x1 convolution is that it goes about as a completely associated layer, yet its significant 
favourable position is that it holds spatial data of the considerable number of pixels.

The second part of the system is known as the decoder. it is developed of layers of transposed convolutions whose yields increment 
the height and width of the input while diminishing it's profundity. Skip connections are likewise utilized at all points in the 
decoder which for this situation connects the contribution to the encoder layer of a similar size with the decoded layer. Skip 
associations permit the system to hold data from earlier layers that were lost in resulting convolution layers. Skip layers 
utilize the yield of one layer as the input to another layer. By utilizing data from numerous image sizes, the model holds more 
data through the layers and is in this way ready to settle on more precise segmentation decisions. On the off chance that there 
was some information that was captured in the underlying layers and was required for remaking amid the up-testing done utilizing 
the FCN layer. In the event that we would not have utilized the skip architecture that information would have been lost (or should 
state would have turned excessively conceptual for it to be utilized further ). So the information that we had in the essential 
layers can be encouraged expressly to the later layers utilizing the skip architecture.


3 encoder layers, 1X1 convolution and 3 decoder layers have been chosen for this model.
Output shape of each layer is:

	Input shape     (?, 160, 160, 3)--------|
	encoder_1_layer (?, 80, 80, 32)------|  |
	encoder_2_layer (?, 40, 40, 64)---|  |  |
	encoder_3_layer (?, 20, 20, 128)  |  |  |
	conv_1_1_layer  (?, 20, 20, 128)  |  |  |
	decoder_1_layer (?, 40, 40, 128)--|  |  |
	decoder_2_layer (?, 80, 80, 64)------|  |
	decoder_3_layer (?, 80, 80, 64)---------|
	Output Layer    (?, 160, 160, 3)


### Hyper parameters

The steps per epoch was constantly set to the aggregate number of images in the training set separated by the batch size so  that each 
each epoch was around one go through all the training images. I have utilized a batch size of 32, or, in other words.The main 
motivation behind using a large batch size was achieving stability  in the training process. At that point I utilized two 
learning rates of 0.0001 and 0.01 separately. I at first kept running with 30 epoch with 0.0001 LR and achieved a total IOU 
score of 0.386098110731.So for my last model, I chose to increase my learning rate to 0.01 and see in the event that it 
accelerates the convergence. On the off chance that sooner or later, I would have seen that combination has been falling apart, 
my plan was to utilize a decaying learning rate. Based on these observation I decided to make a model on 115 epoch with 0.01 
learning rate and fortunately it worked and I got score 0.428192393756.

At epoch 115 LR 0.0001:
![LR 0.01][Images1]

At epoch 115 LR 0.01:
![LR 0.01][Images2]

#### IOU scores 

# Scores for while the quad is following behind the target.
number of validation samples intersection over the union evaulated on 542
average intersection over union for background is 0.9965457222108018
average intersection over union for other people is 0.4018328779965078
average intersection over union for the hero is 0.9243780427567977
number true positives: 539, number false positives: 0, number false negatives: 0

# Scores for images while the quad is on patrol and the target is not visable
number of validation samples intersection over the union evaulated on 270
average intersection over union for background is 0.9895129469709045
average intersection over union for other people is 0.792995878343493
average intersection over union for the hero is 0.0
number true positives: 0, number false positives: 40, number false negatives: 0

# This score measures how well the neural network can detect the target from far away
number of validation samples intersection over the union evaulated on 322
average intersection over union for background is 0.9970756497071918
average intersection over union for other people is 0.4792198829927118
average intersection over union for the hero is 0.22224049493137998
number true positives: 119, number false positives: 1, number false negatives: 182

# Sum all the true positives, etc from the three datasets to get a weight for the score
weight:0.7468785471055619


# The IoU for the dataset that never includes the hero is excluded from grading
final_IoU:0.573309268844

# And the final grade score is 
final_score:0.428192393756


### Performance on simulator 

While testing the model in the simulator, it performed remarkably well to detect the hero from a reasonably large distance and 
once it detects and zeros in on the hero, it never lost sight of it. It has a 100% success ratio when it is close to the hero. 
This is evident from the 539 true positives we obtained from the model evaluation.
The models performance could be improved by using more training data, particularly when the hero is far away. Other than that 
the model performs really well.
I believe this model could be used to also idenify and track multiple objects one at a time where each object zeros in to the 
other object and once it reaches the other object, the drone leaves the first object and follows the 2nd one. This can have a 
huge military application.
This model could be used to also identify and track and another object (based on suggestions of using this model for tracking a 
dog or car) but training data specific too these objects would be required.
