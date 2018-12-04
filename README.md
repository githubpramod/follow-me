# follow-me
[![Udacity-Robotics NanoDegree Program](https://s3-us-west-1.amazonaws.com/udacity-robotics/Extra+Images/RoboND_flag.png)](https://www.udacity.com/robotics)

## RoboND-DeepLearning-Project ##

Train a model using deep neural network for identify the target and track it. This approach is used for "follow me" application
and this kind of approach can be used in autonomous vehicle for ACC application.

[image_0]: ./docs/misc/sim_screenshot.png
![alt text][image_0] 

## Setup Preparation 
 
** First make a clone the repository of RoboND-DeepLearning **
```
$ git clone https://github.com/udacity/RoboND-DeepLearning.git
```

**Download the training and validation data**

Save the following three files into the data folder of the cloned repository. 

[Training Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/train.zip) 

[Validation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Lab/validation.zip)

[Sample Evaluation Data](https://s3-us-west-1.amazonaws.com/udacity-robotics/Deep+Learning+Data/Project/sample_evaluation_data.zip)

**Download the QuadSim binary**

Download the latest QuadSim simulator for interface it with neural network for this project and previous version might not work 
for this project which is used for control lab.

The simulator binary can be downloaded [here](https://github.com/udacity/RoboND-DeepLearning/releases/latest)

**Install Dependencies**

Python 3 and jupyter notebook is needed for this project so install these software. The best way to install these software is 
if you can download the RoboND-Python-Starterkit and qownload [here][RoboND-Python-Starterkit](https://github.com/udacity/RoboND-Python-StarterKit)

These below packages and frameworks must be available in your system:
* Python 3.x
* Tensorflow 1.2.1
* NumPy 1.11
* SciPy 0.17.0
* eventlet 
* Flask
* h5py
* PIL
* python-socketio
* scikit-image
* transforms3d
* PyQt4/Pyqt5

## Implement the Segmentation Network
1. Download the training dataset from above and extract to the project `data` directory.
2. Implement your solution in model_training.ipynb
3. Train the network locally, or on [AWS](https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/b5a6a0db-f238-432b-a876-17b641268ca9).
4. Continue to experiment with the training data and network until you attain the score you desire.
5. Once you are comfortable with performance on the training dataset, see how it performs in live simulation!

## Collecting Training Data ##
A simple training dataset has been provided in this project's repository. This dataset will allow you to verify that your 
segmentation network is semi-functional. However, if your interested in improving your score,you may want to collect additional 
training data. To do it, please see the following steps.

The data directory is organized as follows:
```
data/runs - contains the results of prediction runs
data/train/images - contains images for the training set
data/train/masks - contains masked (labeled) images for the training set
data/validation/images - contains images for the validation set
data/validation/masks - contains masked (labeled) images for the validation set
data/weights - contains trained TensorFlow models

data/raw_sim_data/train/run1
data/raw_sim_data/validation/run1
```

### Training Set ###
1. Run QuadSim
2. Click the `DL Training` button
3. Set patrol points, path points, and spawn points. **TODO** add link to data collection doc
3. With the simulator running, press "r" to begin recording.
4. In the file selection menu navigate to the `data/raw_sim_data/train/run1` directory
5. **optional** to speed up data collection, press "9" (1-9 will slow down collection speed)
6. When you have finished collecting data, hit "r" to stop recording.
7. To reset the simulator, hit "`<esc>`"
8. To collect multiple runs create directories `data/raw_sim_data/train/run2`, `data/raw_sim_data/train/run3` 
   and repeat the above steps.


### Validation Set ###
To collect the validation set, repeat both sets of steps above, except using the directory `data/raw_sim_data/validation` 
instead rather than `data/raw_sim_data/train`.

### Image Preprocessing ###
Before the network is trained, the images first need to be undergo a preprocessing step. The preprocessing step transforms 
the depth masks from the sim, into binary masks suitable for training a neural network. It also converts the images from .png 
to .jpeg to create a reduced sized dataset, suitable for uploading to AWS. 

To run preprocessing:
```
$ python preprocess_ims.py
```
**Note**: If your data is stored as suggested in the steps above, this script should run without error.

**Important Note 1:** 

Running `preprocess_ims.py` does *not* delete files in the processed_data folder. This means if you leave images in processed 
data and collect a new dataset, some of the data in processed_data will be overwritten some will be left as is. It is 
recommended to **delete** the train and validation folders inside processed_data(or the entire folder) before running 
`preprocess_ims.py` with a new set of collected data.

**Important Note 2:**

The notebook, and supporting code assume your data for training/validation is in data/train, and data/validation. After you run 
`preprocess_ims.py` you will have new `train`, and possibly `validation` folders in the `processed_ims`.Rename or move 
`data/train`, and `data/validation`, then move `data/processed_ims/train`, into `data/`, and `data/processed_ims/validation`also into `data/`

**Important Note 3:**

Merging multiple `train` or `validation` may be difficult, it is recommended that data choices be determined by what you include 
in `raw_sim_data/train/run1` with possibly many different runs in the directory. You can create a temporary folder in `data/` 
and store raw run data you don't currently want to use, but that may be useful for later. Choose which `run_x` folders to include 
in `raw_sim_data/train`, and `raw_sim_data/validation`, then run  `preprocess_ims.py` from within the 'code/' directory to generate 
your new training and validation sets. 


## Training, Predicting and Scoring ##
With your training and validation data have been generated or downloaded from the above section of this repository, you are 
free to begin working with the neural net.

**Note**: Training CNNs is a very compute-intensive process. If your system does not have a recent Nvidia graphics card, with 
[cuDNN](https://developer.nvidia.com/cudnn) and [CUDA](https://developer.nvidia.com/cuda) installed , you may need to perform 
the training step in the cloud. Instructions for using AWS to train your network in the cloud may be 
found [here](https://classroom.udacity.com/nanodegrees/nd209/parts/c199593e-1e9a-4830-8e29-2c86f70f489e/modules/cac27683-d5f4-40b4-82ce-d708de8f5373/lessons/197a058e-44f6-47df-8229-0ce633e0a2d0/concepts/b5a6a0db-f238-432b-a876-17b641268ca9)

### Training your Model ###
**Prerequisites**
- Training data is in `data` directory
- Validation data is in the `data` directory
- The folders `data/train/images/`, `data/train/masks/`, `data/validation/images/`, and `data/validation/masks/` should exist 
  and contain the appropriate data

To train complete the network definition in the `model_training.ipynb` notebook and then run the training cell with appropriate 
hyper-parameters selected.

After the training run has completed, your model will be stored in the `data/weights` directory as an 
[HDF5](https://en.wikipedia.org/wiki/Hierarchical_Data_Format) file, and a configuration_weights file. As long as they are both 
in the same location, things should work. 

**Important Note** 
'validation' directory is used to store data that will be used during training to produce the plots of the loss, and help 
determine when the network is overfitting your data. 

**sample_evalution_data** 
'sample_evalution_datadirectory' contains data specifically designed to test the networks performance on the FollowME task. 
In sample_evaluation data are three directories each generated using a different sampling method. The structure of these 
directories is exactly the same as `validation`, and `train` datasets provided to you. For instance `patrol_with_targ` contains 
an `images` and `masks` subdirectory. If you would like to the evaluation code on your `validation` data a copy of the it should 
be moved into `sample_evaluation_data`, and then the appropriate arguments changed to the function calls in the `model_training.ipynb` 
notebook.

The notebook has examples of how to evaulate your model once you finish training. Think about the sourcing methods, and how the 
information provided in the evaluation sections relates to the final score. Then try things out that seem like they may work. 

## Scoring ##

To score the network on the Follow Me task, two types of error are measured. First the intersection over the union for the 
pixelwise classifications is computed for the target channel.
In addition to this we determine whether the network detected the target person or not. If more then 3 pixels have probability 
greater then 0.5 of being the target person then this counts as the network guessing the target is in the image. 
We determine whether the target is actually in the image by whether there are more then 3 pixels containing the target in the 
label mask. 
Using the above the number of detection true_positives, false positives, false negatives are counted. 

**How the Final score is Calculated**
The final score is the pixelwise `average_IoU*(n_true_positive/(n_true_positive+n_false_positive+n_false_negative))` on data 
similar to that provided in sample_evaulation_data

**Ideas for Improving your Score**
Collect more data from the sim. Look at the predictions think about what the network is getting wrong, then collect data to 
counteract this. Or improve your network architecture and hyperparameters. 

## Experimentation: Testing in Simulation
1. Copy your saved model to the weights directory `data/weights`.
2. Launch the simulator, select "Spawn People", and then click the "Follow Me" button.
3. Run the realtime follower script
```
$ python follower.py model_weights.h5
```

**Note:** 
If you'd like to see an overlay of the detected region on each frame, simply pass the `--pred_viz` parameter to `follower.py`

