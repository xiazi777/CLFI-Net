# Enhancing Fine-Grained Visual Classification via Curriculum Learning and Global-Local Feature Interaction
Authors: Xueqing Zhang, Shuo Wang, Fengjuan Feng, Jianlei Liu

# Dependencies and requirements
Python 3.9 <br>
Install PyTorch 1.8 or above.<br>
Other dependencies (e.g., torchvision), see requirements.txt.<br>
For pip users, please type the command 
## pip install -r requirements.txt.

## Datasets
Here are the download links for each dataset, and the division of the datasets is detailed in the article.<br>
CUB_200_2011 (CUB) - <https://www.vision.caltech.edu/datasets/cub_200_2011/><br>
Stanford Cars (CAR) - <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>.Note:This is the official download link, but it is possible that the link is not working. If it does not work, please visit: https://pan.baidu.com/s/1zfCC6jeZpmrsH4cXaZLquQ?pwd=6666. Please note that this is not the only way to get it.<br>
FGVC-Aircraft (AIR) - <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/><br>
ALGAE - The algae dataset is still being collected and may be publicly available in the future

# Descriptions of key algorithms
Resnet.py serves as the backbone and is responsible for feature extraction; anchors.py is mainly used for setting and generating anchor frames; dataset.py is responsible for loading datasets and pre-processing; model.py is the core code that defines the structure of the whole model and the forward propagation logic; train.py is responsible for training the model including data loading, loss calculation and optimization update; test.py is used for testing the model performance and evaluating its generalization ability; utils.py provides various auxiliary tool functions to enhance the modularization and optimization of code; and tools.py provides various auxiliary tool functions to enhance the modularization and optimization of code. data loading, loss calculation and optimization update; test.py is used to test the model performance; utils.py provides a variety of auxiliary tool functions.

# Implementations of CLFI-Net
The training code for CLFI-Net is train.py and the test code is test.py.<br>
The implementation of our approach is very simple, just run train.py to complete the end-to-end training without any additional tedious process. During the training process, the system will automatically generate a directory, e.g., cub_resnet50_4, which contains the model parameters saved during the training process, model.pth, as well as information about the training and testing process, such as results_train and results_test, for subsequent analysis and tuning.<br>
## Train the CLFI-Net model: python train.py<br>
Note:Please modify the code accordingly to the storage location.

# Reference
if this code is helpful to you, please cite as the following format:<br>
@ARTICLE{1,<br>
  &emsp author={Zhang, Xueqing and Wang, Shuo and Feng, Fengjuan and Liu, Jianlei},<br>
  &emsp journal={The Visual Computer}, <br>
  &emsp title={Enhancing Fine-Grained Visual Classification via Curriculum Learning and Global-Local Feature Interaction}<br>
}



