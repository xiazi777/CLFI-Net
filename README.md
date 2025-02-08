# Enhancing Fine-Grained Visual Classification via Curriculum Learning and Global-Local Feature Interaction
Authors: Xueqing Zhang, Shuo Wang, Fengjuan Feng, Jianlei Liu

# Preparation
Python 3.9 <br>
For packages, see requirements.txt.
##Getting started
Install PyTorch 1.8 or above and other dependencies (e.g., torchvision).<br>
For pip users, please type the command pip instal(requirements.txt).
## Datasets
CUB_200_2011 (CUB) - <https://www.vision.caltech.edu/datasets/cub_200_2011/>

Stanford Cars (CAR) - <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>.note:This is the official download link, but it is possible that the link is not working. If it does not work, please visit: https://pan.baidu.com/s/1zfCC6jeZpmrsH4cXaZLquQ?pwd=6666. Please note that this is not the only way to get it.

FGVC-Aircraft (AIR) - <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>

ALGAE - The algae dataset is still being collected and may be publicly available in the future

# CLFI-Net Training and evaluation
A one train example is provided.<br>
Train the CLFI-Net model: python train.py


