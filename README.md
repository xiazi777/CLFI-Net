# Enhancing Fine-Grained Visual Classification via Curriculum Learning and Global-Local Feature Interaction
Authors: Xueqing Zhang, Shuo Wang, Fengjuan Feng, Jianlei Liu

# Dependencies and requirements
Python 3.9 <br>
Install PyTorch 1.8 or above and other dependencies (e.g., torchvision).<br>
For packages, see requirements.txt.<br>
For pip users, please type the command pip install -r requirements.txt.

## Datasets
Here are the download links for each dataset, and the division of the datasets is detailed in the article.
CUB_200_2011 (CUB) - <https://www.vision.caltech.edu/datasets/cub_200_2011/>
Stanford Cars (CAR) - <http://ai.stanford.edu/~jkrause/cars/car_dataset.html>.Note:This is the official download link, but it is possible that the link is not working. If it does not work, please visit: https://pan.baidu.com/s/1zfCC6jeZpmrsH4cXaZLquQ?pwd=6666. Please note that this is not the only way to get it.
FGVC-Aircraft (AIR) - <https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/>
ALGAE - The algae dataset is still being collected and may be publicly available in the future

# Descriptions of key algorithms


# CLFI-Net Training and evaluation
The training code for CLFI-Net is train.py and the test code is test.py<br>
Train the CLFI-Net model: python train.py<br>
Test the CLFI-Net model: python test.py<br>
Note:Please modify the code accordingly to the storage location.

# Reference
if this code is helpful to you, please cite as the following format:
@ARTICLE{1,
  author={Zhang, Xueqing and Wang, Shuo and Feng, Fengjuan and Liu, Jianlei},
  journal={The Visual Computer}, 
  title={Enhancing Fine-Grained Visual Classification via Curriculum Learning and Global-Local Feature Interaction}
}



