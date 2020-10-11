# maskipy 
## A multi-class classifier to detect unmasked, improperly masked and properly masked human faces.

A lot of businesses have been trying to leverage Machine Learning to maintain thier foothold in today's world ravaged by COVID-19. Food delivery platforms have deployed AI-driven systems, which detects if the delivery partner is wearning a mask, like the one [here](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/). However, I often find the parters wearning their masks on the chin or mouth-chin which compromises consumer safety.

Maskipy improves upon the same by performing multilabel classification of the face data to not only identify if someone is wearing a mask but if they are wearing it properly.

The model is trained using the [MaskedFace-Net dataset](https://github.com/cabani/MaskedFace-Net) and the [source dataset (Flickr-Faces-HQ)](https://github.com/NVlabs/ffhq-dataset), using which the MaskedFace-Net dataset was generated, to train the model for unmasked faces. However, it was ensured that faces weren't reused between the three classes (unmasked, imroperly masked, properly masked) to ensure proper generalization.

The classification model comprises of a sequential keras model head stacked upon headless MobileNetV2. Since the latter is high performance, low-latency model proficient in classification tasks designed of mobile devices, it fits the potential usecase. The head is trained to obtain desired classification as in the article linked earlier.

The driver file uses [OpenCV frontal face detector](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) to extract face data from the camera feed. The model predicts the class based on the same which is displayed on the screen above the bounding box(es).

### Installation and Usage:
Install the packages listed in requirements.txt and run classifier_driver.py in Python3.
The model can be retrained by downloading both (MaskedFace-Net dataset and Flickr-Faces-HQ) datasets as in the dataset_sample and running maskipy_train.py in Python3.

### Demo:
![](demo.gif)
