import cv2
import numpy as np
import pandas as pa
import cv2
import gc
import os
import seaborn as sn 
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D,Dropout,Flatten,Dense,Input
from tensorflow.keras.optimizers import Adam,RMSprop
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

from imutils import paths
import tensorflow as tf


class Neural_Visuals:
    def __init__(self):
        self.image_original = None
        self.orig = None
        
        
    def visualize(self,image_path,model):
        GRAD_CAM = True
        
        file = image_path
        self.image_original = cv2.imread(file)
        #plt.imshow(self.image_original)
        #plt.show()
            
        self.orig = cv2.cvtColor(self.image_original, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(self.orig, (256, 256))
        dataXG = np.array(resized) / 255.0
        dataXG = np.expand_dims(dataXG, axis=0)
            
        preds = model.predict(dataXG)
        i = np.argmax(preds[0])
        print(i, preds)
            
        if GRAD_CAM == True:
            out_image = self.grad_cam_visulaization(dataXG,i,model)
        
        
        output = [out_image,i]
        
        return output
        
        
    def grad_cam_visulaization(self,dataXG,i,model):
        cam = GradCAM(model=model, classIdx=i, layerName='mixed10')
        heatmap = cam.compute_heatmap(dataXG)
        #plt.imshow(heatmap)
        #plt.show()
        
        heatmapY = cv2.resize(heatmap, (self.orig.shape[1], self.orig.shape[0]))
        heatmapY = cv2.applyColorMap(heatmapY, cv2.COLORMAP_HOT)  # COLORMAP_JET, COLORMAP_VIRIDIS, COLORMAP_HOT
        imageY = cv2.addWeighted(heatmapY, 0.5, self.image_original, 1.0, 0)
        print(heatmapY.shape, self.orig.shape)# draw the orignal x-ray, the heatmap, and the overlay together
        #output = np.hstack([self.orig, heatmapY, imageY])
        #fig, ax = plt.subplots(figsize=(20, 18))
        #ax.imshow(np.random.rand(1, 99), interpolation='nearest')
        #plt.imshow(imageY)
        #plt.show()
        
        return imageY
    
    ## Soon
    def grad_cam_plusplus_visulaization(self):
        pass
    
class GradCAM:
    def __init__(self, model, classIdx, layerName=None):
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        if self.layerName is None:
            self.layerName = self.find_target_layer()
            
    def find_target_layer(self):
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:
                return layer.name
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")
        
    def compute_heatmap(self, image, eps=1e-8):
        
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                self.model.output])
        
        with tf.GradientTape() as tape:
            inputs = tf.cast(image, tf.float32)
            (convOutputs, predictions) = gradModel(inputs)
            loss = predictions[:, self.classIdx]
            
        
        grads = tape.gradient(loss, convOutputs)
        
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        castGrads = tf.cast(grads > 0, "float32")
        guidedGrads = castConvOutputs * castGrads * grads
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        (w, h) = (image.shape[2], image.shape[1])
        heatmap = cv2.resize(cam.numpy(), (w, h))
        numer = heatmap - np.min(heatmap)
        denom = (heatmap.max() - heatmap.min()) + eps
        
        heatmap = numer / denom
        heatmap = (heatmap * 255).astype("uint8")
        return heatmap