from django.shortcuts import render, HttpResponse, redirect
from .models import xRayy
from .forms import xRayForm

import numpy as np
import pandas as pa
import cv2
import gc
import os
import seaborn as sn 
# import matplotlib.pyplot as plt

from InceptionV3.Neural_Visuals import Neural_Visuals

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

model = tf.keras.models.load_model('Models/covid_model.h5')

# Create your views here.

def name(request):
    xrayy=xRayy.objects.all()
    return render(request, "./index.html", {'xRayForm': xRayForm(), 'xrayy':xrayy})

def xRay(request):
    global model
    img = request.FILES['xray_img']    
    path = "D:/GitHub/ImagineHacks/media/xray/"+str(img)
    inp = xRayy.objects.create(xray_img=img)
    pkk=inp.pk
    obj = Neural_Visuals()
    output = obj.visualize(path,model)
    out_path = 'D:/GitHub/ImagineHacks/media/output/'
    cv2.imwrite(os.path.join(out_path , str(img)), output[0])
    label = output[1]
    data_out_path = str(out_path + str('/')+str(img))
    xRayy.objects.filter(pk=pkk).update(output_one=data_out_path)
    out_label = ''
    if label == 0:
        out_label = 'The person has covid'
    elif label == 1:
        out_label = 'The person is not infected'
    else:
        out_label = 'The person has phenemonia'
    img_inp_path = str("/media/xray/" + str(img))
    img_out_path = str("/media/output/" + str(img))
    context = {
        'out_label':out_label,
        'pkk':pkk,
        'inp':img_inp_path,
        'out':img_out_path,
    }
    return render(request, "./xray.html", context=context)