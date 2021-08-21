from django.db import models

# Create your models here.

class xRayy(models.Model):
    xray_img = models.ImageField(upload_to='xray/')
    output_one = models.ImageField(upload_to='output/',default='',null=True,blank=True)
    
