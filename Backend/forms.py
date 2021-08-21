from django import forms
from .models import *

class xRayForm(forms.ModelForm):
    class Meta:
        model = xRayy
        fields = ["xray_img",]
