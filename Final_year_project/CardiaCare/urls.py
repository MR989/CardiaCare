
from django.contrib import admin
from django.urls import path
from CardiaCare import views
urlpatterns = [
    path("",views.index,name='home'),
    path("result",views.result,name='result'),
]
