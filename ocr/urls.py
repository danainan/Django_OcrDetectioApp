from django.urls import path
from .views import camera_feed, index

urlpatterns = [
    path('', index, name='index'),
    path('camera/', camera_feed, name='camera_feed'),
]