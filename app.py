import streamlit as st
from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import segmentation_models as sm
from segmentation_models import Unet

st.title('Brain MRI Tumour Segmentation')
mri_image = st.file_uploader("Upload Brain MRI Image")

if mri_image is not None:
  img = Image.open(mri_image)
  st.image(img)

mri_mask = st.file_uploader("Upload Brain MRI mask")

if mri_mask is not None:
  image_mask = Image.open(mri_mask)
  st.image(image_mask)

model = Unet('resnet34', encoder_weights='imagenet', classes=2, activation='softmax', input_shape=(256,256,3), encoder_freeze=True)
model.load_weights("/content/my_best_model.epoch13-iou_score0.94.hdf5")

image = np.array(img)

predicted = model.predict(image[np.newaxis,:,:,:])
predicted = tf.argmax(predicted, axis=-1) 
predicted = tf.expand_dims(predicted, axis=-1)
predicted = predicted[0,:,:,0]

if st.button('Show predicted mask'):
  predicted_image = Image.fromarray((np.array(predicted) * 255).astype(np.uint8))
  st.image(predicted_image)

def IoU_score(result1, result2):
  intersection = np.logical_and(result1, result2)
  union = np.logical_or(result1, result2)
  iou_score = np.sum(intersection) / np.sum(union)
  return iou_score

if st.button('Calculate IoU Score'):
  iou_score = IoU_score(predicted, image_mask)
  st.write(iou_score)
