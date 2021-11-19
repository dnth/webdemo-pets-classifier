import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

import os
for root, dirs, files in os.walk(r'sample_image/'):

title = "Pet Breed Classifier"
description = "A pet breed classifier trained on the Oxford Pets dataset"
interpretation='default'
examples = ["sample_images/"+file for file in files] 
article="<p style='text-align: center'><a href='https://dicksonneoh.com' target='_blank'>Blog post</a></p>"
enable_queue=True

gr.Interface(fn=predict,inputs=gr.inputs.Image(shape=(512, 512)),outputs=gr.outputs.Label(num_top_classes=3),title=title,description=description,article=article,examples=examples,interpretation=interpretation,enable_queue=enable_queue).launch()