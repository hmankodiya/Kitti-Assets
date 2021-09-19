import tensorflow as tf
import numpy as np
import cv2

use_unet = True

def modelPredictWrapper(model, inputImg):
    
    modelOutput = model.predict(np.expand_dims(inputImg,0))# * 640 * 0.3

    if use_unet:
        zeros = np.zeros([modelOutput.shape[0], modelOutput.shape[1], modelOutput.shape[2], 1])
        modelOutput = np.concatenate((modelOutput, zeros), axis=3)
        modelOutput[:,:,:,[0,1,2]] = modelOutput[:,:,:,[1,2,0]]
    
    return modelOutput

def displayOutput(output, dim=0):

    output = np.squeeze(output)
    outputTransformed = np.transpose(  output,    axes=[1,0,2])
    outputTransformed = np.clip(outputTransformed / np.max(outputTransformed) * 255, 0, 255).astype('uint8')
    return outputTransformed

def pred_output(model,inputImg):

    rawImage = cv2.resize(inputImg, (640,192))
    inputImgOrig  = cv2.resize(inputImg, (640,192))
    inputImg  = np.transpose(inputImgOrig.astype('float32'), axes=[1,0,2])
    output = modelPredictWrapper(model, inputImg)
   
    outputDisplay = displayOutput(output,2)
    overlayedImage = cv2.addWeighted(inputImgOrig, 0.8, outputDisplay, 0.2, 0)
    return (outputDisplay,overlayedImage)