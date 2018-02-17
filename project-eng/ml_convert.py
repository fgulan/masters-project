#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import coremltools
import numpy as np

model_data = ('./model.json', './weights/weights-79-0.947356.hdf5')
image_scale = 1/255.

class_labels = ['A', 'B', 'C', 'D', 'E', 
                'F', 'G', 'H', 'I', 'J', 
                'K', 'L', 'M', 'N', 'O', 
                'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']
coreml_model = coremltools.converters.keras.convert(model_data,
                                                   input_names='image',
                                                   image_input_names='image',
                                                   output_names = ['letter'],
                                                   class_labels = class_labels,
                                                   image_scale = image_scale)
coreml_model.author = 'Filip Gulan'
coreml_model.license = 'FER'
coreml_model.short_description = 'Model used for classifying English handwritten letters'
coreml_model.input_description['image'] = 'Grayscale image 28x28 of hand written letter'
coreml_model.output_description['letter'] = 'Predicted letter'
coreml_model.save('LetterClass_eng_image.mlmodel')