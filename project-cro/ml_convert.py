#!/usr/local/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import coremltools

model_data = ('./10_000_twohidden/model.json', './10_000_twohidden/model.h5')
image_scale = 1/255.
class_labels = ['A', 'B', 'C', 'Č', 'Ć', 'D', 'Đ', 'E',
                'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
                'N', 'O', 'P', 'R', 'S', 'Š', 'T', 'U',
                'V', 'Z', 'Ž', 'X', 'Y', 'W', 'Q', '-']
coreml_model = coremltools.converters.keras.convert(model_data,
                                                   input_names='image',
                                                   image_input_names='image',
                                                   output_names = ['letter'],
                                                   class_labels = class_labels,
                                                   image_scale = image_scale)
coreml_model.author = 'Filip Gulan'
coreml_model.license = 'FER'
coreml_model.short_description = 'Model used for classifying Croatian hand written letters'
coreml_model.input_description['image'] = 'Grayscale image 30x30 of hand written letter'
coreml_model.output_description['letter'] = 'Predicted letter'
coreml_model.save('LetterClass_image.mlmodel')