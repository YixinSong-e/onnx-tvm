#!/usr/bin/env python
# coding=utf-8
import torch
import torchvision.models as models

#resnet169 = models.densenet169(pretrained=True).cuda()
inception_v3 = models.inception_v3(pretrained=True).cuda()
dummy_input = torch.randn(1, 3, 224, 224, device='cuda')
input_names = ['data']
output_names = ['outputs']

torch.onnx.export(inception_v3, dummy_input, f='inception_v3.onnx', verbose=True, input_names=input_names,
                                   output_names=output_names, opset_version=10)  # generate onnx model of 244M
