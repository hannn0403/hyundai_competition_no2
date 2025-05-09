"""
Copyright (c) 2019-present NAVER Corp.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch.nn as nn
from transformation import TPS_SpatialTransformerNetwork
from feature_extraction import ResNet_FeatureExtractor
from sequence_modeling import BidirectionalLSTM
from prediction import Attention


class OCRModel(nn.Module):
    def __init__(self, opt):
        super(OCRModel, self).__init__()
        self.opt = opt
        self.stages = {'Trans': "TPS Spatial Transformer", 'Feat': "ResNet Feature Extractor",
                       'Seq': "Bidirectional LSTM", 'Pred': "Attention"}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            F=opt.num_fiducial, I_size=(opt.imgH, opt.imgW), I_r_size=(opt.imgH, opt.imgW), I_channel_num=opt.input_channel)

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(opt.input_channel, opt.output_channel)
        self.FeatureExtraction_output = opt.output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size),
            BidirectionalLSTM(opt.hidden_size, opt.hidden_size, opt.hidden_size))
        self.SequenceModeling_output = opt.hidden_size

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, opt.hidden_size, opt.num_class)

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        transformer_feature = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(transformer_feature)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(visual_feature)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt.batch_max_length)

        return prediction
