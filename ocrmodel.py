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
from model.transformation import TPS_SpatialTransformerNetwork
from model.feature_extraction import ResNet_FeatureExtractor
from model.sequence_modeling import BidirectionalLSTM
from model.prediction import Attention


class OCRModel(nn.Module):
    def __init__(self, config):
        super(OCRModel, self).__init__()
        self.config = config
        self.stages = {'Trans': "TPS Spatial Transformer", 'Feat': "ResNet Feature Extractor",
                       'Seq': "Bidirectional LSTM", 'Pred': "Attention"}

        """ Transformation """
        self.Transformation = TPS_SpatialTransformerNetwork(
            config=self.config,
            F=self.config.num_fiducial,
            I_size=(self.config.img_height, self.config.img_width),
            I_r_size=(self.config.img_height, self.config.img_width),
            I_channel_num=self.config.input_channel
        )

        """ FeatureExtraction """
        self.FeatureExtraction = ResNet_FeatureExtractor(
            input_channel=self.config.input_channel,
            output_channel=self.config.output_channel
        )
        self.FeatureExtraction_output = self.config.output_channel  # int.img_height/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final .img_height/16-1) -> 1

        """ Sequence modeling"""
        if self.config.SequenceModeling:
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, self.config.hidden_size, self.config.hidden_size),
                BidirectionalLSTM(self.config.hidden_size, self.config.hidden_size, self.config.hidden_size))
            self.SequenceModeling_output = self.config.hidden_size
        else:
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        self.Prediction = Attention(
            config=self.config,
            input_size=self.SequenceModeling_output,
            hidden_size=self.config.hidden_size,
            num_classes=self.config.num_class
        )

    def forward(self, input, text, is_train=True):
        """ Transformation stage """
        transformer_feature = self.Transformation(input)  # → (batch, 1, 32, 100)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(transformer_feature)  # → (batch, 512, 1, 26)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # → (batch, 26, 512, 1)
        visual_feature = visual_feature.squeeze(3)  # → (batch, 26, 512)

        """ Sequence modeling stage """
        if self.config.SequenceModeling:
            contextual_feature = self.SequenceModeling(visual_feature)  # → (batch, 26, 256)
        else:
            contextual_feature = visual_feature  # → (batch, 26, 512)

        """ Prediction stage """
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.config.text_max_length)
        return prediction
