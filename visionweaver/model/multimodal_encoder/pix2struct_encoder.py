# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import re
from email.mime import image

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
    CLIPImageProcessor,
    Pix2StructForConditionalGeneration,
    Pix2StructProcessor,
    Pix2StructVisionModel,
)


class Pix2StructVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        
        self.args = args
        self.vision_tower_name = vision_tower
        self.select_layer = getattr(args, 'mm_vision_select_layer', -1)
        self.freeze_vision = getattr(args, 'freeze_vision_tower', True)
        
        self.input_image_size = getattr(args, 'input_image_size', 336)

        self.do_resize = getattr(args, 'do_resize', True)
        self.de_normalize = getattr(args, 'de_normalize', True)# de-normalize the input image and perform preprocessing with pix2struct processor
        
        self.load_model()

    def load_model(self):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return
        
        self.image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        whole_model = Pix2StructForConditionalGeneration.from_pretrained(self.vision_tower_name)
        self.vision_tower = whole_model.encoder
        self.pix2struct_processor = AutoProcessor.from_pretrained(self.vision_tower_name)
        self.pix2struct_processor.image_processor.is_vqa = False

        if self.freeze_vision:
            self.vision_tower.requires_grad_(False)
        
        self.image_mean = torch.tensor(self.image_processor.image_mean).view(1, 3, 1, 1)
        self.image_std = torch.tensor(self.image_processor.image_std).view(1, 3, 1, 1)
        
        self.is_loaded = True


    def forward(self, images):

        if self.de_normalize:
            mean = self.image_mean.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
            std = self.image_std.clone().view(1, 3, 1, 1).to(dtype=images.dtype, device=images.device)
            x = (images * std + mean) * 255.0
            x = self.pix2struct_processor(images=x.float(), return_tensors="pt")
        else:
            x = images

        image_features = self.vision_tower(**(x.to(device=self.device, dtype=self.dtype))).last_hidden_state
        bs, n, c = image_features.shape
        image_features  = image_features[:, :2025, :] # HARD CODE
        
        if self.do_resize:
            image_features = image_features.transpose(1,2).reshape(bs, c, 45, 45) # HARD CODE
            image_features = F.interpolate(image_features.float(), size=(32, 32), mode='bilinear', align_corners=True).to(dtype=image_features.dtype) # HARD CODE
            
        image_features = self.interpolate_pos_encoding(image_features)

        return image_features

    def interpolate_pos_encoding(self, image_features):
        if len(image_features.shape) == 3:
            b, n, c = image_features.shape
            w = h = int(n ** 0.5)
            image_features = image_features.transpose(1, 2).reshape(b, c, h, w)
        else:
            b, c, h, w = image_features.shape

        # if w != self.input_image_tokens:
        #     image_features = F.interpolate(image_features.float(), size=(self.num_grids, self.num_grids), mode='bilinear', align_corners=True)
            
        return image_features.flatten(2, 3).transpose(1, 2)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        # Hard code
        hidden_dim = 1536
        return hidden_dim

    @property
    def num_patches(self):
        return self.config['num_patches']
