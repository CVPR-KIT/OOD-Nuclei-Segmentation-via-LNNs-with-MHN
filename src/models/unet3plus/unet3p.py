# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv_modules_unet3p import unetConv2, MaxBlurPool2d
from .init_weights import init_weights

from src.models.liquid_nn import LiquidNN

from src.models.quantization import ResidualQuantization
from src.models.hopfield_head import HopfieldHead


class UNet_3Plus(nn.Module):
    def __init__(
        self,
        config,
        exp_dir,
        in_channels=3,
        n_classes=2,
        feature_scale=4,
        is_deconv=True,
        is_batchnorm=True,
    ):
        super().__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale
        self.activation = config.model.activation
        self.kernel_size = config.model.kernel_size
        self.ch = config.model.base_filters
        self.segment_type = config.model.segmentation_type
        self.dropout = nn.Dropout2d(p=config.model.dropout)
        self.useMaxBPool = True
        self.exp_dir = exp_dir
        self.config = config

        self.n_classes = 4 if self.segment_type == "instance" else n_classes

        filters = [self.ch, self.ch * 2, self.ch * 4, self.ch * 8, self.ch * 16]

        # ---------------------- LiquidNN ----------------------
        try:
            selected_lnn_type = config.model.lnn_type
        except Exception:
            selected_lnn_type = "lnn1"

        try:
            self.use_lnn = config.model.use_lnn
        except Exception:
            self.use_lnn = True

        self.embedding_size = 16
        if self.use_lnn:
            plots_dir = os.path.join(self.exp_dir, "plots/") if self.exp_dir is not None else None

            self.liquid_nn = LiquidNN(
                feature_channels=filters[::-1],
                hidden_size=64,
                embedding_size=self.embedding_size,
                ncp_type="CfC",
                exp_dir=plots_dir,
            )
            
        else:
            self.liquid_nn = None

        # Simple concat fusion (always used when use_lnn=True)
        self.concat_conv = None
        if self.use_lnn:
            self.concat_conv = nn.Conv2d(
                in_channels=filters[4] + self.embedding_size,
                out_channels=filters[4],
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True,
            )

        # ---------------------- Quantizer (optional) ----------------------
        try:
            self.use_quantizer = config.model.use_quantizer
        except Exception:
            self.use_quantizer = False

        if self.use_quantizer:
            self.lnn_quant = ResidualQuantization(
                input_dim=self.embedding_size,
                num_quantizers=8,
                codebook_size=512,
                shared_codebook=True,
            )
            self.image_quant = ResidualQuantization(
                input_dim=filters[4],
                num_quantizers=8,
                codebook_size=512,
                shared_codebook=True,
            )

        # ---------------------- Hopfield ----------------------
        try:
            self.use_hopfield = config.model.use_hopfield
        except Exception:
            self.use_hopfield = True

        if self.use_hopfield:
            self.hopfield_head = HopfieldHead(use_lnn=False)
        else:
            self.hopfield_head = None

        # ---------------------- Encoder ----------------------
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        self.maxpool1 = MaxBlurPool2d(kernel_size=2) if self.useMaxBPool else nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        self.maxpool2 = MaxBlurPool2d(kernel_size=2) if self.useMaxBPool else nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        self.maxpool3 = MaxBlurPool2d(kernel_size=2) if self.useMaxBPool else nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm, ks=self.kernel_size, act=self.activation)
        self.maxpool4 = MaxBlurPool2d(kernel_size=2) if self.useMaxBPool else nn.MaxPool2d(kernel_size=2)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm, ks=self.kernel_size, act=self.activation)

        # ---------------------- Decoder ----------------------
        self.CatChannels = filters[0]
        self.CatBlocks = 5
        self.UpChannels = self.CatChannels * self.CatBlocks

        # stage 4d
        self.h1_PT_hd4 = nn.MaxPool2d(8, 8, ceil_mode=True)
        self.h1_PT_hd4_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd4 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h2_PT_hd4_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h3_PT_hd4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h3_PT_hd4_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_PT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_PT_hd4_relu = nn.ReLU(inplace=True)

        self.h4_Cat_hd4_conv = nn.Conv2d(filters[3], self.CatChannels, self.kernel_size, padding=1)
        self.h4_Cat_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.h4_Cat_hd4_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd4 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd5_UT_hd4_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd4_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd4_relu = nn.ReLU(inplace=True)

        self.conv4d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)
        self.bn4d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu4d_1 = nn.ReLU(inplace=True)

        # stage 3d
        self.h1_PT_hd3 = nn.MaxPool2d(4, 4, ceil_mode=True)
        self.h1_PT_hd3_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h2_PT_hd3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h2_PT_hd3_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_PT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_PT_hd3_relu = nn.ReLU(inplace=True)

        self.h3_Cat_hd3_conv = nn.Conv2d(filters[2], self.CatChannels, self.kernel_size, padding=1)
        self.h3_Cat_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.h3_Cat_hd3_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd3 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd4_UT_hd3_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd3_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd3 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.hd5_UT_hd3_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd3_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd3_relu = nn.ReLU(inplace=True)

        self.conv3d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)
        self.bn3d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu3d_1 = nn.ReLU(inplace=True)

        # stage 2d
        self.h1_PT_hd2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.h1_PT_hd2_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_PT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_PT_hd2_relu = nn.ReLU(inplace=True)

        self.h2_Cat_hd2_conv = nn.Conv2d(filters[1], self.CatChannels, self.kernel_size, padding=1)
        self.h2_Cat_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.h2_Cat_hd2_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd2 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd3_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd2 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.hd4_UT_hd2_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd2_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd2 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.hd5_UT_hd2_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd2_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd2_relu = nn.ReLU(inplace=True)

        self.conv2d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)
        self.bn2d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu2d_1 = nn.ReLU(inplace=True)

        # stage 1d
        self.h1_Cat_hd1_conv = nn.Conv2d(filters[0], self.CatChannels, self.kernel_size, padding=1)
        self.h1_Cat_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.h1_Cat_hd1_relu = nn.ReLU(inplace=True)

        self.hd2_UT_hd1 = nn.Upsample(scale_factor=2, mode="bilinear")
        self.hd2_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd2_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd2_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd3_UT_hd1 = nn.Upsample(scale_factor=4, mode="bilinear")
        self.hd3_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd3_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd3_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd4_UT_hd1 = nn.Upsample(scale_factor=8, mode="bilinear")
        self.hd4_UT_hd1_conv = nn.Conv2d(self.UpChannels, self.CatChannels, self.kernel_size, padding=1)
        self.hd4_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd4_UT_hd1_relu = nn.ReLU(inplace=True)

        self.hd5_UT_hd1 = nn.Upsample(scale_factor=16, mode="bilinear")
        self.hd5_UT_hd1_conv = nn.Conv2d(filters[4], self.CatChannels, self.kernel_size, padding=1)
        self.hd5_UT_hd1_bn = nn.BatchNorm2d(self.CatChannels)
        self.hd5_UT_hd1_relu = nn.ReLU(inplace=True)

        self.conv1d_1 = nn.Conv2d(self.UpChannels, self.UpChannels, self.kernel_size, padding=1)
        self.bn1d_1 = nn.BatchNorm2d(self.UpChannels)
        self.relu1d_1 = nn.ReLU(inplace=True)

        self.outconv1 = nn.Conv2d(self.UpChannels, self.n_classes, self.kernel_size, padding=1)

        # ---------------------- init ----------------------
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type="kaiming")
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type="kaiming")

    def guidedForward(self, x):
        h1 = self.conv1(x)

        h2 = self.maxpool1(h1)
        h2 = self.conv2(h2)

        h3 = self.maxpool2(h2)
        h3 = self.conv3(h3)

        h4 = self.maxpool3(h3)
        h4 = self.conv4(h4)

        h5 = self.maxpool4(h4)
        hd5 = self.conv5(h5)

        # LNN embeddings
        lnn_embeddings = None
        if self.use_lnn and (self.liquid_nn is not None):
            lnn_embeddings = self.liquid_nn([hd5, h4, h3, h2, h1])

        # Quantizer (optional)
        commit_loss_image, commit_loss_lnn = None, None
        if getattr(self, "use_quantizer", False):
            quant_image, _, commit_loss_image = self.image_quant(hd5)
            if self.use_lnn and (lnn_embeddings is not None):
                quant_lnn, _, commit_loss_lnn = self.lnn_quant(lnn_embeddings)
            else:
                quant_lnn = None
        else:
            quant_image, quant_lnn = None, None

        # Simple concat fusion (hd5 + lnn_embeddings) -> hd5
        if self.use_lnn and (lnn_embeddings is not None) and (self.concat_conv is not None):
            hd5 = self.concat_conv(torch.cat([hd5, lnn_embeddings], dim=1))

        # Optional Hopfield refinement on fused hd5
        if self.use_hopfield and (self.hopfield_head is not None):
            try:
                hd5 = self.hopfield_head(hd5)
            except TypeError:
                if getattr(self, "use_quantizer", False):
                    if self.use_lnn and (quant_lnn is not None):
                        hd5 = self.hopfield_head(quant_image, quant_lnn)
                    else:
                        hd5 = self.hopfield_head(quant_image)
                else:
                    if self.use_lnn and (lnn_embeddings is not None):
                        hd5 = self.hopfield_head(hd5, lnn_embeddings)
                    else:
                        hd5 = self.hopfield_head(hd5)

        # Decoder
        h1_PT_hd4 = self.h1_PT_hd4_relu(self.h1_PT_hd4_bn(self.h1_PT_hd4_conv(self.h1_PT_hd4(h1))))
        h2_PT_hd4 = self.h2_PT_hd4_relu(self.h2_PT_hd4_bn(self.h2_PT_hd4_conv(self.h2_PT_hd4(h2))))
        h3_PT_hd4 = self.h3_PT_hd4_relu(self.h3_PT_hd4_bn(self.h3_PT_hd4_conv(self.h3_PT_hd4(h3))))
        h4_Cat_hd4 = self.h4_Cat_hd4_relu(self.h4_Cat_hd4_bn(self.h4_Cat_hd4_conv(h4)))
        hd5_UT_hd4 = self.hd5_UT_hd4_relu(self.hd5_UT_hd4_bn(self.hd5_UT_hd4_conv(self.hd5_UT_hd4(hd5))))
        hd4 = self.relu4d_1(self.bn4d_1(self.conv4d_1(torch.cat((h1_PT_hd4, h2_PT_hd4, h3_PT_hd4, h4_Cat_hd4, hd5_UT_hd4), 1))))

        h1_PT_hd3 = self.h1_PT_hd3_relu(self.h1_PT_hd3_bn(self.h1_PT_hd3_conv(self.h1_PT_hd3(h1))))
        h2_PT_hd3 = self.h2_PT_hd3_relu(self.h2_PT_hd3_bn(self.h2_PT_hd3_conv(self.h2_PT_hd3(h2))))
        h3_Cat_hd3 = self.h3_Cat_hd3_relu(self.h3_Cat_hd3_bn(self.h3_Cat_hd3_conv(h3)))
        hd4_UT_hd3 = self.hd4_UT_hd3_relu(self.hd4_UT_hd3_bn(self.hd4_UT_hd3_conv(self.hd4_UT_hd3(hd4))))
        hd5_UT_hd3 = self.hd5_UT_hd3_relu(self.hd5_UT_hd3_bn(self.hd5_UT_hd3_conv(self.hd5_UT_hd3(hd5))))
        hd3 = self.relu3d_1(self.bn3d_1(self.conv3d_1(torch.cat((h1_PT_hd3, h2_PT_hd3, h3_Cat_hd3, hd4_UT_hd3, hd5_UT_hd3), 1))))

        h1_PT_hd2 = self.h1_PT_hd2_relu(self.h1_PT_hd2_bn(self.h1_PT_hd2_conv(self.h1_PT_hd2(h1))))
        h2_Cat_hd2 = self.h2_Cat_hd2_relu(self.h2_Cat_hd2_bn(self.h2_Cat_hd2_conv(h2)))
        hd3_UT_hd2 = self.hd3_UT_hd2_relu(self.hd3_UT_hd2_bn(self.hd3_UT_hd2_conv(self.hd3_UT_hd2(hd3))))
        hd4_UT_hd2 = self.hd4_UT_hd2_relu(self.hd4_UT_hd2_bn(self.hd4_UT_hd2_conv(self.hd4_UT_hd2(hd4))))
        hd5_UT_hd2 = self.hd5_UT_hd2_relu(self.hd5_UT_hd2_bn(self.hd5_UT_hd2_conv(self.hd5_UT_hd2(hd5))))
        hd2 = self.relu2d_1(self.bn2d_1(self.conv2d_1(torch.cat((h1_PT_hd2, h2_Cat_hd2, hd3_UT_hd2, hd4_UT_hd2, hd5_UT_hd2), 1))))

        h1_Cat_hd1 = self.h1_Cat_hd1_relu(self.h1_Cat_hd1_bn(self.h1_Cat_hd1_conv(h1)))
        hd2_UT_hd1 = self.hd2_UT_hd1_relu(self.hd2_UT_hd1_bn(self.hd2_UT_hd1_conv(self.hd2_UT_hd1(hd2))))
        hd3_UT_hd1 = self.hd3_UT_hd1_relu(self.hd3_UT_hd1_bn(self.hd3_UT_hd1_conv(self.hd3_UT_hd1(hd3))))
        hd4_UT_hd1 = self.hd4_UT_hd1_relu(self.hd4_UT_hd1_bn(self.hd4_UT_hd1_conv(self.hd4_UT_hd1(hd4))))
        hd5_UT_hd1 = self.hd5_UT_hd1_relu(self.hd5_UT_hd1_bn(self.hd5_UT_hd1_conv(self.hd5_UT_hd1(hd5))))
        hd1 = self.relu1d_1(self.bn1d_1(self.conv1d_1(torch.cat((h1_Cat_hd1, hd2_UT_hd1, hd3_UT_hd1, hd4_UT_hd1, hd5_UT_hd1), 1))))

        d1 = self.outconv1(hd1)

        if self.segment_type == "instance":
            return d1, commit_loss_image, commit_loss_lnn
        return torch.sigmoid(d1)

    def forward(self, inputs):
        return self.guidedForward(inputs)
