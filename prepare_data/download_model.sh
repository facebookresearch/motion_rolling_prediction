#!/bin/sh
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

# Download model command
wget URL_TO_BE_UPDATED

unzip rpm_pretrained_weights.zip -d checkpoints
rm rpm_pretrained_weights.zip

echo "Pre-trained model was downloaded into './checkpoints' folder!"
