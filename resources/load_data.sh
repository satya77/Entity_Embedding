#!/bin/bash
# (c) 2019 Satya Almasian

LOAD_NETWORK="https://dbs.ifi.uni-heidelberg.de/files/datasets/LOAD_Network_For_Embeddings.zip"
PRETRAINED_MODELS="https://dbs.ifi.uni-heidelberg.de/files/datasets/pre-trained_embeddings.zip"

echo "This script downloads and extracts the example files, this may take some time..."
echo "Downloading Network files (1.29 GB)..."
curl -o load_network.zip $LOAD_NETWORK 
echo "Extracting  Network files ..."
unzip load_network.zip
rm load_network.zip
echo "Downloading pre-trained models (2.27GB)..."
curl -o pretrained_embeddings.zip $PRETRAINED_MODELS
echo "Extracting pretrained models ..."
unzip pretrained_embeddings.zip
rm pretrained_embeddings.zip
echo "Sucessfully downloaded and extracted files!"
