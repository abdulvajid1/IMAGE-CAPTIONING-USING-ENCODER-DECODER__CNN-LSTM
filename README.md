# Image Captioning Using CNN-LSTM (Encoder-Decoder)
### Overview
This project implements an Image Captioning Model inspired by the research paper "Show and Tell: A Neural Image Caption Generator". The model combines a Convolutional Neural Network (CNN) as an encoder and a Long Short-Term Memory (LSTM) network as a decoder to generate descriptive captions for images. This approach bridges the gap between computer vision and natural language processing by enabling machines to understand and describe visual content.

### Features

- CNN Encoder: Extracts visual features from input images using pre-trained models like InceptionV3 or ResNet.
- LSTM Decoder: Generates natural language captions by processing the encoded image features.
- End-to-End Learning: Seamlessly integrates vision and language tasks in a single pipeline.
- Custom Dataset Support: Works with datasets of images and captions, such as FLICKER8K.
- Evaluation Metrics: Implements BLEU scores to assess caption quality.

### Architecture

- Encoder:
A pre-trained CNN extracts feature vectors from input images.
Fully connected layers reduce the dimensionality of these features.

- Decoder:
An LSTM processes the image features and generates captions word by word.
The model uses teacher forcing during training to improve sequence prediction.
