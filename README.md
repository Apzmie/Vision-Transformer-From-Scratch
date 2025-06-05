# Vision-Transformer-From-Scratch
Vision Transformer has become a potential alternative to CNN, which performs well in image classification. This project is the implementation of Vision Transformer from scratch in PyTorch, based on the paper [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/pdf/2010.11929).
# run_train.py
```python
from train import train

folder_name = "folder"        # Subfolder names become labels
d_model = 128
num_heads = 4
num_layers = 4
train_num_epochs = 10
train_save_period = 1

train(folder_name, d_model, num_heads, num_layers, train_num_epochs, train_save_period)
```
```text
[1/10] Loss: 0.6501
Model saved at epoch 1 with the best loss: 0.6501
[2/10] Loss: 0.0784
Model saved at epoch 2 with the best loss: 0.0784
[3/10] Loss: 0.0391
Model saved at epoch 3 with the best loss: 0.0391
```

# run_classify.py
```python
from classify import classify

image_path = "image.jpg"

# The values must be the same as the ones used during training
folder_name = "folder"
d_model = 128
num_heads = 4
num_layers = 4

classify(folder_name, d_model, num_heads, num_layers, image_path)
```
```text
Cat
```
