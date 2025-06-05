from classify import classify

image_path = "cat.jpg"

# The values must be the same as the ones used during training
folder_name = "folder"
d_model = 128
num_heads = 4
num_layers = 4

classify(folder_name, d_model, num_heads, num_layers, image_path)        # 'vit.pth' is required to run
