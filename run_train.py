from train import train

folder_name = "folder"        # Subfolder names become labels
d_model = 128
num_heads = 4
num_layers = 4
train_num_epochs = 10
train_save_period = 1

train(folder_name, d_model, num_heads, num_layers, train_num_epochs, train_save_period)
