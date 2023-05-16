# import modules
import numpy as np
import os
import pickle
import time
import glob

### parameters ###
# data
model_name = "cnn_model"
input_dir = "./input_data"
output_result = "./results"
test_file = "test.txt"

# machine
# 0-3
gpu = "0"

# input image
img_size = 40
batch_size = 16
ch = 1

# params of machie learning
num_epochs = 200
lr = 5e-7
weight_decay = 0.1  # L2 regularization

## when below parameter is filepath to saved model,
## training continue from where training left off last time.
# load_model = None
load_model = output_result + "/" + model_name + ".pkl"

# when you want to test CNN, please switch below parameter to True.
prediction = False
#load_model = output_result + "/min_val_model_0.p"

#####################################################
# select gpu
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import torch

torch.backends.cudnn.benchmark = True

# print version od pytorch
total_time_start = time.time()
print("PyTorch Version: ", torch.__version__)

num_data = len(glob.glob(input_dir + '/*'))
print('total number of data is {}'.format(num_data))

# import my modules
from utils import make_shuffle_number, calc_mean_std_of_output
from utils import MyTransformer, MyDataset, filenames_and_labels
from utils import run_training, run_test

print("data loading follows {}.".format(input_dir))

# make shuffle number array
shuffle_number = make_shuffle_number(num_total=num_data, seed=12345)

# make dataloader for training and validation
(
    images_train,
    labels_train,
    images_val,
    labels_val,
    images_test,
    labels_test,
) = filenames_and_labels(
    path_to_data=input_dir,
    split_list=shuffle_number,
    num_train=int(num_data*0.7),
    num_val=int(num_data*0.1),
)

mean, std = calc_mean_std_of_output(
    path_to_data=input_dir, filename="fsigma8.dat", num_data=num_data
)

transform_train = MyTransformer(
    box_size=int(img_size), ch=int(ch), phase="train"
)
dataset_train = MyDataset(
    file_names=images_train,
    labels=labels_train,
    mean=mean,
    std=std,
    transform=transform_train,
)

transform_val = MyTransformer(box_size=int(img_size), ch=int(ch), phase="val")
dataset_val = MyDataset(
    file_names=images_val,
    labels=labels_val,
    mean=mean,
    std=std,
    transform=transform_val,
)

transform_test = MyTransformer(
    box_size=int(img_size), ch=int(ch), phase="test"
)
dataset_test = MyDataset(
    file_names=images_test,
    labels=labels_test,
    mean=mean,
    std=std,
    transform=transform_test,
)

train_loader = torch.utils.data.DataLoader(
    dataset_train, batch_size=int(batch_size), shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    dataset_val, batch_size=int(batch_size), shuffle=False
)
test_loader = torch.utils.data.DataLoader(
    dataset_test, batch_size=1, shuffle=False
)

# constract CNN
from cnn_model import myCNN

# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model
if load_model != None:
    if os.path.isfile(load_model) == True:
        print("load model {} and restart traning.".format(load_model))
        print("learning rate : {}".format(lr))
        with open(load_model, "rb") as f:
            model = pickle.load(f)
            model = model.to(device)
        train_loss_list, val_loss_list = np.load(output_result + "/loss_list.npy")

    else:
        print(
            "{} does not exist. traning start from the begining.".format(
                load_model
            )
        )
        print("learning rate : {}".format(lr))
        model = myCNN().to(device)
        print("==== model architecture ===")
        train_loss_list = []
        val_loss_list = []

else:
    print("traning start from the begining.")
    print("learning rate : {}".format(lr))
    model = myCNN().to(device)
    print("=======")
    train_loss_list = []
    val_loss_list = []

num_params = 0
for p in model.parameters():
    if p.requires_grad:
        num_params += p.numel()
        
print('totale number of parameters in this CNN architecture is', num_params)
    

# loss function and optimizer
import torch.nn as nn
import torch.optim as optim

criterion = nn.MSELoss(reduction="sum")

optimizer = torch.optim.Adam(
    model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=False
)


if prediction == False:
    train_loss_list, val_loss_list = run_training(
        num_epochs=num_epochs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        train_loss_list=train_loss_list,
        val_loss_list=val_loss_list,
        output_dir=output_result,
    )

    with open(output_result + "/" + model_name + ".pkl", "wb") as f:
        pickle.dump(model, f)

    np.save(
        output_result + "/loss_list.npy",
        np.array([train_loss_list, val_loss_list]),
    )

    print(
        "trained model is saved in {}".format(
            output_result + "/" + model_name + ".pkl"
        )
    )

    # stock model
    import shutil
    import datetime

    dt_now = datetime.datetime.now()
    stock_name = "model_{}{}{}_{}_{}".format(
        dt_now.year, dt_now.month, dt_now.day, dt_now.hour, dt_now.minute
    )
    shutil.copy(
        output_result + "/" + model_name + ".pkl",
        output_result + "/models/" + stock_name + ".pkl",
    )


elif prediction == True:
    print("training is skipped. run test.")

    run_test(
        model=model,
        filename_list=images_test,
        output_file=output_result + "/" + test_file,
        dataloader=test_loader,
        device=device,
        mean=mean,
        std=std,
    )

    dataset_test = MyDataset(
        file_names=images_train, labels=labels_train, mean=mean, std=std, transform=transform_test
    )

    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False
    )

    run_test(
        model=model,
        filename_list=images_train,
        output_file=output_result + "/" + "train.txt",
        dataloader=test_loader,
        device=device,
        mean=mean,
        std=std
    )

else:
    raise ValueError("prediction should be True or False.")

print(f"total time: {time.time()-total_time_start:.1f}s \n")
