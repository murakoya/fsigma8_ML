# import modules
import numpy as np
import torch
import glob
import time
import os
import pickle

# data preprocessing
class MyTransformer:
    def __init__(self, box_size, ch=1, phase="train"):
        self.box_size = box_size
        self.ch = ch
        self.phase = phase

        if not phase in ["train", "val", "test"]:
            raise ValueError("phase={} is unexpected.".format(phase))

    def augmentation():
        pass

    def __call__(self, file_path):
        box = np.load(file_path)
        box.resize(box.shape[0], box.shape[1], box.shape[2], 1)
        box = box.astype(np.float32)

        return box


def make_shuffle_number(num_total, seed=12345):
    # make array to divide the data to training, validation and test
    np.random.seed(seed=seed)
    x = np.arange(num_total)
    np.random.shuffle(x)

    return x.astype(np.int32)


def calc_mean_std_of_output(path_to_data, filename, num_data):
    # calc values for normalization and standarizaiton of output of CNN and ground truth value
    param_list = np.array(
        [
            np.loadtxt(path_to_data + "/dir{}/".format(i) + filename)
            for i in range(int(num_data))
        ]
    )
    mean = np.mean(param_list)
    std = np.std(param_list)

    return mean, std


def filenames_and_labels(
    path_to_data, split_list, num_train=1500, num_val=100
):
    all_box_train = [
        path_to_data + "/dir{}/df.npy".format(int(split_list[i]))
        for i in range(num_train)
    ]

    all_box_val = [
        path_to_data + "/dir{}/df.npy".format(int(split_list[i + num_train]))
        for i in range(num_val)
    ]

    all_box_test = [
        path_to_data
        + "/dir{}/df.npy".format(int(split_list[int(i + num_train + num_val)]))
        for i in range(int(len(split_list) - num_train - num_val))
    ]

    all_label_train = [
        np.array(
            [
                np.loadtxt(
                    path_to_data
                    + "/dir{}/fsigma8.dat".format(int(split_list[i]))
                )
            ]
        )
        for i in range(num_train)
    ]

    all_label_val = [
        np.array(
            [
                np.loadtxt(
                    path_to_data
                    + "/dir{}/fsigma8.dat".format(
                        int(split_list[i + num_train])
                    )
                )
            ]
        )
        for i in range(num_val)
    ]

    all_label_test = [
        np.array(
            [
                np.loadtxt(
                    path_to_data
                    + "/dir{}/fsigma8.dat".format(
                        int(split_list[int(i + num_train + num_val)])
                    )
                )
            ]
        )
        for i in range(int(len(split_list) - num_train - num_val))
    ]

    return (
        all_box_train,
        all_label_train,
        all_box_val,
        all_label_val,
        all_box_test,
        all_label_test,
    )


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, file_names, labels, mean, std, transform=None):
        self.file_names = file_names
        self.labels = labels
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.file_names)

    # when you use self[idx], this method is called
    def __getitem__(self, idx):
        if self.transform == None:
            box = torch.tensor(
                np.load(self.file_names[idx]), dtype=torch.float32
            )

            label = torch.tensor(self.labels[idx], dtype=torch.float32)

        # data preprocessing
        else:
            box = torch.tensor(
                self.transform(self.file_names[idx]), dtype=torch.float32
            )
            converted = (self.labels[idx] - self.mean) / self.std
            label = torch.tensor(converted, dtype=torch.float32)

        return box.permute(3, 0, 1, 2), label


def train_epoch(model, optimizer, criterion, dataloder, device, num_params=5):
    train_loss = 0.0
    loss_count = 0.0

    output_list = np.zeros(int(num_params))
    true_value_list = np.zeros(int(num_params))
    data_count = 0

    # training mode
    model.train()

    for i, (box, labels) in enumerate(dataloder):
        # load data & labels
        box, labels = box.to(device), labels.to(device)

        # initialize gradient
        optimizer.zero_grad()
        # get outputs from CNN
        outputs = model(box)

        # get the value of CNN output and true labels
        for j in range(len(labels.cpu().numpy())):
            output_list += outputs.detach().cpu().numpy()[j]
            true_value_list += labels.cpu().numpy()[j]
            data_count += 1

        # calc loss
        loss = criterion(outputs, labels)

        # back propagation
        loss.backward()
        # optimization
        optimizer.step()

        train_loss += loss.item()
        loss_count += 1.0

    # average loss for one epoch
    train_loss = train_loss / loss_count
    output_list /= data_count
    true_value_list /= data_count

    return train_loss, output_list, true_value_list


def validation(model, criterion, dataloader, device, num_params=1):
    test_loss = 0
    loss_count = 0

    # evaluation mode
    model.eval()

    # gradient is not calculated in this block
    with torch.no_grad():
        for i, (box, labels) in enumerate(dataloader):
            box, labels = box.to(device), labels.to(device)
            outputs = model(box)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            loss_count += 1.0

        test_loss = test_loss / loss_count

    return test_loss


# perform training
def run_training(
    num_epochs,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    train_loss_list=[],
    val_loss_list=[],
    output_dir="./results",
):

    min_val = 1e10

    if len(train_loss_list) == 0:
        train_loss_list = []
        val_loss_list=[]
        start_epoch = 0

    # you want to start where you stopped training
    else:
        train_loss_list = train_loss_list.tolist()
        val_loss_list = val_loss_list.tolist()
        start_epoch = len(train_loss_list)

    for epoch in range(num_epochs):

        if os.path.isfile(output_dir + "/stop"):
            os.remove(output_dir + "/stop")
            print("get the call of stop. training is finished and save model.")
            break

        start = time.time()
        train_loss, output_list, true_value_list = train_epoch(
            model, optimizer, criterion, train_loader, device
        )
        test_loss = validation(model, criterion, val_loader, device)

        if test_loss < min_val:
            min_epoch = np.copy(epoch)
            min_val = np.copy(test_loss)
            with open(output_dir + "/min_val_model_{}.p".format(start_epoch), "wb") as f:
                pickle.dump(model, f)

        print(
            f"Epoch [{epoch+1+start_epoch}], time : {time.time()-start:.1f} s, train_loss : {train_loss:.4f}, test_loss : {test_loss:.4f}"
        )
        train_loss_list.append(train_loss)
        val_loss_list.append(test_loss)

    print("min_val_model is saved at epoch {}.".format(min_epoch+start_epoch))

    return train_loss_list, val_loss_list


def run_test(model, filename_list, output_file, dataloader, device, mean, std):
    model.eval()

    true_list = []
    pred_list = []

    with torch.no_grad():
        with open(output_file, "w") as f:
            for i, (box, labels) in enumerate(dataloader):
                box, labels = box.to(device), labels.to(device)
                outputs = model(box)

                true_value = labels.cpu().numpy()
                predicted_value = outputs.detach().cpu().numpy()

                f.write(filename_list[i] + "\n")
                f.write("true_value : {}\n".format(true_value[0] * std + mean))
                f.write(
                    "predicted_value: {}\n\n".format(
                        predicted_value[0] * std + mean
                    )
                )

                true_list.append(true_value[0] * std + mean)
                pred_list.append(predicted_value[0] * std + mean)

    np.save(
        output_file.replace(".txt", ".npy"), np.array([true_list, pred_list])
    )
