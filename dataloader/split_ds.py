import os
import os.path as osp
import random


# for aneurysm
def split_ds(data_path, train_rate, val_rate):
    images_path = []
    labels_path = []

    for curr_path, sec_paths, _ in os.walk(data_path):
        # print(curr_path)
        for sec_path in sec_paths:
            sec_path = osp.join(curr_path, sec_path)
            # print(sec_path)
            for _, _, files_name in os.walk(sec_path):
                for file_name in files_name:
                    file_path = osp.join(sec_path, file_name)
                    if 'origin' in file_path:
                        images_path.append(file_path)
                    elif 'ias' in file_path:
                        labels_path.append(file_path)
    total_size = len(images_path)
    total_index = [i for i in range(total_size)]
    random.shuffle(total_index)

    train_index = total_index[:int(total_size * train_rate)]
    val_index = total_index[int(total_size * train_rate):int(total_size * (train_rate + val_rate))]
    test_index = total_index[int(total_size * (train_rate + val_rate)):]
    train_list = []
    val_list = []
    test_list = []
    for i, (image, label) in enumerate(zip(images_path, labels_path)):
        if i in train_index:
            train_list.append((image, label))
        if i in val_index:
            val_list.append((image, label))
        if i in test_index:
            test_list.append((image, label))
    train_dic = [{'image': image, 'label': label} for image, label in train_list]
    val_dic = [{'image': image, 'label': label} for image, label in val_list]
    test_dic = [{'image': image, 'label': label} for image, label in test_list]

    # return train_dic[:6], val_dic[:4], test_dic[:1]
    return train_dic[:216], val_dic[:24], test_dic[:24]
    # return train_dic, val_dic


# for spleen
def spleen_ds(data_path, split):
    images_path = []
    labels_path = []
    for curr_path, _, files_name in os.walk(data_path):
        # print(curr_path)
        for file_name in files_name:
            file_path = osp.join(curr_path, file_name)
            images_path.append(file_path)
            labels_path.append(file_path.replace('imagesTr', 'labelsTr'))
    total_size = len(images_path)
    total_index = [i for i in range(total_size)]
    random.shuffle(total_index)

    train_index = total_index[:int(total_size * split)]
    val_index = total_index[int(total_size * split):]
    train_list = []
    val_list = []
    for i, (image, label) in enumerate(zip(images_path, labels_path)):
        if i in train_index:
            train_list.append((image, label))
        if i in val_index:
            val_list.append((image, label))
    train_dic = [{'image': image, 'label': label} for image, label in train_list]
    val_dic = [{'image': image, 'label': label} for image, label in val_list]
    return train_dic[:24], val_dic[4], val_dic


if __name__ == '__main__':
    path = r'../../Datasets/Task09_Spleen/imagesTr'
    # train_dict, val_dict = split_ds(path, 0.8)
    train_dict, val_dict, test_dict = spleen_ds(path, 0.8)
    print(train_dict)
    print(val_dict)
