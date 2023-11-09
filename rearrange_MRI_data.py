## This script rearranges the .py files from the MRI dataset based on their classification

import numpy as np
import csv
from PIL import Image
from matplotlib import pyplot as plt


def four_element_number(num):
    """Makes a string number 4 digits and returns a string"""
    num = int(num)
    if num < 10:
        return "000{}".format(num)
    elif num < 100:
        return "00{}".format(num)
    elif num < 1000:
        return "0{}".format(num)
    else:
        return "{}".format(num)


def dictionary_reader(csv_path):
    """Reads in a csv file into a dictionary where the first column is the key"""

    d = dict()
    d["img"] = list()
    d["label"] = list()

    with open(csv_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            d["img"].append(four_element_number(row["img"]))
            d["label"].append(row["label"])
            line_count += 1

        return d


def read_np_array(file_name):
    """Reads in .npy file"""
    data = np.load(file_name)
    return data


def save_image_file(slice_array, file_name, save_location):
    """Saves np slice to specified save location and file name"""
    im = Image.fromarray(slice_array)
    im.save("{}/{}".format(save_location, file_name))


def get_save_file_name(image_number, slice_idx, extension):
    """Assembles save file name"""
    return "{}_{}.{}".format(image_number, slice_idx, extension)


def get_save_location(label):
    """Get the global save location depending on the image classification"""
    if label == '0':
        save_location = 'C:/Users/jrb187/PycharmProjects/FITNet/subset_data/0NORMAL'
    elif label == '1':
        save_location = 'C:/Users/jrb187/PycharmProjects/FITNet/subset_data/1ACLINJ'
    return save_location


if __name__ == '__main__':
    start_path = "C:/Users/jrb187/Downloads/MRNet-v1.0/train"  # location of data
    label_file = "C:/Users/jrb187/Downloads/MRNet-v1.0/train-acl.csv"  # location of classification file

    dimension = "sagittal"  # dimension we want

    d = dictionary_reader(label_file)
    num_slices_total = list()

    for img, label in zip(d["img"], d["label"]):
        data = read_np_array(start_path + "/" + dimension + "/" + img + ".npy")
        num_slices = data.shape[0]
        num_slices_total.append(num_slices)

        if num_slices == 24:

            for slice_idx in range(0, num_slices):
                slice_num = four_element_number(slice_idx)
                slice_data = data[slice_idx, 12:236, 12:236]
                save_file_name = get_save_file_name(img, slice_num, 'bmp')
                save_location = get_save_location(label)

                save_image_file(slice_data, save_file_name, save_location)

    fewest_slices = min(num_slices_total)
    print(fewest_slices)

    plt.hist(num_slices_total, bins=range(min(num_slices_total), max(num_slices_total) + 1, 1))
    plt.show()