"""
Data preparation and storage services for GameMusicVAE
"""

import pretty_midi
import numpy as np
# For plotting
import mir_eval.display
import librosa.display
import matplotlib.pyplot as plt
import os
import pickle

from config import *

class DataPreparation(object):
    """
    prepare model-using data from raw midi files collected on internet
    """
    dataset = []
    trio_dataset = []
    def __init__(self):
        print("preparing data...")
        DataPreparation.dataset = self.build_dataset(DATA_POOL_PATH, DATABASE_PATH)
        DataPreparation.trio_dataset = self.generate_trainable_data(DATABASE_PATH, self.dataset)
        print("data prepared!")

    @classmethod
    def build_dataset(data_pool_path, save_path):
        """
        build dataset entity from raw midi files
        parse as pretty_midi PrettyMIDI object
        save out at local holder
        :param data_pool_path: the folder path that stores all raw midi files
        :param save_path: the path of persisting dataset
        :return: dataset(list of PrettyMIDI object) for GameMusicVAE
        """
        # load midi files as PrettyMIDI
        midi_file_list = []
        file_name_list = os.listdir(data_pool_path)
        print("loading midi files...")
        for file_name in file_name_list:
            if not os.path.isdir(file_name) and file_name.endswith(".mid"):
                midi_file = pretty_midi.PrettyMIDI(os.path.join(data_pool_path,  file_name))
                midi_file_list.append(midi_file)
        print("load success! (#%d)".format(len(midi_file_list)))

        # persisting dataset
        print("saving dataset...")
        file_path_name = os.path.join(save_path, "dataset.txt")
        f = open(file_path_name, 'wb')
        pickle.dump(midi_file_list, f)
        f.close()
        print("saved dataset at " + file_path_name)

        # return
        return midi_file_list

    @classmethod
    def generate_trainable_data(dataset_path, alive_dataset=[]):
        """
        parse dataset to screen and split layers
        generate a wrapped dataset for training networks
        save trainable_dataset to same path with dataset
        :param dataset_path: path of loading dataset and saving trainable_dataset
        :param alive_dataset: caller can pass a in-memory dataset into this function
        :return:
        """
        # restore dataset
        print("loading dataset to memory...")
        dataset = []
        if alive_dataset:
            dataset = alive_dataset
        else:
            f = open(dataset_path, 'rb')
            dataset = pickle.load(f)
            f.close()
        print("loaded dataset into running memory!")

        # screen data for trio dataset
        print("screening dataset...")
        trio_dataset = []
        pass
        print("built trio dataset")

        # split tracks into pieces
        print("splitting tracks")
        pass
        print("split done!")

        # persisting trainable_dataset
        print("saving trio dataset...")
        file_path_name = os.path.join(dataset_path, "trio_dataset.txt")
        f = open(file_path_name, 'wb')
        pickle.dump(trio_dataset, f)
        f.close()
        print("saved trio dataset at " + file_path_name)

        # return
        return trio_dataset