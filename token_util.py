import pretty_midi as midi
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn import preprocessing
MIDI_FOLDER = '/Users/outsutomuchou/566project/midi_all'
FILE_NAMES = os.listdir(MIDI_FOLDER)


def read_data(files):
    count = np.zeros((len(files), 128))
    piano = np.zeros((count.shape[0], 8))
    index = []
    midi_data = []
    for i in range(count.shape[0]):
        try:
            data = midi.PrettyMIDI(os.path.join(MIDI_FOLDER, files[i]))
        except:
            continue
        for instr in data.instruments:
            program = instr.program
            is_drum = instr.is_drum
            if is_drum:
                count[i, 8] += 1
            else:
                count[i, program] += 1
        piano[i] = count[i, :8]
        if np.sum(piano[i]) != 0:
            index.append(i)
            midi_data.append(data)
    piano = count[index, :8]
    return midi_data, piano


def choose_instrument(data, piano):
    """
    :param data:
    :param piano: np.array (8,)
    :return:
    """
    piano_index = []
    all_instr = data.instruments
    for i, instrument in enumerate(all_instr):
        if instrument.is_drum or instrument.program > 7:
            continue
        else:
            piano_index.append(i)
    num = len(np.nonzero(piano)[0])
    piano_instr = [all_instr[i] for i in piano_index]
    if num == 1:
        if len(piano_index) == 1 :
            instr1 = all_instr[piano_index[0]]
            instr2 = 0
        elif len(piano_index) == 2:
            instr1,instr2 = [all_instr[i] for i in piano_index]
        else:
            t = np.zeros(len(piano_index))
            for j, instrument in enumerate(piano_instr):
                t[j] = instrument.get_onsets().shape[0]
                index = t.argsort()[-2:][::-1]
                instr1 = piano_instr[index[0]]
                instr2 = piano_instr[index[1]]
    else:
        length = np.zeros(len(piano_instr))
        for j, instrument in enumerate(piano_instr):
            length[j] = instrument.get_onsets().shape[0]
        index = length.argsort()[-2:][::-1]
        instr1 = piano_instr[index[0]]
        instr2 = piano_instr[index[1]]

    return instr1, instr2


def get_allinstr(midi_data,piano):
    all_instr = []
    for i in range(len(midi_data)):
        data = midi_data[i]
        piano_sample = piano[i]
        instr1, instr2 = choose_instrument(data, piano_sample)
        all_instr.append(instr1)
        if instr2 != 0:
            all_instr.append(instr2)
    return all_instr


def add_time_label(instr1):
    label = []
    note1 = instr1.notes
    length1 = len(note1)
    delta1 = np.zeros(length1)
    for i in range(length1):
        delta1[i] = note1[i].end-note1[i].start
    min_val = np.min(delta1)
    max_val = np.max(delta1)
    d = (max_val-min_val)/4
    for i in range(len(delta1)):
        vel = delta1[i]
        if min_val <= vel <= min_val+d:
            label.append(0)
        if min_val+d < vel <= min_val+2*d:
            label.append(1)
        if min_val+2*d < vel <= min_val+3*d:
            label.append(2)
        else:
            label.append(3)
    return label


def tokenize(all_instr):
    seq_list = []
    for i in range(len(all_instr)):
        if len(all_instr[i].notes) < 20:
            continue
        else:
            label1 = add_time_label(all_instr[i])
            note = all_instr[i].notes
            length = len(note)
            seq_num = int(length/20)
            result = np.zeros((seq_num, 20, 3))
            for j in range(seq_num*20):
                pitch1 = note[j].pitch
                label = label1[j]
                seq = int(j/20)
                pos = j % 20
                vector = np.array([pitch1, label, pos])
                result[seq, pos, :] = vector
            seq_list.append(result)
    a = seq_list[0]
    for t in range(1, len(seq_list)):
        a = np.concatenate((a,seq_list[t]))
    return a


def run():
    midi_data, piano = read_data(FILE_NAMES)
    all_instr = get_allinstr(midi_data, piano)
    return tokenize(all_instr)

final = run()








