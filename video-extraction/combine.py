#This script will go through images folder to combine all .npy files into one data.npy file, and 
#iterate through data.csv to combine all labels into one label.npy file.
#This script is used after each team member has completed labeling videos.
import numpy as np
import csv

IMAGES_FOLDER = './images'

ATTRIBUTE1_NAME = 'intensity'
ATTRIBUTE2_NAME = 'tse_numerator'
ATTRIBUTE3_NAME = 'tse_denominator'
ATTRIBUTE4_NAME = 'bpm'
ATTRIBUTE5_NAME = 'energy'
ATTRIBUTE6_NAME = 'style'

all_data = []
all_labels = []
all_subjective_labels = []
with open('data.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        filename = row['name']
        data = np.load(IMAGES_FOLDER+"/"+filename+".npy")
        all_data.append(data)

        label = [row[ATTRIBUTE2_NAME], row[ATTRIBUTE3_NAME], row[ATTRIBUTE4_NAME], row[ATTRIBUTE5_NAME]]
        all_labels.append(label)

        subjective_label = [row[ATTRIBUTE1_NAME], row[ATTRIBUTE6_NAME]]
        all_subjective_labels.append(subjective_label)

all_data = np.stack(all_data)
all_labels = np.stack(all_labels)
all_subjective_labels = np.stack(all_subjective_labels)

np.save('data', all_data)
np.save('label', all_labels)
np.save('subjective_label', all_subjective_labels)

print('Done.')
