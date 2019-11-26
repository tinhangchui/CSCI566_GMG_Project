
#This script will go through images folder to combine all .npy files into one data.npy file, and
#iterate through data.csv to combine all labels into one label.npy file.
#This script is used after each team member has completed labeling videos.
import numpy as np
import csv

IMAGES_FOLDER = './images'

#ATTRIBUTE1_NAME = 'intensity'
ATTRIBUTE2_NAME = 'tse_numerator'
ATTRIBUTE3_NAME = 'tse_denominator'
ATTRIBUTE4_NAME = 'bpm'
ATTRIBUTE5_NAME = 'energy'

all_data = []
all_labels = []
with open('data.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        filename = row['name']
        data = np.load(IMAGES_FOLDER+"/"+filename+".npy")
        all_data.append(data)

        label = [row[ATTRIBUTE2_NAME], row[ATTRIBUTE3_NAME], row[ATTRIBUTE4_NAME], row[ATTRIBUTE5_NAME]]
        all_labels.append(label)

all_data = np.stack(all_data)
all_labels = np.stack(all_labels)

np.save('data', all_data)
np.save('label', all_labels)

print('Done.')
