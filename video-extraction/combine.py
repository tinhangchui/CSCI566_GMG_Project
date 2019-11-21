#This script will go through images folder to combine all .npy files into one data.npy file, and 
#iterate through data.csv to combine all labels into one label.npy file.
#This script is used after each member of team has completed labeling videos.
import numpy as np
import csv

IMAGES_FOLDER = './images'

all_data = []
all_labels = []
with open('data.csv', mode='r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        filename = row['name']
        data = np.load(IMAGES_FOLDER+"/"+filename+".npy")
        all_data.append(data)

        label = [row['intensity'], row['happiness'], row['attribute3'], row['attribute4'], row['attribute5']]
        all_labels.append(label)

all_data = np.stack(all_data)
all_labels = np.stack(all_labels)

np.save('data', all_data)
np.save('label', all_labels)

print('Done.')
