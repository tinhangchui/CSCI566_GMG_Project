import cv2
import os
import csv
import requests
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from pytube import YouTube

VIDEO_FOLDER = './video'           #Folder to hold the video
MIDI_FOLDER = './midi'             #Folder to hold the midi
IMAGES_FOLDER = './images'         #Folder to hold the image numpy data
IMAGES_DEBUG_FOLDER = './debug_images' #Folder to hold the debug images
DATA_FILENAME = 'data.csv'   #Name of the data file that records attributes

IMAGE_RESIZE_SCALE = (256, 256)      #The output picture's size
FRAME_SKIP_NUM = 10                    #How much frame to skip after capture a picture

ATTRIBUTE1_NAME = 'intensity'
ATTRIBUTE2_NAME = 'tse_numerator'
ATTRIBUTE3_NAME = 'tse_denominator'
ATTRIBUTE4_NAME = 'bpm'
ATTRIBUTE5_NAME = 'energy'

ATTRIBUTE1_RANGE = (0, 4)
ATTRIBUTE2_RANGE = (0, 256)
ATTRIBUTE3_RANGE = (0, 128)
ATTRIBUTE4_RANGE = (31250, 150000)
ATTRIBUTE5_RANGE = (1, 20)

def getYouTube():
    url = input("Enter the url of Youtube : ")
    try:
        yt = YouTube(url)
    except:
        print("An error occured when fetching youtube video.")
        return None, None
    stream = yt.streams.first()
    filename = input("Enter the filename of video (Press Enter to use default filename) : ")
    if filename == '':
        filename = None
    return stream, filename

def getClipTime():
    buffer = ''
    while buffer == '':
        buffer = input("Enter the section starting time (sec after 0:00. ending time is 5s after starting time) : ")
        try:
            start = int(buffer)
            if start < 0:
                print("Please input a positive integer.")
                buffer = ''
        except ValueError:
            print("Please input a positive integer.")
            buffer = ''
    """
    # If the length of image arrays are inconsistent, there will be issues in stacking all images arrays into one.
    # So for now we assume the ending time is 5 seconds after the starting time.
    buffer = ''
    while buffer == '':
        buffer = input("Enter the section ending time (sec after 0:00) : ")
        try:
            end = int(buffer)
            if (end <= start):
                print("Ending time must be greater than starting time.")
                buffer = ''
        except ValueError:
            print("Please input a positive integer.")
            buffer = ''
    """

    return start, start + 5

def outputNumpyFile(section_dict, frame_array):
    numpy_arr = np.array(frame_array)
    filename = IMAGES_FOLDER+"/"+section_dict['name']

    #Output data
    np.save(filename, frame_array)
    print("Output numpy array with shape {}.".format(numpy_arr.shape))

def getSectionDone():
    ans = input("You have finished adding a section. Any more section for this video? (y/n) : ")
    if ans == 'y' or ans == 'yes':
        return False                #because if answer yes, it means we are not done yet
    elif ans =='n' or ans == 'no':
        return True
    while True:
        ans = input("Please answer y or n.")
        if ans == 'y' or ans == 'yes':
            return False
        elif ans =='n' or ans == 'no':
            return True

def getAttribute():
    def getValue(name, low_bound, up_bound):
        attribute = -999
        while True:
            attribute = float(input(f"{name} (Input a number {low_bound} to {up_bound}) : "))
            if attribute >= low_bound and attribute <= up_bound:
                return attribute
            else:
                print("Invalid input!")

    result = dict()
    result[ATTRIBUTE1_NAME] = getValue(ATTRIBUTE1_NAME, ATTRIBUTE1_RANGE[0], ATTRIBUTE1_RANGE[1])
    result[ATTRIBUTE2_NAME] = getValue(ATTRIBUTE2_NAME, ATTRIBUTE2_RANGE[0], ATTRIBUTE2_RANGE[1])
    result[ATTRIBUTE3_NAME] = getValue(ATTRIBUTE3_NAME, ATTRIBUTE3_RANGE[0], ATTRIBUTE3_RANGE[1])   
    result[ATTRIBUTE4_NAME] = getValue(ATTRIBUTE4_NAME, ATTRIBUTE4_RANGE[0], ATTRIBUTE4_RANGE[1])
    result[ATTRIBUTE5_NAME] = getValue(ATTRIBUTE5_NAME, ATTRIBUTE5_RANGE[0], ATTRIBUTE5_RANGE[1])
    return result


def writeToCSV(section_dict):
    fieldnames = ['name', ATTRIBUTE1_NAME, ATTRIBUTE2_NAME, ATTRIBUTE3_NAME, ATTRIBUTE4_NAME, ATTRIBUTE5_NAME]
    if not os.path.exists("./"+DATA_FILENAME):
        with open(DATA_FILENAME, 'w') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(section_dict)
    else:
        with open(DATA_FILENAME, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writerow(section_dict)

def addMIDI(filename):
    url = input("(Optional) Add section MIDI url to download. Enter nothing to skip this part. : ")
    if url != '':
        try:
            req = requests.get(url)
            with open(MIDI_FOLDER+"/"+filename, 'wb') as f:
                f.write(req.content)
                print("Done writing MIDI file!")
        except:
            print("Error while trying to download MIDI file.")      

def get_video_path(filename, section_num=None):
    if section_num == None:
        return VIDEO_FOLDER+"/"+filename+".mp4"
    return VIDEO_FOLDER+"/"+filename+"_"+str(section_num)+".mp4"

def get_debug_image_folder(filename, section_num):
    if section_num == None:
        return IMAGES_DEBUG_FOLDER+"/"+filename
    return IMAGES_DEBUG_FOLDER+"/"+filename+"_"+str(section_num)

def run():
    while True:
        print("-----------------------------------------------")
        stream, filename = getYouTube()
        if stream != None:          
            print("Downloading...")
            stream.download(VIDEO_FOLDER, filename=filename)
            print("Download Complete!")

            done_section = False
            section_num = 0
            while not done_section:
                start, end = getClipTime()
                ffmpeg_extract_subclip(get_video_path(filename), start, end, targetname=get_video_path(filename, section_num))

                cap = cv2.VideoCapture(get_video_path(filename, section_num))
                frame_num = 0
                count = 0
                os.makedirs(get_debug_image_folder(filename, section_num))
                frame_arr = []
                while(cap.isOpened()):
                    ret, frame = cap.read()
                    if ret == False:
                        break
                    if frame_num % FRAME_SKIP_NUM == 0:
                        resized_frame = cv2.resize(frame, IMAGE_RESIZE_SCALE)
                        frame_arr.append(resized_frame)
                        cv2.imwrite(get_debug_image_folder(filename, section_num)+"/"+filename+"_"+str(count)+'.jpg', resized_frame)
                        count += 1
                    frame_num += 1

                section_dict = getAttribute()
                section_dict['name'] = filename+"_"+str(section_num)

                outputNumpyFile(section_dict, frame_arr)
                print("Image output Complete!")

                writeToCSV(section_dict)
                print("Written to data csv file!")

                addMIDI(filename)

                section_num += 1
                done_section = getSectionDone()

if __name__ == "__main__":
    if not os.path.exists(VIDEO_FOLDER):
        os.makedirs(VIDEO_FOLDER)
    if not os.path.exists(MIDI_FOLDER):
        os.makedirs(MIDI_FOLDER)
    if not os.path.exists(IMAGES_FOLDER):
        os.makedirs(IMAGES_FOLDER)
    if not os.path.exists(IMAGES_DEBUG_FOLDER):
        os.makedirs(IMAGES_DEBUG_FOLDER)
    run()