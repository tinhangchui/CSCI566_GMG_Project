// MIDIFeatureExtractor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "MidiFile.h"
#include "Options.h"
#include <iostream>
#include <iomanip>

#include <iostream>
#include "fstream"
#include "string"
#include <io.h>   
#include <vector>  
#include <algorithm>
#include <stdio.h>

using namespace std;
using namespace smf;

void getFiles(std::string folderPath, vector<string>& files) {
	// refer: https://www.cnblogs.com/fnlingnzb-learner/p/6424563.html
	// file handel
	long hFile = 0;
	// file info
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(folderPath).append("\\*").c_str(), &fileinfo)) != -1) {
		do {
			if ((fileinfo.attrib &  _A_SUBDIR)) {
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
					getFiles(p.assign(folderPath).append("\\").append(fileinfo.name), files);
			}
			else {
				files.push_back(p.assign(folderPath).append("\\").append(fileinfo.name));
			}
		
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

vector<string> getMIDIFiles(std::string folderPath, bool is_print, bool is_write) {
	vector<string> files;
	getFiles(folderPath, files);

	if (is_print)
		for (int i = 0; i < files.size(); i++) {
			cout << files[i].c_str() << endl;
		}
	if (is_write) {
		std::ofstream fout;
		fout.open(folderPath + "\\midifile_names.txt");

		for (int i = 0; i < files.size(); i++)
			fout << std::setprecision(10) << files.at(i) << std::endl;

		fout.close();
	}

	return files;
}

string extractFeaturesFromMIDI(std::string fileName) {
	// parse midi file
	MidiFile midifile;
	midifile.read(fileName);
	midifile.doTimeAnalysis();
	midifile.linkNotePairs();
	
	// extract features
	cout << "file_name: " << fileName << endl;
	// print hard-coded meta info
	int controlTracks = 0;
	int nn = 4;
	int dd = 4;
	// tse
	int tracks = midifile.getTrackCount();
	for (int track = 0; track < tracks; track++) {
		if (midifile[track].size() > 2) {
			controlTracks++;
		}
		if (track == 0) {
			for (int event = 0; event < midifile[track].size(); event++) {
				for (int i = 0; i < midifile[track][event].size(); i++) {
					if ((int)midifile[track][event][0] == 0xff && (int)midifile[track][event][1] == 0x58) {
						nn = (int)midifile[track][event][2];
						dd = (int)pow(2, (int)midifile[track][event][3]);
					}
				}
			}
		}
	}
	cout << "    tse: " << nn << "/" << dd << endl;
	// bpm
	int bpm = (60 * 1000000) / (midifile.getTicksPerQuarterNote() * 4);
	cout << "    bpm: " << bpm << endl;
	// energy
	cout << "    energy: " << controlTracks << endl;
	
	// return
	return to_string(nn) + " " + to_string(dd) + " " + to_string(bpm) + " " + to_string(controlTracks);
}

int main(int argc, char** argv) {
	Options options;
	options.process(argc, argv);
	// MidiFile midifile;
	// if (options.getArgCount() == 0) midifile.read(cin);
	// else midifile.read(options.getArg(1));
	// midifile.doTimeAnalysis();
	// midifile.linkNotePairs();


	// batch read midi files
	// std::string folderPath = "C:\\Users\\ziang\\OneDrive\\work_space\\CSCI-566\\Project\\CSCI566_GMG_Project\\MIDIFeatureExtractor\\MIDIFeatureExtractor\\midi";
	// if (options.getArgCount() != 0)
	// 	folderPath = options.getArg(1);
	std::string folderPath = "C:\\Users\\ziang\\OneDrive\\work_space\\CSCI-566\\Project\\CSCI566_GMG_Project\\MIDIFeatureExtractor\\MIDIFeatureExtractor\\midi\\adventure-island";
	printf("Enter midi folder path:\n");
	scanf("%s", folderPath);
	std::vector<string> files = getMIDIFiles(folderPath, false, true);
	sort(files.begin(), files.end());

	// extract features from midi fiels
	std::vector<string> paramList;
	for (int i = 0; i < files.size(); i++) {
		paramList.push_back(extractFeaturesFromMIDI(files.at(i)));
	}

	// write file
	std::ofstream fout;
	fout.open(folderPath + "\\param_list.txt");
	for (int i = 0; i < paramList.size(); i++)
		fout << std::setprecision(10) << paramList.at(i) << std::endl;
	fout.close();

	return 0;
}



// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
