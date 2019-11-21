// MIDIFeatureExtractor.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "pch.h"
#include "MidiFile.h"
#include "Options.h"
#include <iostream>
#include <iomanip>

using namespace std;
using namespace smf;

int main(int argc, char** argv) {
	Options options;
	options.process(argc, argv);
	MidiFile midifile;
	if (options.getArgCount() == 0) midifile.read(cin);
	else midifile.read(options.getArg(1));
	// cout << options.getArg(1) << "  " << midifile.read(options.getArg(1)) << endl;
	midifile.doTimeAnalysis();
	midifile.linkNotePairs();


	int controlTracks = 0;
	int nn = 4;
	int dd = 4;

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

	// print hard-coded meta info
	cout << "bpm: " << (60 * 1000000) / (midifile.getTicksPerQuarterNote() * 4) << endl;
	cout << "energy: " << controlTracks << endl;
	cout << "tse: " << nn << "/" << dd << endl;

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
