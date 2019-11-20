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
	cout << options.getArg(1) << "  " << midifile.read(options.getArg(1)) << "\n";
	midifile.doTimeAnalysis();
	midifile.linkNotePairs();

	int tracks = midifile.getTrackCount();
	cout << "TPQ: " << midifile.getTicksPerQuarterNote() << endl;
	if (tracks > 1) cout << "TRACKS: " << tracks << endl;
	for (int track = 0; track < tracks; track++) {
		if (tracks > 1) cout << "\nTrack " << track << endl;
		cout << "Tick\tSeconds\tDur\tMessage" << endl;
		for (int event = 0; event < midifile[track].size(); event++) {
			cout << dec << midifile[track][event].tick;
			cout << '\t' << dec << midifile[track][event].seconds;
			cout << '\t';
			if (midifile[track][event].isNoteOn())
				cout << midifile[track][event].getDurationInSeconds();
			cout << '\t' << hex;
			for (int i = 0; i < midifile[track][event].size(); i++)
				cout << (int)midifile[track][event][i] << ' ';
			cout << endl;
		}
	}

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
