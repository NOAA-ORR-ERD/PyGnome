#ifndef CODEWARRIOR
	#include <Carbon/Carbon.r>
#endif

resource 'DITL' (3801, "M38b: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{17, 17, 33, 246},
		StaticText {
			disabled,
			"Select a map file:"
		}
	}
};

resource 'DITL' (1200, "M12: Load / Define LEs") {
	{	/* array DITLarray: 8 elements */
		/* [1] */
		{104, 277, 124, 335},
		Button {
			enabled,
			"Create"
		},
		/* [2] */
		{100, 274, 128, 338},
		UserItem {
			disabled
		},
		/* [3] */
		{104, 206, 124, 264},
		Button {
			enabled,
			"Load"
		},
		/* [4] */
		{104, 136, 124, 194},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{54, 61, 72, 169},
		UserItem {
			disabled
		},
		/* [6] */
		{20, 21, 52, 350},
		StaticText {
			disabled,
			"Please select the type of  spill to be a"
			"dded:"
		},
		/* [7] */
		{55, 20, 71, 57},
		StaticText {
			disabled,
			"Type:"
		},
		/* [8] */
		{104, 66, 124, 124},
		Button {
			enabled,
			"Help"
		}
	}
};

resource 'DITL' (3802, "M38c: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a current file:"
		}
	}
};

resource 'DITL' (3803, "M38d: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a time file:"
		}
	}
};

resource 'DITL' (1900, "M19: Format selection") {
	{	/* array DITLarray: 8 elements */
		/* [1] */
		{72, 27, 95, 300},
		Button {
			enabled,
			"real, real"
		},
		/* [2] */
		{67, 22, 100, 304},
		UserItem {
			disabled
		},
		/* [3] */
		{115, 27, 138, 300},
		Button {
			enabled,
			"magnitude, degrees"
		},
		/* [4] */
		{155, 27, 178, 300},
		Button {
			enabled,
			"degrees, magnitude"
		},
		/* [5] */
		{195, 27, 218, 300},
		Button {
			enabled,
			"magnitude, NESW"
		},
		/* [6] */
		{234, 27, 257, 300},
		Button {
			enabled,
			"NESW, magnitude"
		},
		/* [7] */
		{282, 229, 304, 299},
		Button {
			enabled,
			"Cancel"
		},
		/* [8] */
		{20, 23, 53, 303},
		StaticText {
			disabled,
			"What is the format of the final two fiel"
			"ds of the file ^0?"
		}
	}
};

resource 'DITL' (1400, "M14: Complex Random Mover") {
	{	/* array DITLarray: 23 elements */
		/* [1] */
		{279, 245, 299, 311},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{275, 241, 303, 315},
		UserItem {
			disabled
		},
		/* [3] */
		{279, 168, 299, 234},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{279, 96, 299, 162},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{21, 68, 37, 249},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{54, 21, 72, 84},
		CheckBox {
			enabled,
			"Active"
		},
		/* [7] */
		{99, 138, 115, 213},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{161, 120, 177, 195},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{161, 226, 177, 301},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{191, 120, 207, 195},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [11] */
		{191, 226, 207, 301},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [12] */
		{240, 97, 256, 172},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [13] */
		{89, 18, 135, 313},
		UserItem {
			disabled
		},
		/* [14] */
		{149, 18, 220, 313},
		UserItem {
			disabled
		},
		/* [15] */
		{101, 24, 117, 130},
		StaticText {
			disabled,
			"coverage = 0 to"
		},
		/* [16] */
		{21, 22, 37, 65},
		StaticText {
			disabled,
			"Name:"
		},
		/* [17] */
		{99, 225, 115, 300},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [18] */
		{162, 24, 178, 111},
		StaticText {
			disabled,
			"magnitude = "
		},
		/* [19] */
		{191, 58, 207, 115},
		StaticText {
			disabled,
			"angle = "
		},
		/* [20] */
		{163, 203, 180, 220},
		StaticText {
			disabled,
			"to"
		},
		/* [21] */
		{193, 203, 210, 220},
		StaticText {
			disabled,
			"to"
		},
		/* [22] */
		{240, 25, 256, 89},
		StaticText {
			disabled,
			"duration:"
		},
		/* [23] */
		{240, 179, 256, 254},
		StaticText {
			disabled,
			"hours"
		}
	}
};

resource 'DITL' (1700, "M17: Choose Scaling Grid") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{129, 193, 151, 260},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{125, 189, 155, 264},
		UserItem {
			disabled
		},
		/* [3] */
		{129, 112, 151, 179},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{129, 23, 151, 90},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{21, 17, 112, 264},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (1800, "M18: Wind Mover Settings") {
	{	/* array DITLarray: 23 elements */
		/* [1] */
		{262, 204, 282, 270},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{258, 200, 286, 274},
		UserItem {
			disabled
		},
		/* [3] */
		{262, 125, 282, 191},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{262, 37, 282, 103},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{30, 55, 48, 117},
		CheckBox {
			enabled,
			"Active"
		},
		/* [6] */
		{143, 153, 163, 214},
		EditText {
			enabled,
			"Uncertain"
		},
		/* [7] */
		{170, 153, 186, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{193, 153, 210, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{217, 153, 232, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{216, 221, 232, 302},
		UserItem {
			enabled
		},
		/* [11] */
		{54, 54, 72, 196},
		CheckBox {
			enabled,
			"Subsurface Active"
		},
		/* [12] */
		{78, 120, 94, 170},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{18, 20, 107, 318},
		UserItem {
			disabled
		},
		/* [14] */
		{129, 20, 241, 318},
		UserItem {
			disabled
		},
		/* [15] */
		{170, 58, 186, 122},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [16] */
		{171, 221, 187, 262},
		StaticText {
			disabled,
			"hours"
		},
		/* [17] */
		{193, 58, 210, 143},
		StaticText {
			disabled,
			"Speed Scale:"
		},
		/* [18] */
		{217, 28, 234, 143},
		StaticText {
			disabled,
			"Total Angle Scale:"
		},
		/* [19] */
		{144, 58, 160, 133},
		StaticText {
			disabled,
			"Start Time:"
		},
		/* [20] */
		{145, 222, 161, 261},
		StaticText {
			disabled,
			"hours"
		},
		/* [21] */
		{120, 24, 136, 99},
		StaticText {
			disabled,
			"Uncertainty"
		},
		/* [22] */
		{78, 54, 93, 109},
		StaticText {
			disabled,
			"Gamma:"
		},
		/* [23] */
		{10, 24, 27, 79},
		StaticText {
			disabled,
			"Settings"
		}
	}
};

resource 'DITL' (3804, "M38e: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a wind file:"
		}
	}
};

resource 'DITL' (2200, "M22:  Save File Type") {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{104, 345, 124, 403},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{101, 343, 129, 407},
		UserItem {
			disabled
		},
		/* [3] */
		{104, 274, 124, 332},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{104, 204, 124, 262},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{61, 61, 80, 370},
		UserItem {
			disabled
		},
		/* [6] */
		{20, 21, 46, 405},
		StaticText {
			disabled,
			"Please select the type of file to be sav"
			"ed:"
		},
		/* [7] */
		{62, 20, 78, 57},
		StaticText {
			disabled,
			"Type:"
		}
	}
};

resource 'DITL' (2100, "M21: Load / Define Movers") {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{58, 270, 78, 328},
		Button {
			disabled,
			"Create"
		},
		/* [2] */
		{54, 267, 82, 331},
		UserItem {
			disabled
		},
		/* [3] */
		{58, 338, 78, 396},
		Button {
			enabled,
			"Load"
		},
		/* [4] */
		{58, 404, 78, 462},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{59, 58, 77, 186},
		UserItem {
			disabled
		},
		/* [6] */
		{20, 16, 39, 426},
		StaticText {
			disabled,
			"Please select the type and source of new"
			" mover to be added:"
		},
		/* [7] */
		{59, 17, 75, 54},
		StaticText {
			disabled,
			"Type:"
		}
	}
};

resource 'DITL' (2300, "M23: Load / Define Weatherers") {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{71, 182, 91, 240},
		Button {
			enabled,
			"Create"
		},
		/* [2] */
		{67, 179, 95, 243},
		UserItem {
			disabled
		},
		/* [3] */
		{71, 254, 91, 312},
		Button {
			enabled,
			"Load"
		},
		/* [4] */
		{71, 320, 91, 378},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{70, 58, 88, 166},
		UserItem {
			disabled
		},
		/* [6] */
		{19, 18, 50, 387},
		StaticText {
			disabled,
			"Please select the type and source of new"
			" weatherer to be added:"
		},
		/* [7] */
		{70, 20, 86, 57},
		StaticText {
			disabled,
			"Type:"
		}
	}
};

resource 'DITL' (2400, "M24: Weatherer Name") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{61, 266, 82, 362},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{57, 262, 86, 365},
		UserItem {
			disabled
		},
		/* [3] */
		{61, 157, 81, 239},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{19, 192, 36, 368},
		EditText {
			enabled,
			"Untitled"
		},
		/* [5] */
		{19, 18, 35, 187},
		StaticText {
			disabled,
			"Name of new weatherer:"
		}
	}
};

resource 'DITL' (2600, "M26: Weather Info") {
	{	/* array DITLarray: 26 elements */
		/* [1] */
		{341, 373, 363, 457},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{337, 369, 367, 461},
		UserItem {
			disabled
		},
		/* [3] */
		{395, 281, 417, 365},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{341, 277, 363, 361},
		Button {
			enabled,
			"Change"
		},
		/* [5] */
		{341, 169, 363, 253},
		Button {
			enabled,
			"Help"
		},
		/* [6] */
		{20, 73, 36, 459},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{76, 21, 154, 463},
		UserItem {
			enabled
		},
		/* [8] */
		{203, 50, 219, 139},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{203, 208, 219, 297},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{203, 365, 219, 454},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{283, 50, 299, 139},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{283, 208, 299, 297},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{283, 365, 299, 454},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{56, 29, 73, 137},
		StaticText {
			disabled,
			"Pollutant Name"
		},
		/* [15] */
		{56, 314, 72, 453},
		StaticText {
			disabled,
			"Standard / Modified"
		},
		/* [16] */
		{178, 21, 235, 463},
		UserItem {
			disabled
		},
		/* [17] */
		{259, 21, 316, 463},
		UserItem {
			disabled
		},
		/* [18] */
		{20, 24, 37, 68},
		StaticText {
			disabled,
			"Name:"
		},
		/* [19] */
		{171, 32, 187, 178},
		StaticText {
			disabled,
			"Half Life Components"
		},
		/* [20] */
		{203, 28, 219, 46},
		StaticText {
			disabled,
			"1:"
		},
		/* [21] */
		{203, 187, 219, 205},
		StaticText {
			disabled,
			"2:"
		},
		/* [22] */
		{203, 343, 219, 361},
		StaticText {
			disabled,
			"3:"
		},
		/* [23] */
		{252, 32, 268, 197},
		StaticText {
			disabled,
			"Percentage Components"
		},
		/* [24] */
		{283, 28, 299, 46},
		StaticText {
			disabled,
			"1:"
		},
		/* [25] */
		{283, 187, 299, 205},
		StaticText {
			disabled,
			"2:"
		},
		/* [26] */
		{283, 343, 299, 361},
		StaticText {
			disabled,
			"3:"
		}
	}
};

resource 'DITL' (7900, "Time File Delete Dialog") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{68, 316, 88, 378},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{64, 312, 92, 382},
		Picture {
			disabled,
			7000
		},
		/* [3] */
		{69, 227, 89, 304},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{19, 61, 56, 384},
		StaticText {
			disabled,
			"Do you really want to delete the time de"
			"pendent file associated with this CATS m"
			"over?"
		},
		/* [5] */
		{19, 15, 51, 47},
		Icon {
			disabled,
			129
		}
	}
};

resource 'DITL' (2700, "M27: Constant Wind Mover") {
	{	/* array DITLarray: 16 elements */
		/* [1] */
		{162, 214, 184, 294},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{158, 210, 188, 298},
		UserItem {
			disabled
		},
		/* [3] */
		{162, 114, 184, 194},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{162, 21, 184, 101},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{18, 65, 35, 300},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{90, 62, 106, 120},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{89, 124, 107, 274},
		UserItem {
			disabled
		},
		/* [8] */
		{54, 111, 72, 191},
		UserItem {
			disabled
		},
		/* [9] */
		{127, 80, 143, 120},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{126, 149, 143, 189},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [11] */
		{18, 13, 34, 57},
		StaticText {
			disabled,
			"Name:"
		},
		/* [12] */
		{90, 12, 107, 59},
		StaticText {
			disabled,
			"Speed:"
		},
		/* [13] */
		{55, 12, 71, 111},
		StaticText {
			disabled,
			"Wind is from:"
		},
		/* [14] */
		{127, 13, 144, 76},
		StaticText {
			disabled,
			"Windage:"
		},
		/* [15] */
		{127, 193, 143, 209},
		StaticText {
			disabled,
			"%"
		},
		/* [16] */
		{126, 126, 143, 142},
		StaticText {
			disabled,
			"to"
		}
	}
};

resource 'DITL' (2800, "M28: Simple Random Mover") {
	{	/* array DITLarray: 13 elements */
		/* [1] */
		{185, 241, 205, 307},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{181, 237, 209, 311},
		UserItem {
			disabled
		},
		/* [3] */
		{185, 164, 205, 230},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{185, 92, 205, 158},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{18, 68, 34, 249},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{51, 21, 69, 84},
		CheckBox {
			enabled,
			"Active"
		},
		/* [7] */
		{99, 183, 115, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{136, 183, 152, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{89, 16, 170, 361},
		UserItem {
			disabled
		},
		/* [10] */
		{99, 24, 116, 173},
		StaticText {
			disabled,
			"Diffusion Coefficient:"
		},
		/* [11] */
		{18, 22, 34, 65},
		StaticText {
			disabled,
			"Name:"
		},
		/* [12] */
		{99, 270, 115, 352},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [13] */
		{135, 24, 151, 166},
		StaticText {
			disabled,
			"Uncertainty Factor:"
		}
	}
};

resource 'DITL' (1665) {
	{	/* array DITLarray: 9 elements */
		/* [1] */
		{196, 358, 216, 428},
		Button {
			enabled,
			"Next"
		},
		/* [2] */
		{192, 354, 220, 432},
		UserItem {
			disabled
		},
		/* [3] */
		{196, 275, 216, 345},
		Button {
			enabled,
			"Previous"
		},
		/* [4] */
		{152, 197, 171, 292},
		StaticText {
			disabled,
			"Steady Winds; WIZTYPE WINDPOPUP;MENUID 1"
			"0050\nVALUE CONSTANTWIND;VALUEB 0;VALUEC "
			"1;\nVALUE VARIABLEWIND;VALUEB 1;VALUEC  0"
			";"
		},
		/* [5] */
		{152, 119, 171, 193},
		StaticText {
			disabled,
			"The wind is"
		},
		/* [6] */
		{152, 324, 171, 399},
		StaticText {
			disabled,
			"over time."
		},
		/* [7] */
		{17, 18, 52, 425},
		StaticText {
			disabled,
			"Wind can significantly influence oil mov"
			"ement and can force oil to move in a dif"
			"ferent direction than the currents."
		},
		/* [8] */
		{80, 18, 140, 393},
		StaticText {
			disabled,
			"You can choose wind that is constant in "
			"direction and speed during the entire mo"
			"del run, or you can choose wind that var"
			"ies in direction and/or speed over time."
		},
		/* [9] */
		{194, 8, 218, 149},
		Button {
			enabled,
			"Finding Wind Data; WIZTYPE HELPBUTTON;TO"
			"PIC Wind Data, Finding;"
		}
	}
};

resource 'DITL' (1660, "Edit WInds") {
	{	/* array DITLarray: 38 elements */
		/* [1] */
		{386, 425, 411, 500},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{382, 421, 415, 504},
		UserItem {
			disabled
		},
		/* [3] */
		{386, 331, 411, 405},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{386, 241, 411, 315},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{196, 161, 221, 281},
		Button {
			enabled,
			"Delete Selected"
		},
		/* [6] */
		{386, 9, 411, 130},
		Button {
			enabled,
			"Delete All"
		},
		/* [7] */
		{196, 15, 221, 140},
		Button {
			enabled,
			"Replace Selected"
		},
		/* [8] */
		{11, 15, 27, 52},
		UserItem {
			enabled
		},
		/* [9] */
		{12, 158, 28, 186},
		EditText {
			disabled,
			""
		},
		/* [10] */
		{12, 195, 28, 235},
		UserItem {
			enabled
		},
		/* [11] */
		{42, 122, 58, 150},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{42, 181, 58, 209},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{75, 74, 91, 112},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{111, 176, 127, 230},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{166, 186, 182, 220},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{240, 15, 255, 94},
		StaticText {
			disabled,
			"Date(m/d/y)"
		},
		/* [17] */
		{240, 115, 255, 149},
		StaticText {
			disabled,
			"Time"
		},
		/* [18] */
		{240, 162, 255, 207},
		StaticText {
			disabled,
			"Speed"
		},
		/* [19] */
		{240, 219, 255, 290},
		StaticText {
			disabled,
			"Wind From"
		},
		/* [20] */
		{258, 9, 373, 294},
		UserItem {
			enabled
		},
		/* [21] */
		{6, 310, 228, 532},
		UserItem {
			enabled
		},
		/* [22] */
		{450, 534, 660, 744},
		Picture {
			disabled,
			1431
		},
		/* [23] */
		{167, 15, 182, 181},
		StaticText {
			disabled,
			"Auto-increment time by:"
		},
		/* [24] */
		{510, 323, 527, 328},
		StaticText {
			disabled,
			""
		},
		/* [25] */
		{43, 15, 60, 112},
		StaticText {
			disabled,
			"Time (24 hour) :"
		},
		/* [26] */
		{75, 15, 91, 67},
		StaticText {
			disabled,
			"Speed:"
		},
		/* [27] */
		{111, 15, 130, 170},
		StaticText {
			disabled,
			"Wind Direction is from:"
		},
		/* [28] */
		{167, 232, 184, 279},
		StaticText {
			disabled,
			"hours"
		},
		/* [29] */
		{509, 501, 525, 512},
		StaticText {
			disabled,
			"/"
		},
		/* [30] */
		{491, 476, 508, 487},
		StaticText {
			disabled,
			"/"
		},
		/* [31] */
		{42, 162, 62, 172},
		StaticText {
			disabled,
			":"
		},
		/* [32] */
		{75, 123, 93, 233},
		UserItem {
			disabled
		},
		/* [33] */
		{493, 380, 511, 459},
		StaticText {
			disabled,
			""
		},
		/* [34] */
		{512, 354, 533, 474},
		Button {
			enabled,
			"Load Wind Data..."
		},
		/* [35] */
		{305, 335, 330, 499},
		Button {
			enabled,
			"Wind Settings..."
		},
		/* [36] */
		{258, 307, 373, 529},
		UserItem {
			disabled
		},
		/* [37] */
		{6, 9, 228, 297},
		UserItem {
			disabled
		},
		/* [38] */
		{133, 15, 153, 280},
		StaticText {
			disabled,
			"Enter  degrees true or text (e.g. \"NNW\")"
		}
	}
};

resource 'DITL' (1650, "Current Uncertainty") {
	{	/* array DITLarray: 30 elements */
		/* [1] */
		{269, 246, 290, 319},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{266, 243, 295, 324},
		UserItem {
			disabled
		},
		/* [3] */
		{269, 152, 290, 225},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{269, 61, 290, 134},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{173, 208, 189, 283},
		EditText {
			enabled,
			"3"
		},
		/* [6] */
		{198, 208, 214, 283},
		EditText {
			enabled,
			"2000000"
		},
		/* [7] */
		{43, 75, 59, 150},
		EditText {
			enabled,
			"from"
		},
		/* [8] */
		{42, 205, 58, 280},
		EditText {
			enabled,
			"to"
		},
		/* [9] */
		{109, 75, 125, 150},
		EditText {
			enabled,
			"from"
		},
		/* [10] */
		{108, 205, 124, 280},
		EditText {
			enabled,
			"to"
		},
		/* [11] */
		{176, 18, 192, 182},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [12] */
		{200, 18, 217, 202},
		StaticText {
			disabled,
			"Eddy Diffusivity Coefficient:"
		},
		/* [13] */
		{22, 20, 39, 187},
		StaticText {
			disabled,
			"Down Current Uncertainty"
		},
		/* [14] */
		{108, 180, 126, 199},
		StaticText {
			disabled,
			"To"
		},
		/* [15] */
		{172, 289, 189, 335},
		StaticText {
			disabled,
			"hours"
		},
		/* [16] */
		{198, 289, 218, 373},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [17] */
		{42, 156, 59, 174},
		StaticText {
			disabled,
			"%"
		},
		/* [18] */
		{108, 284, 125, 312},
		StaticText {
			disabled,
			"%"
		},
		/* [19] */
		{42, 180, 60, 201},
		StaticText {
			disabled,
			"To"
		},
		/* [20] */
		{43, 284, 61, 300},
		StaticText {
			disabled,
			"%"
		},
		/* [21] */
		{88, 17, 104, 210},
		StaticText {
			disabled,
			"Cross Current Uncertainty"
		},
		/* [22] */
		{108, 156, 124, 172},
		StaticText {
			disabled,
			"%"
		},
		/* [23] */
		{44, 30, 61, 71},
		StaticText {
			disabled,
			"From:"
		},
		/* [24] */
		{110, 30, 127, 71},
		StaticText {
			disabled,
			"From:"
		},
		/* [25] */
		{151, 19, 167, 94},
		StaticText {
			disabled,
			"Start Time:"
		},
		/* [26] */
		{147, 208, 163, 283},
		EditText {
			enabled,
			"0"
		},
		/* [27] */
		{147, 288, 163, 331},
		StaticText {
			disabled,
			"hours"
		},
		/* [28] */
		{224, 18, 242, 138},
		StaticText {
			disabled,
			"Eddy V0:"
		},
		/* [29] */
		{224, 208, 240, 283},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{223, 289, 243, 373},
		StaticText {
			disabled,
			"m/sec"
		}
	}
};

resource 'DITL' (1050, "RUNUNTIL") {
	{	/* array DITLarray: 15 elements */
		/* [1] */
		{167, 247, 188, 324},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{163, 244, 192, 329},
		UserItem {
			disabled
		},
		/* [3] */
		{167, 148, 188, 225},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{35, 90, 54, 214},
		UserItem {
			disabled
		},
		/* [5] */
		{38, 232, 54, 263},
		EditText {
			enabled,
			""
		},
		/* [6] */
		{35, 272, 54, 340},
		UserItem {
			disabled
		},
		/* [7] */
		{77, 165, 93, 193},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{77, 211, 93, 239},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{112, 156, 128, 362},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [10] */
		{132, 156, 148, 362},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [11] */
		{77, 199, 93, 207},
		StaticText {
			disabled,
			":"
		},
		/* [12] */
		{11, 17, 29, 130},
		StaticText {
			disabled,
			"Run model until:"
		},
		/* [13] */
		{72, 91, 109, 157},
		StaticText {
			disabled,
			"Stop time:\n(24-hour)"
		},
		/* [14] */
		{112, 17, 129, 152},
		StaticText {
			disabled,
			"Current model time:"
		},
		/* [15] */
		{132, 42, 148, 150},
		StaticText {
			disabled,
			"Model end time:"
		}
	}
};

resource 'DITL' (1060, "SELECTUNITS") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{58, 224, 78, 282},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{54, 221, 82, 285},
		UserItem {
			disabled
		},
		/* [3] */
		{58, 141, 78, 199},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{19, 116, 37, 224},
		UserItem {
			disabled
		},
		/* [5] */
		{19, 19, 36, 105},
		StaticText {
			disabled,
			"Units in file:"
		}
	}
};

resource 'DITL' (3800, "M38: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{16, 14, 32, 246},
		StaticText {
			disabled,
			"Select a splot file:"
		}
	}
};

resource 'DITL' (3805, "M38g: SFGetFile", purgeable) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{11, 16, 27, 359},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (5301, "M53C: Caveat") {
	{	/* array DITLarray: 14 elements */
		/* [1] */
		{236, 314, 256, 371},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{232, 310, 260, 375},
		UserItem {
			enabled
		},
		/* [3] */
		{236, 224, 256, 291},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{236, 16, 256, 136},
		Button {
			enabled,
			"Standard Caveat"
		},
		/* [5] */
		{20, 63, 53, 382},
		EditText {
			enabled,
			""
		},
		/* [6] */
		{61, 63, 94, 382},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{102, 63, 135, 382},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{143, 63, 176, 382},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{184, 63, 217, 382},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{20, 12, 36, 58},
		StaticText {
			disabled,
			"Line 1:"
		},
		/* [11] */
		{61, 12, 77, 58},
		StaticText {
			disabled,
			"Line 2:"
		},
		/* [12] */
		{102, 12, 118, 58},
		StaticText {
			disabled,
			"Line 3:"
		},
		/* [13] */
		{143, 12, 159, 58},
		StaticText {
			disabled,
			"Line 4:"
		},
		/* [14] */
		{184, 12, 200, 58},
		StaticText {
			disabled,
			"Line 5:"
		}
	}
};

resource 'DITL' (5300, "M53") {
	{	/* array DITLarray: 47 elements */
		/* [1] */
		{250, 436, 270, 493},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{247, 432, 275, 497},
		UserItem {
			enabled
		},
		/* [3] */
		{250, 350, 270, 417},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{250, 265, 270, 332},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{59, 274, 75, 491},
		EditText {
			enabled,
			""
		},
		/* [6] */
		{44, 253, 84, 503},
		UserItem {
			enabled
		},
		/* [7] */
		{113, 58, 129, 103},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{114, 160, 130, 232},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{99, 8, 139, 243},
		UserItem {
			enabled
		},
		/* [10] */
		{168, 58, 184, 103},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{169, 160, 185, 232},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{154, 8, 194, 243},
		UserItem {
			enabled
		},
		/* [13] */
		{114, 274, 130, 491},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{99, 253, 139, 503},
		UserItem {
			enabled
		},
		/* [15] */
		{169, 273, 185, 490},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{154, 252, 194, 502},
		UserItem {
			enabled
		},
		/* [17] */
		{326, 65, 344, 160},
		RadioButton {
			enabled,
			"TAT window"
		},
		/* [18] */
		{344, 65, 362, 166},
		RadioButton {
			enabled,
			"printer page"
		},
		/* [19] */
		{315, 59, 374, 172},
		UserItem {
			enabled
		},
		/* [20] */
		{31, 49, 49, 159},
		RadioButton {
			enabled,
			"black & white"
		},
		/* [21] */
		{43, 49, 61, 137},
		RadioButton {
			enabled,
			"gray scale"
		},
		/* [22] */
		{54, 49, 72, 104},
		RadioButton {
			enabled,
			"color"
		},
		/* [23] */
		{18, 43, 78, 172},
		UserItem {
			enabled
		},
		/* [24] */
		{307, 65, 323, 94},
		StaticText {
			disabled,
			"Size"
		},
		/* [25] */
		{211, 12, 231, 128},
		Button {
			enabled,
			"Edit Caveat..."
		},
		/* [26] */
		{10, 49, 26, 129},
		StaticText {
			disabled,
			"Print Mode"
		},
		/* [27] */
		{410, 101, 442, 162},
		CheckBox {
			enabled,
			"visual\nrange"
		},
		/* [28] */
		{389, 144, 406, 577},
		UserItem {
			enabled
		},
		/* [29] */
		{409, 200, 441, 232},
		Icon {
			disabled,
			1001
		},
		/* [30] */
		{409, 242, 441, 274},
		Icon {
			disabled,
			1002
		},
		/* [31] */
		{409, 284, 441, 316},
		Icon {
			disabled,
			1003
		},
		/* [32] */
		{409, 326, 441, 358},
		Icon {
			disabled,
			1004
		},
		/* [33] */
		{409, 368, 441, 400},
		Icon {
			disabled,
			1005
		},
		/* [34] */
		{409, 410, 441, 442},
		Icon {
			disabled,
			1006
		},
		/* [35] */
		{409, 452, 441, 484},
		Icon {
			disabled,
			1007
		},
		/* [36] */
		{409, 494, 441, 526},
		Icon {
			disabled,
			1008
		},
		/* [37] */
		{386, 92, 445, 580},
		UserItem {
			enabled
		},
		/* [38] */
		{113, 13, 129, 54},
		StaticText {
			disabled,
			"Time:"
		},
		/* [39] */
		{114, 115, 130, 156},
		StaticText {
			disabled,
			"Date:"
		},
		/* [40] */
		{168, 13, 184, 54},
		StaticText {
			disabled,
			"Time:"
		},
		/* [41] */
		{169, 115, 185, 156},
		StaticText {
			disabled,
			"Date:"
		},
		/* [42] */
		{36, 268, 52, 378},
		StaticText {
			disabled,
			"Scenario Name:"
		},
		/* [43] */
		{91, 18, 107, 108},
		StaticText {
			disabled,
			"Estimate for:"
		},
		/* [44] */
		{146, 21, 162, 91},
		StaticText {
			disabled,
			"Prepared:"
		},
		/* [45] */
		{91, 268, 107, 358},
		StaticText {
			disabled,
			"Prepared by:"
		},
		/* [46] */
		{146, 267, 162, 373},
		StaticText {
			disabled,
			"Contact Phone:"
		},
		/* [47] */
		{255, 12, 273, 118},
		CheckBox {
			enabled,
			"Omit Footer"
		}
	}
};

resource 'DITL' (1670, "Use Shio Ref Pt?") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{101, 250, 121, 335},
		Button {
			enabled,
			"Yes"
		},
		/* [2] */
		{97, 246, 125, 339},
		Picture {
			disabled,
			1431
		},
		/* [3] */
		{101, 138, 121, 218},
		Button {
			enabled,
			"No"
		},
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{37, 57, 82, 418},
		StaticText {
			disabled,
			"Use the reference point contained in thi"
			"s file?\n"
		}
	}
};

resource 'DITL' (5000, "M50:  TMap Settings") {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{333, 307, 353, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{329, 303, 357, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{333, 229, 353, 292},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{333, 154, 353, 217},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{11, 18, 27, 62},
		StaticText {
			disabled,
			"Name:"
		},
		/* [6] */
		{11, 71, 42, 362},
		StaticText {
			disabled,
			"map name"
		},
		/* [7] */
		{79, 18, 94, 132},
		StaticText {
			disabled,
			"Refloat Half Life:"
		},
		/* [8] */
		{79, 137, 95, 187},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{78, 200, 98, 257},
		StaticText {
			disabled,
			"hours"
		},
		/* [10] */
		{111, 17, 128, 236},
		CheckBox {
			enabled,
			"Change Bitmap Bounds"
		},
		/* [11] */
		{141, 47, 157, 112},
		StaticText {
			disabled,
			"Lat (Top):\n"
		},
		/* [12] */
		{142, 116, 158, 196},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{141, 114, 158, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [14] */
		{142, 147, 158, 174},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{141, 185, 158, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [16] */
		{142, 218, 158, 258},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{142, 218, 158, 238},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{141, 249, 159, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [19] */
		{142, 280, 158, 320},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{141, 336, 159, 413},
		UserItem {
			enabled
		},
		/* [21] */
		{166, 35, 183, 110},
		StaticText {
			disabled,
			"Long (Left):"
		},
		/* [22] */
		{167, 116, 183, 196},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{166, 114, 183, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [24] */
		{167, 147, 183, 174},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{166, 185, 183, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [26] */
		{167, 218, 183, 258},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{167, 218, 183, 238},
		EditText {
			enabled,
			""
		},
		/* [28] */
		{166, 249, 183, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [29] */
		{167, 280, 183, 320},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{166, 336, 184, 413},
		UserItem {
			enabled
		},
		/* [31] */
		{202, 23, 218, 109},
		StaticText {
			disabled,
			"Lat (Bottom):"
		},
		/* [32] */
		{202, 115, 218, 195},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{201, 113, 218, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [34] */
		{202, 146, 218, 173},
		EditText {
			enabled,
			""
		},
		/* [35] */
		{201, 184, 218, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [36] */
		{202, 217, 218, 257},
		EditText {
			enabled,
			""
		},
		/* [37] */
		{202, 217, 218, 237},
		EditText {
			enabled,
			""
		},
		/* [38] */
		{201, 248, 219, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [39] */
		{202, 279, 218, 319},
		EditText {
			enabled,
			""
		},
		/* [40] */
		{201, 336, 219, 413},
		UserItem {
			enabled
		},
		/* [41] */
		{227, 26, 244, 113},
		StaticText {
			disabled,
			"Long (Right):"
		},
		/* [42] */
		{227, 115, 243, 195},
		EditText {
			enabled,
			""
		},
		/* [43] */
		{226, 113, 243, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [44] */
		{227, 146, 243, 173},
		EditText {
			enabled,
			""
		},
		/* [45] */
		{226, 184, 243, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [46] */
		{227, 217, 243, 257},
		EditText {
			enabled,
			""
		},
		/* [47] */
		{227, 217, 243, 237},
		EditText {
			enabled,
			""
		},
		/* [48] */
		{226, 248, 243, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [49] */
		{227, 279, 243, 319},
		EditText {
			enabled,
			""
		},
		/* [50] */
		{226, 336, 244, 413},
		UserItem {
			enabled
		},
		/* [51] */
		{264, 16, 282, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [52] */
		{283, 16, 301, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [53] */
		{302, 16, 320, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [54] */
		{45, 17, 65, 127},
		Button {
			enabled,
			"Replace Map"
		},
		/* [55] */
		{129, 17, 259, 453},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (3000, "M30: PtCurMover Settings") {
	{	/* array DITLarray: 33 elements */
		/* [1] */
		{350, 293, 371, 366},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{346, 289, 375, 370},
		UserItem {
			disabled
		},
		/* [3] */
		{350, 172, 371, 245},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{350, 58, 371, 131},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{12, 15, 32, 98},
		StaticText {
			disabled,
			"User Name:"
		},
		/* [6] */
		{11, 98, 29, 398},
		StaticText {
			enabled,
			"name"
		},
		/* [7] */
		{46, 21, 64, 82},
		CheckBox {
			enabled,
			"Active"
		},
		/* [8] */
		{71, 21, 91, 200},
		CheckBox {
			enabled,
			"Show Velocities @ 1 in = "
		},
		/* [9] */
		{73, 207, 89, 244},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{73, 251, 89, 283},
		StaticText {
			disabled,
			"m/s"
		},
		/* [11] */
		{109, 22, 127, 168},
		StaticText {
			disabled,
			"Multiplicative Scalar:"
		},
		/* [12] */
		{107, 187, 123, 244},
		EditText {
			enabled,
			"2000000"
		},
		/* [13] */
		{168, 56, 187, 233},
		StaticText {
			disabled,
			"Along Current Uncertainty:"
		},
		/* [14] */
		{168, 241, 184, 316},
		EditText {
			enabled,
			"val"
		},
		/* [15] */
		{167, 322, 184, 340},
		StaticText {
			disabled,
			"%"
		},
		/* [16] */
		{197, 56, 214, 236},
		StaticText {
			disabled,
			"Cross Current Uncertainty:"
		},
		/* [17] */
		{197, 241, 213, 316},
		EditText {
			enabled,
			"val"
		},
		/* [18] */
		{197, 322, 214, 350},
		StaticText {
			disabled,
			"%"
		},
		/* [19] */
		{224, 56, 242, 180},
		StaticText {
			disabled,
			"Minimum Current:"
		},
		/* [20] */
		{224, 241, 240, 316},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{223, 322, 243, 376},
		StaticText {
			disabled,
			"m/sec"
		},
		/* [22] */
		{253, 57, 269, 132},
		StaticText {
			disabled,
			"Start Time:"
		},
		/* [23] */
		{250, 241, 266, 316},
		EditText {
			enabled,
			"0"
		},
		/* [24] */
		{251, 322, 267, 365},
		StaticText {
			disabled,
			"hours"
		},
		/* [25] */
		{279, 55, 295, 219},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [26] */
		{279, 241, 295, 316},
		EditText {
			enabled,
			"3"
		},
		/* [27] */
		{279, 322, 296, 368},
		StaticText {
			disabled,
			"hours"
		},
		/* [28] */
		{156, 23, 313, 390},
		UserItem {
			disabled
		},
		/* [29] */
		{146, 42, 163, 124},
		StaticText {
			disabled,
			"Uncertainty"
		},
		/* [30] */
		{74, 296, 90, 319},
		StaticText {
			disabled,
			"at"
		},
		/* [31] */
		{73, 329, 90, 374},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [32] */
		{74, 383, 90, 406},
		StaticText {
			disabled,
			"m"
		},
		/* [33] */
		{108, 263, 128, 411},
		Button {
			enabled,
			"Baromodes Input"
		}
	}
};

resource 'DITL' (3100, "M31: Spray Can Settings") {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{71, 166, 92, 243},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{67, 163, 96, 248},
		UserItem {
			disabled
		},
		/* [3] */
		{25, 45, 42, 122},
		RadioButton {
			enabled,
			"Small"
		},
		/* [4] */
		{49, 45, 66, 131},
		RadioButton {
			enabled,
			"Medium"
		},
		/* [5] */
		{72, 45, 91, 132},
		RadioButton {
			enabled,
			"Large"
		},
		/* [6] */
		{13, 26, 98, 146},
		UserItem {
			disabled
		},
		/* [7] */
		{6, 36, 22, 69},
		StaticText {
			disabled,
			"Size"
		}
	}
};

resource 'DITL' (1680, "Change Model Start Time ?") {
	{	/* array DITLarray: 6 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"Change"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 155, 121, 265},
		Button {
			enabled,
			"Don't Change"
		},
		/* [4] */
		{101, 36, 121, 136},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [6] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"Normal usage is to set the Model Start T"
			"ime equal to the Overflight Time.  \n\nCha"
			"nge the Model Start Time to the Overflig"
			"ht Time ?"
		}
	}
};

resource 'DITL' (1681, "Change Model Start Time ?") {
	{	/* array DITLarray: 6 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"Change"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 155, 121, 265},
		Button {
			enabled,
			"Don't Change"
		},
		/* [4] */
		{101, 36, 121, 136},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [6] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"Normal usage is to set the Model Start T"
			"ime equal to the Release Start Time of t"
			"he Spill.  \n\nChange the Model Start Time"
			" to the  Release Start Time ?"
		}
	}
};

resource 'DITL' (1690, "Are You Sure ?") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"Continue"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 165, 121, 265},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (1500, "M15: Overflight Info Dialog") {
	{	/* array DITLarray: 39 elements */
		/* [1] */
		{371, 419, 391, 477},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{367, 415, 395, 481},
		UserItem {
			disabled
		},
		/* [3] */
		{371, 336, 391, 399},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{371, 253, 391, 316},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{82, 25, 98, 137},
		UserItem {
			enabled
		},
		/* [6] */
		{82, 168, 98, 191},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{82, 197, 98, 253},
		UserItem {
			enabled
		},
		/* [8] */
		{116, 107, 132, 135},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{116, 153, 132, 181},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{110, 27, 142, 96},
		StaticText {
			disabled,
			"     Time:\n (24-hour)"
		},
		/* [11] */
		{116, 141, 132, 149},
		StaticText {
			disabled,
			":"
		},
		/* [12] */
		{180, 15, 196, 83},
		StaticText {
			disabled,
			"Pollutant:"
		},
		/* [13] */
		{179, 85, 197, 162},
		UserItem {
			enabled
		},
		/* [14] */
		{180, 309, 198, 372},
		StaticText {
			disabled,
			"# Splots:"
		},
		/* [15] */
		{181, 378, 197, 445},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [16] */
		{216, 15, 234, 134},
		StaticText {
			disabled,
			"Amount Released:"
		},
		/* [17] */
		{216, 141, 232, 216},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [18] */
		{215, 225, 233, 352},
		UserItem {
			enabled
		},
		/* [19] */
		{216, 376, 231, 396},
		StaticText {
			disabled,
			"at"
		},
		/* [20] */
		{215, 395, 233, 582},
		UserItem {
			enabled
		},
		/* [21] */
		{275, 25, 291, 137},
		UserItem {
			enabled
		},
		/* [22] */
		{275, 170, 291, 193},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{275, 202, 291, 268},
		UserItem {
			enabled
		},
		/* [24] */
		{309, 107, 325, 135},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{309, 153, 325, 181},
		EditText {
			enabled,
			""
		},
		/* [26] */
		{303, 27, 335, 96},
		StaticText {
			disabled,
			"     Time:\n (24-hour)"
		},
		/* [27] */
		{309, 141, 325, 149},
		StaticText {
			disabled,
			":"
		},
		/* [28] */
		{265, 15, 284, 116},
		StaticText {
			disabled,
			"Pollutant Age:"
		},
		/* [29] */
		{266, 135, 282, 180},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [30] */
		{266, 190, 284, 223},
		StaticText {
			disabled,
			"hrs"
		},
		/* [31] */
		{287, 15, 310, 262},
		StaticText {
			disabled,
			"( Overflight Time -  Time of Spill)"
		},
		/* [32] */
		{60, 14, 153, 389},
		UserItem {
			disabled
		},
		/* [33] */
		{52, 28, 68, 133},
		StaticText {
			disabled,
			"Overflight Time"
		},
		/* [34] */
		{253, 14, 346, 389},
		UserItem {
			disabled
		},
		/* [35] */
		{245, 28, 260, 110},
		StaticText {
			disabled,
			"Time of Spill"
		},
		/* [36] */
		{366, 13, 389, 173},
		UserItem {
			disabled
		},
		/* [37] */
		{180, 463, 200, 550},
		Button {
			enabled,
			"Windage"
		},
		/* [38] */
		{16, 19, 32, 100},
		StaticText {
			disabled,
			"Spill Name:"
		},
		/* [39] */
		{16, 102, 32, 407},
		EditText {
			enabled,
			"Edit Text"
		}
	}
};

resource 'DITL' (1685, "Exit Editing Mode ?") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 165, 121, 265},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (1750, "Choose BNA Map") {
	{	/* array DITLarray: 4 elements */
		/* [1] */
		{129, 193, 151, 260},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{125, 189, 155, 264},
		UserItem {
			disabled
		},
		/* [3] */
		{129, 112, 151, 179},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{21, 17, 112, 264},
		UserItem {
			enabled
		}
	}
};

resource 'DITL' (2500, "M25: Vector Map Settings") {
	{	/* array DITLarray: 55 elements */
		/* [1] */
		{333, 307, 353, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{329, 303, 357, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{333, 229, 353, 292},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{333, 154, 353, 217},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{11, 18, 27, 62},
		StaticText {
			disabled,
			"Name:"
		},
		/* [6] */
		{11, 71, 42, 362},
		StaticText {
			disabled,
			"map name"
		},
		/* [7] */
		{79, 18, 94, 132},
		StaticText {
			disabled,
			"Refloat Half Life:"
		},
		/* [8] */
		{79, 137, 95, 187},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{78, 200, 98, 257},
		StaticText {
			disabled,
			"hours"
		},
		/* [10] */
		{111, 17, 128, 236},
		CheckBox {
			enabled,
			"Extend Map Bounds"
		},
		/* [11] */
		{141, 47, 157, 112},
		StaticText {
			disabled,
			"Lat (Top):\n"
		},
		/* [12] */
		{142, 116, 158, 196},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{141, 114, 158, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [14] */
		{142, 147, 158, 174},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{141, 185, 158, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [16] */
		{142, 218, 158, 258},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{142, 218, 158, 238},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{141, 249, 159, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [19] */
		{142, 280, 158, 320},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{141, 336, 159, 413},
		UserItem {
			enabled
		},
		/* [21] */
		{166, 35, 183, 110},
		StaticText {
			disabled,
			"Long (Left):"
		},
		/* [22] */
		{167, 116, 183, 196},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{166, 114, 183, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [24] */
		{167, 147, 183, 174},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{166, 185, 183, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [26] */
		{167, 218, 183, 258},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{167, 218, 183, 238},
		EditText {
			enabled,
			""
		},
		/* [28] */
		{166, 249, 183, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [29] */
		{167, 280, 183, 320},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{166, 336, 184, 413},
		UserItem {
			enabled
		},
		/* [31] */
		{202, 23, 218, 109},
		StaticText {
			disabled,
			"Lat (Bottom):"
		},
		/* [32] */
		{202, 115, 218, 195},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{201, 113, 218, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [34] */
		{202, 146, 218, 173},
		EditText {
			enabled,
			""
		},
		/* [35] */
		{201, 184, 218, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [36] */
		{202, 217, 218, 257},
		EditText {
			enabled,
			""
		},
		/* [37] */
		{202, 217, 218, 237},
		EditText {
			enabled,
			""
		},
		/* [38] */
		{201, 248, 219, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [39] */
		{202, 279, 218, 319},
		EditText {
			enabled,
			""
		},
		/* [40] */
		{201, 336, 219, 413},
		UserItem {
			enabled
		},
		/* [41] */
		{227, 26, 244, 113},
		StaticText {
			disabled,
			"Long (Right):"
		},
		/* [42] */
		{227, 115, 243, 195},
		EditText {
			enabled,
			""
		},
		/* [43] */
		{226, 113, 243, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [44] */
		{227, 146, 243, 173},
		EditText {
			enabled,
			""
		},
		/* [45] */
		{226, 184, 243, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [46] */
		{227, 217, 243, 257},
		EditText {
			enabled,
			""
		},
		/* [47] */
		{227, 217, 243, 237},
		EditText {
			enabled,
			""
		},
		/* [48] */
		{226, 248, 243, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [49] */
		{227, 279, 243, 319},
		EditText {
			enabled,
			""
		},
		/* [50] */
		{226, 336, 244, 413},
		UserItem {
			enabled
		},
		/* [51] */
		{264, 16, 282, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [52] */
		{283, 16, 301, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [53] */
		{302, 16, 320, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [54] */
		{45, 17, 65, 127},
		Button {
			enabled,
			"Replace Map"
		},
		/* [55] */
		{129, 17, 259, 453},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (2000, "M20: Component Mover") {
	{	/* array DITLarray: 74 elements */
		/* [1] */
		{560, 412, 581, 485},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{556, 408, 585, 489},
		UserItem {
			disabled
		},
		/* [3] */
		{560, 326, 581, 399},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{560, 244, 581, 317},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{560, 24, 581, 133},
		Button {
			enabled,
			"Uncertainty..."
		},
		/* [6] */
		{32, 31, 53, 140},
		Button {
			enabled,
			"Load…"
		},
		/* [7] */
		{35, 148, 58, 487},
		StaticText {
			disabled,
			""
		},
		/* [8] */
		{65, 186, 82, 239},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{96, 139, 113, 192},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{94, 199, 115, 330},
		UserItem {
			disabled
		},
		/* [11] */
		{96, 413, 113, 466},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{174, 31, 195, 140},
		Button {
			enabled,
			"Load…"
		},
		/* [13] */
		{178, 142, 202, 486},
		StaticText {
			disabled,
			""
		},
		/* [14] */
		{207, 32, 226, 271},
		StaticText {
			disabled,
			"Wind direction relative to pattern 1:"
		},
		/* [15] */
		{207, 274, 225, 332},
		RadioButton {
			enabled,
			"+ 90°"
		},
		/* [16] */
		{207, 337, 225, 391},
		RadioButton {
			enabled,
			"- 90°"
		},
		/* [17] */
		{208, 417, 225, 444},
		StaticText {
			disabled,
			"150"
		},
		/* [18] */
		{238, 138, 255, 191},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{236, 198, 257, 330},
		UserItem {
			disabled
		},
		/* [20] */
		{238, 413, 255, 466},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{317, 33, 333, 61},
		StaticText {
			disabled,
			"Lat:"
		},
		/* [22] */
		{318, 83, 334, 163},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{317, 81, 334, 111},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [24] */
		{318, 114, 334, 141},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{317, 152, 334, 182},
		StaticText {
			disabled,
			"Min:"
		},
		/* [26] */
		{318, 185, 334, 225},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{318, 185, 334, 205},
		EditText {
			enabled,
			""
		},
		/* [28] */
		{317, 216, 335, 244},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [29] */
		{318, 247, 334, 287},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{317, 308, 335, 385},
		UserItem {
			enabled
		},
		/* [31] */
		{342, 33, 359, 72},
		StaticText {
			disabled,
			"Long:"
		},
		/* [32] */
		{343, 83, 359, 163},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{342, 81, 359, 111},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [34] */
		{343, 114, 359, 141},
		EditText {
			enabled,
			""
		},
		/* [35] */
		{342, 152, 359, 182},
		StaticText {
			disabled,
			"Min:"
		},
		/* [36] */
		{343, 185, 359, 225},
		EditText {
			enabled,
			""
		},
		/* [37] */
		{343, 185, 359, 205},
		EditText {
			enabled,
			""
		},
		/* [38] */
		{342, 216, 359, 244},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [39] */
		{343, 247, 359, 287},
		EditText {
			enabled,
			""
		},
		/* [40] */
		{342, 308, 360, 385},
		UserItem {
			enabled
		},
		/* [41] */
		{374, 31, 392, 160},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [42] */
		{374, 176, 392, 257},
		RadioButton {
			enabled,
			"deg/min"
		},
		/* [43] */
		{376, 278, 392, 385},
		RadioButton {
			enabled,
			"deg/min/sec"
		},
		/* [44] */
		{497, 30, 515, 295},
		CheckBox {
			enabled,
			"Use Averaged Winds (HABS Users)"
		},
		/* [45] */
		{433, 33, 452, 141},
		UserItem {
			enabled
		},
		/* [46] */
		{434, 233, 452, 495},
		StaticText {
			disabled,
			""
		},
		/* [47] */
		{461, 285, 480, 405},
		UserItem {
			enabled
		},
		/* [48] */
		{16, 15, 147, 505},
		UserItem {
			disabled
		},
		/* [49] */
		{160, 15, 286, 505},
		UserItem {
			disabled
		},
		/* [50] */
		{300, 15, 400, 505},
		UserItem {
			disabled
		},
		/* [51] */
		{8, 32, 24, 102},
		StaticText {
			disabled,
			"Pattern #1"
		},
		/* [52] */
		{96, 32, 112, 133},
		StaticText {
			disabled,
			"For wind speed "
		},
		/* [53] */
		{97, 354, 113, 408},
		StaticText {
			disabled,
			"scale to"
		},
		/* [54] */
		{97, 474, 112, 504},
		StaticText {
			disabled,
			"m/s"
		},
		/* [55] */
		{64, 31, 82, 179},
		StaticText {
			disabled,
			"Wind direction is from:"
		},
		/* [56] */
		{152, 32, 169, 104},
		StaticText {
			disabled,
			"Pattern #2"
		},
		/* [57] */
		{208, 402, 225, 412},
		StaticText {
			disabled,
			"="
		},
		/* [58] */
		{208, 445, 224, 467},
		StaticText {
			disabled,
			"°"
		},
		/* [59] */
		{238, 31, 254, 132},
		StaticText {
			disabled,
			"For wind speed "
		},
		/* [60] */
		{239, 354, 255, 408},
		StaticText {
			disabled,
			"scale to"
		},
		/* [61] */
		{495, 333, 515, 451},
		Button {
			enabled,
			"Set Parameters"
		},
		/* [62] */
		{520, 30, 538, 295},
		CheckBox {
			enabled,
			"Extrapolate Winds"
		},
		/* [63] */
		{414, 15, 541, 505},
		UserItem {
			disabled
		},
		/* [64] */
		{293, 32, 308, 146},
		StaticText {
			disabled,
			"Reference Point"
		},
		/* [65] */
		{124, 32, 140, 264},
		StaticText {
			disabled,
			"Unscaled Value at Reference Point:"
		},
		/* [66] */
		{124, 265, 140, 331},
		StaticText {
			disabled,
			"0ddffdf"
		},
		/* [67] */
		{124, 333, 140, 366},
		StaticText {
			disabled,
			"m/s"
		},
		/* [68] */
		{266, 32, 282, 264},
		StaticText {
			disabled,
			"Unscaled Value at Reference Point:"
		},
		/* [69] */
		{266, 266, 282, 332},
		StaticText {
			disabled,
			"0ddffdf"
		},
		/* [70] */
		{266, 334, 282, 367},
		StaticText {
			disabled,
			"m/s"
		},
		/* [71] */
		{64, 247, 84, 487},
		StaticText {
			disabled,
			" degrees true or text (e.g. , \"NNW\")"
		},
		/* [72] */
		{463, 175, 479, 275},
		StaticText {
			disabled,
			"Scale using : "
		},
		/* [73] */
		{434, 185, 451, 229},
		StaticText {
			enabled,
			"Name:"
		},
		/* [74] */
		{239, 474, 254, 504},
		StaticText {
			disabled,
			"m/s"
		},
		/* [75] */
		{407, 33, 422, 141},
		StaticText {
			disabled,
			"Time file / mover"
		}
	}
};

resource 'DITL' (2075, "Compound Mover") {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{542, 412, 563, 485},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{538, 408, 567, 489},
		UserItem {
			disabled
		},
		/* [3] */
		{542, 326, 563, 399},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{542, 244, 563, 317},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{274, 29, 295, 138},
		Button {
			enabled,
			"Delete"
		},
		/* [6] */
		{181, 31, 202, 140},
		Button {
			enabled,
			"Load…"
		},
		/* [7] */
		{184, 148, 207, 487},
		StaticText {
			disabled,
			""
		},
		/* [8] */
		{314, 31, 335, 140},
		Button {
			enabled,
			"Delete All"
		},
		/* [9] */
		{51, 15, 162, 490},
		UserItem {
			enabled
		},
		/* [10] */
		{29, 32, 45, 142},
		StaticText {
			disabled,
			"Pattern Name"
		},
		/* [11] */
		{234, 31, 255, 140},
		Button {
			enabled,
			"Move Up"
		},
		/* [12] */
		{29, 192, 45, 467},
		StaticText {
			disabled,
			"(Currents are listed in order of priority)"
		}
	}
};

resource 'DITL' (3150, "ADCP Mover") {
	{	/* array DITLarray: 14 elements */
		/* [1] */
		{542, 412, 563, 485},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{538, 408, 567, 489},
		UserItem {
			disabled
		},
		/* [3] */
		{542, 326, 563, 399},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{542, 244, 563, 317},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{274, 29, 295, 138},
		Button {
			enabled,
			"Delete"
		},
		/* [6] */
		{181, 31, 202, 140},
		Button {
			enabled,
			"Load…"
		},
		/* [7] */
		{184, 148, 207, 487},
		StaticText {
			disabled,
			""
		},
		/* [8] */
		{314, 31, 335, 140},
		Button {
			enabled,
			"Delete All"
		},
		/* [9] */
		{51, 15, 162, 490},
		UserItem {
			enabled
		},
		/* [10] */
		{29, 32, 45, 132},
		StaticText {
			disabled,
			"Station Name"
		},
		/* [11] */
		{234, 31, 255, 140},
		Button {
			enabled,
			"Move Up"
		},
		/* [12] */
		{29, 142, 45, 200},
		StaticText {
			disabled,
			"Top Bin"
		},
		/* [13] */
		{359, 31, 380, 100},
		StaticText {
			disabled,
			"Use Bin # "
		},
		/* [14] */
		{359, 120, 380, 160},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{29, 210, 45, 280},
		StaticText {
			disabled,
			"Top Depth"
		},
		/* [16] */
		{29, 300, 45, 360},
		StaticText {
			disabled,
			"Bin Size"
		},
		/* [17] */
		{29, 370, 45, 470},
		StaticText {
			disabled,
			"Station Depth"
		},
		/* [18] */
		{359, 180, 380, 440},
		StaticText {
			disabled,
			"(Bin 1 is the surface, Bin 0 uses all bins)"
		}
	}
};

resource 'DITL' (1682, "Change Model Start Time ?") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"Change"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 155, 121, 265},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (1683, "Change Model Start Time ?") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{101, 295, 121, 395},
		Button {
			enabled,
			"Change"
		},
		/* [2] */
		{97, 291, 126, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{101, 155, 121, 265},
		Button {
			enabled,
			"Don't Change"
		},
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{15, 58, 89, 418},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (1600, "M16: CATS Mover Settings") {
	{	/* array DITLarray: 61 elements */
		/* [1] */
		{393, 310, 414, 383},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{389, 306, 418, 387},
		UserItem {
			disabled
		},
		/* [3] */
		{393, 230, 414, 298},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{393, 148, 414, 211},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{13, 104, 29, 409},
		StaticText {
			disabled,
			"name"
		},
		/* [6] */
		{36, 21, 54, 82},
		CheckBox {
			enabled,
			"Active"
		},
		/* [7] */
		{61, 21, 81, 200},
		CheckBox {
			enabled,
			"Show Velocities @ 1 in = "
		},
		/* [8] */
		{117, 42, 136, 124},
		StaticText {
			disabled,
			"File Name : "
		},
		/* [9] */
		{118, 126, 136, 384},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [10] */
		{222, 27, 240, 232},
		RadioButton {
			enabled,
			"No Reference Point Scaling"
		},
		/* [11] */
		{243, 27, 261, 102},
		RadioButton {
			enabled,
			"Scale To:"
		},
		/* [12] */
		{244, 113, 260, 188},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [13] */
		{266, 27, 284, 174},
		RadioButton {
			enabled,
			"Scale To Other Grid:"
		},
		/* [14] */
		{266, 177, 282, 252},
		StaticText {
			disabled,
			"name"
		},
		/* [15] */
		{299, 27, 315, 55},
		StaticText {
			disabled,
			"Lat:"
		},
		/* [16] */
		{300, 77, 316, 157},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{299, 75, 316, 105},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [18] */
		{300, 108, 316, 135},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{299, 146, 316, 176},
		StaticText {
			disabled,
			"Min:"
		},
		/* [20] */
		{300, 179, 316, 219},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{300, 179, 316, 199},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{299, 210, 317, 238},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [23] */
		{300, 241, 316, 281},
		EditText {
			enabled,
			""
		},
		/* [24] */
		{299, 302, 317, 379},
		UserItem {
			enabled
		},
		/* [25] */
		{324, 27, 341, 66},
		StaticText {
			disabled,
			"Long:"
		},
		/* [26] */
		{325, 77, 341, 157},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{324, 75, 341, 105},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [28] */
		{325, 108, 341, 135},
		EditText {
			enabled,
			""
		},
		/* [29] */
		{324, 146, 341, 176},
		StaticText {
			disabled,
			"Min:"
		},
		/* [30] */
		{325, 179, 341, 219},
		EditText {
			enabled,
			""
		},
		/* [31] */
		{325, 179, 341, 199},
		EditText {
			enabled,
			""
		},
		/* [32] */
		{324, 210, 341, 238},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [33] */
		{325, 241, 341, 281},
		EditText {
			enabled,
			""
		},
		/* [34] */
		{324, 302, 342, 379},
		UserItem {
			enabled
		},
		/* [35] */
		{352, 27, 370, 156},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [36] */
		{352, 172, 370, 253},
		RadioButton {
			enabled,
			"deg/min"
		},
		/* [37] */
		{354, 274, 370, 381},
		RadioButton {
			enabled,
			"deg/min/sec"
		},
		/* [38] */
		{178, 15, 377, 414},
		UserItem {
			disabled
		},
		/* [39] */
		{63, 207, 79, 282},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [40] */
		{194, 251, 210, 317},
		StaticText {
			disabled,
			"0ddffdf"
		},
		/* [41] */
		{13, 21, 29, 102},
		StaticText {
			disabled,
			"Current File:"
		},
		/* [42] */
		{63, 287, 79, 319},
		StaticText {
			disabled,
			"m/s"
		},
		/* [43] */
		{171, 22, 186, 127},
		StaticText {
			disabled,
			"Reference Point"
		},
		/* [44] */
		{194, 28, 210, 260},
		StaticText {
			disabled,
			"Unscaled Value at Reference Point:"
		},
		/* [45] */
		{244, 193, 261, 384},
		StaticText {
			disabled,
			"* file value at reference point"
		},
		/* [46] */
		{193, 319, 209, 352},
		StaticText {
			disabled,
			"m/s"
		},
		/* [47] */
		{393, 20, 414, 129},
		Button {
			enabled,
			"Uncertainty..."
		},
		/* [48] */
		{142, 41, 158, 141},
		StaticText {
			disabled,
			"Time File Units:"
		},
		/* [49] */
		{142, 142, 158, 370},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [50] */
		{142, 41, 158, 188},
		StaticText {
			disabled,
			"Time File Scale Factor:"
		},
		/* [51] */
		{142, 207, 158, 282},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [52] */
		{142, 41, 158, 231},
		StaticText {
			disabled,
			"Hydrology File Scale Factor :"
		},
		/* [53] */
		{142, 236, 158, 304},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [54] */
		{90, 21, 109, 140},
		UserItem {
			enabled
		},
		/* [55] */
		{142, 41, 158, 191},
		StaticText {
			disabled,
			"Integrated Transport :"
		},
		/* [56] */
		{142, 191, 158, 259},
		StaticText {
			disabled,
			""
		},
		/* [57] */
		{142, 259, 158, 307},
		StaticText {
			disabled,
			"m3/s"
		},
		/* [58] */
		{244, 27, 260, 96},
		StaticText {
			disabled,
			"Scale To :"
		},
		/* [59] */
		{244, 96, 260, 164},
		StaticText {
			disabled,
			""
		},
		/* [60] */
		{244, 164, 260, 312},
		StaticText {
			disabled,
			"m/s at reference point"
		},
		/* [61] */
		{37, 298, 57, 426},
		Button {
			enabled,
			"Replace Current"
		}
	}
};

resource 'DITL' (1687, "Download Location File") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{90, 305, 110, 395},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{86, 301, 115, 401},
		UserItem {
			disabled
		},
		/* [3] */
		/*{90, 175, 110, 265},
		Button {
			disabled,
			"Help"
		},*/
		/* [4] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [5] */
		{15, 58, 95, 428},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (3200) {
	{	/* array DITLarray: 20 elements */
		/* [1] */
		{288, 309, 309, 382},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{284, 305, 313, 386},
		UserItem {
			disabled
		},
		/* [3] */
		{288, 224, 309, 297},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{288, 144, 309, 217},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{13, 20, 29, 111},
		StaticText {
			disabled,
			"Station Name:"
		},
		/* [6] */
		{13, 113, 29, 408},
		StaticText {
			disabled,
			"name"
		},
		/* [7] */
		{36, 20, 54, 81},
		StaticText {
			enabled,
			"Position:"
		},
		/* [8] */
		{36, 96, 52, 328},
		StaticText {
			disabled,
			"m/s"
		},
		/* [9] */
		{61, 20, 81, 69},
		StaticText {
			enabled,
			"Units:"
		},
		/* [10] */
		{61, 78, 77, 153},
		StaticText {
			enabled,
			"Units"
		},
		/* [11] */
		{90, 19, 155, 404},
		StaticText {
			disabled,
			"In order to correctly scale the current "
			"pattern to the hydrology time series Gno"
			"me needs integrated transport or scaling"
			" information.  Please select the type of"
			" information you have:"
		},
		/* [12] */
		{168, 19, 187, 138},
		UserItem {
			enabled
		},
		/* [13] */
		{198, 16, 218, 414},
		StaticText {
			disabled,
			"The hydrology current pattern has an int"
			"egrated transport of "
		},
		/* [14] */
		{227, 20, 243, 65},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{224, 78, 245, 120},
		UserItem {
			disabled
		},
		/* [16] */
		{220, 209, 250, 381},
		StaticText {
			disabled,
			"across a section through the reference p"
			"oint"
		},
		/* [17] */
		{199, 16, 213, 372},
		StaticText {
			disabled,
			"When the hydrology time series shows a t"
			"ransport of "
		},
		/* [18] */
		{225, 209, 241, 460},
		StaticText {
			disabled,
			"the reference point velocity should be"
		},
		/* [19] */
		{258, 19, 274, 65},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{255, 78, 276, 206},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (1850) {
	{	/* array DITLarray: 13 elements */
		/* [1] */
		{132, 204, 152, 270},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{128, 200, 156, 274},
		UserItem {
			disabled
		},
		/* [3] */
		{132, 125, 152, 191},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{132, 37, 152, 103},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{40, 144, 56, 299},
		UserItem {
			enabled
		},
		/* [6] */
		{69, 125, 85, 165},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{68, 194, 85, 234},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{18, 20, 107, 309},
		UserItem {
			disabled
		},
		/* [9] */
		{39, 54, 56, 138},
		StaticText {
			disabled,
			"Persistence:"
		},
		/* [10] */
		{68, 55, 85, 118},
		StaticText {
			disabled,
			"Windage:"
		},
		/* [11] */
		{69, 238, 85, 254},
		StaticText {
			disabled,
			"%"
		},
		/* [12] */
		{68, 171, 85, 187},
		StaticText {
			disabled,
			"to"
		},
		/* [13] */
		{10, 24, 27, 79},
		StaticText {
			disabled,
			"Settings"
		}
	}
};

resource 'DITL' (1688, "Have Topology File ?") {
	{	/* array DITLarray: 6 elements */
		/* [1] */
		{103, 305, 123, 395},
		Button {
			enabled,
			"Yes"
		},
		/* [2] */
		{99, 301, 128, 401},
		UserItem {
			disabled
		},
		/* [3] */
		{103, 175, 123, 265},
		Button {
			enabled,
			"No"
		},
		/* [4] */
		{103, 43, 123, 143},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [6] */
		{15, 58, 88, 428},
		StaticText {
			disabled,
			"^0"
		}
	}
};

resource 'DITL' (3900) {
	{	/* array DITLarray: 58 elements */
		/* [1] */
		{287, 528, 307, 586},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{283, 524, 311, 590},
		UserItem {
			disabled
		},
		/* [3] */
		{287, 454, 307, 512},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{287, 384, 307, 442},
		Button {
			disabled,
			"Help..."
		},
		/* [5] */
		{20, 449, 36, 524},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{130, 344, 146, 384},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{130, 390, 148, 441},
		UserItem {
			disabled
		},
		/* [8] */
		{18, 88, 36, 165},
		UserItem {
			enabled
		},
		/* [9] */
		{94, 21, 110, 133},
		UserItem {
			enabled
		},
		/* [10] */
		{94, 167, 110, 190},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{94, 197, 110, 263},
		UserItem {
			enabled
		},
		/* [12] */
		{121, 99, 137, 127},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{121, 145, 137, 173},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{79, 312, 95, 337},
		StaticText {
			disabled,
			"Lat:"
		},
		/* [15] */
		{80, 342, 96, 422},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{79, 340, 96, 370},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [17] */
		{80, 373, 96, 400},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{79, 411, 96, 441},
		StaticText {
			disabled,
			"Min:"
		},
		/* [19] */
		{80, 444, 96, 484},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{80, 444, 96, 464},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{79, 475, 97, 503},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [22] */
		{80, 506, 96, 546},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{79, 563, 97, 640},
		UserItem {
			enabled
		},
		/* [24] */
		{104, 301, 121, 336},
		StaticText {
			disabled,
			"Long:"
		},
		/* [25] */
		{105, 342, 121, 422},
		EditText {
			enabled,
			""
		},
		/* [26] */
		{104, 340, 121, 370},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [27] */
		{105, 373, 121, 400},
		EditText {
			enabled,
			""
		},
		/* [28] */
		{104, 411, 121, 441},
		StaticText {
			disabled,
			"Min:"
		},
		/* [29] */
		{105, 444, 121, 484},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{105, 444, 121, 464},
		EditText {
			enabled,
			""
		},
		/* [31] */
		{104, 475, 121, 503},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [32] */
		{105, 506, 121, 546},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{104, 563, 122, 640},
		UserItem {
			enabled
		},
		/* [34] */
		{202, 23, 218, 135},
		UserItem {
			enabled
		},
		/* [35] */
		{201, 149, 217, 172},
		EditText {
			enabled,
			""
		},
		/* [36] */
		{200, 179, 216, 245},
		UserItem {
			enabled
		},
		/* [37] */
		{230, 101, 246, 129},
		EditText {
			enabled,
			""
		},
		/* [38] */
		{230, 147, 246, 175},
		EditText {
			enabled,
			""
		},
		/* [39] */
		{168, 16, 186, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [40] */
		{187, 16, 205, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [41] */
		{206, 16, 224, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [42] */
		{72, 12, 155, 675},
		UserItem {
			disabled
		},
		/* [43] */
		{178, 11, 265, 261},
		UserItem {
			disabled
		},
		/* [44] */
		{121, 133, 137, 141},
		StaticText {
			disabled,
			":"
		},
		/* [45] */
		{230, 135, 246, 143},
		StaticText {
			disabled,
			":"
		},
		/* [46] */
		{19, 382, 36, 444},
		StaticText {
			disabled,
			"# Splots:"
		},
		/* [47] */
		{19, 19, 35, 87},
		StaticText {
			disabled,
			"Pollutant:"
		},
		/* [48] */
		{64, 26, 80, 122},
		StaticText {
			disabled,
			"Release Start:"
		},
		/* [49] */
		{170, 22, 187, 111},
		StaticText {
			enabled,
			"Release End:"
		},
		/* [50] */
		{131, 295, 147, 339},
		StaticText {
			disabled,
			"Depth:"
		},
		/* [51] */
		{118, 23, 150, 92},
		StaticText {
			disabled,
			"Start Time:\n (24-hour)"
		},
		/* [52] */
		{227, 29, 259, 94},
		StaticText {
			disabled,
			"End Time:\n(24-hour)"
		},
		/* [53] */
		{17, 551, 37, 638},
		Button {
			enabled,
			"Windage"
		},
		/* [54] */
		{175, 375, 195, 625},
		Button {
			enabled,
			"Deepwater Spill Release Parameters"
		},
		/* [55] */
		{203, 410, 223, 590},
		Button {
			enabled,
			"Circulation Information"
		},
		/* [56] */
		{234, 441, 254, 559},
		Button {
			enabled,
			"Output Options"
		},
		/* [57] */
		{18, 19, 34, 100},
		StaticText {
			disabled,
			"Spill Name:"
		},
		/* [58] */
		{18, 102, 34, 357},
		EditText {
			enabled,
			"Edit Text"
		}
	}
};

resource 'DITL' (3920) {
	{	/* array DITLarray: 8 elements */
		/* [1] */
		{162, 247, 182, 313},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{158, 243, 186, 317},
		UserItem {
			disabled
		},
		/* [3] */
		{162, 168, 182, 234},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{162, 93, 182, 159},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{17, 17, 36, 288},
		StaticText {
			disabled,
			"Select Temperature and Salinity Profiles"
		},
		/* [6] */
		{45, 42, 67, 185},
		UserItem {
			disabled
		},
		/* [7] */
		{75, 24, 106, 279},
		StaticText {
			disabled,
			""
		},
		/* [8] */
		{115, 24, 146, 279},
		StaticText {
			disabled,
			""
		}
	}
};

resource 'DITL' (3910, "Diffusivity and Time Step") {
	{	/* array DITLarray: 13 elements */
		/* [1] */
		{161, 204, 181, 270},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{157, 200, 185, 274},
		UserItem {
			disabled
		},
		/* [3] */
		{161, 120, 181, 186},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{161, 37, 181, 103},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{69, 18, 88, 162},
		StaticText {
			disabled,
			"Horizontal Diffusivity :"
		},
		/* [6] */
		{69, 168, 85, 226},
		EditText {
			disabled,
			""
		},
		/* [7] */
		{69, 231, 88, 311},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [8] */
		{102, 18, 121, 152},
		StaticText {
			disabled,
			"Vertical Diffusivity :"
		},
		/* [9] */
		{101, 161, 117, 219},
		EditText {
			disabled,
			""
		},
		/* [10] */
		{102, 225, 121, 305},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [11] */
		{25, 18, 40, 232},
		StaticText {
			disabled,
			"Time Step of Advection/Diffusion:"
		},
		/* [12] */
		{25, 234, 41, 292},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{24, 296, 44, 353},
		StaticText {
			disabled,
			"minutes"
		}
	}
};

resource 'DITL' (3950) {
	{	/* array DITLarray: 49 elements */
		/* [1] */
		{246, 307, 266, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{242, 303, 270, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{246, 229, 266, 292},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{246, 154, 266, 217},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{24, 17, 41, 170},
		CheckBox {
			enabled,
			"Set Map Box Bounds"
		},
		/* [6] */
		{54, 47, 70, 112},
		StaticText {
			disabled,
			"Lat (Top):\n"
		},
		/* [7] */
		{55, 116, 71, 196},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{54, 114, 71, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [9] */
		{55, 147, 71, 174},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{54, 185, 71, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [11] */
		{55, 218, 71, 258},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{55, 218, 71, 238},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{54, 249, 72, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [14] */
		{55, 280, 71, 320},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{54, 336, 72, 413},
		UserItem {
			enabled
		},
		/* [16] */
		{79, 35, 96, 110},
		StaticText {
			disabled,
			"Long (Left):"
		},
		/* [17] */
		{80, 116, 96, 196},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{79, 114, 96, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [19] */
		{80, 147, 96, 174},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{79, 185, 96, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [21] */
		{80, 218, 96, 258},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{80, 218, 96, 238},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{79, 249, 96, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [24] */
		{80, 280, 96, 320},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{79, 336, 97, 413},
		UserItem {
			enabled
		},
		/* [26] */
		{115, 23, 131, 109},
		StaticText {
			disabled,
			"Lat (Bottom):"
		},
		/* [27] */
		{115, 115, 131, 195},
		EditText {
			enabled,
			""
		},
		/* [28] */
		{114, 113, 131, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [29] */
		{115, 146, 131, 173},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{114, 184, 131, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [31] */
		{115, 217, 131, 257},
		EditText {
			enabled,
			""
		},
		/* [32] */
		{115, 217, 131, 237},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{114, 248, 132, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [34] */
		{115, 279, 131, 319},
		EditText {
			enabled,
			""
		},
		/* [35] */
		{114, 336, 132, 413},
		UserItem {
			enabled
		},
		/* [36] */
		{140, 26, 157, 113},
		StaticText {
			disabled,
			"Long (Right):"
		},
		/* [37] */
		{140, 115, 156, 195},
		EditText {
			enabled,
			""
		},
		/* [38] */
		{139, 113, 156, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [39] */
		{140, 146, 156, 173},
		EditText {
			enabled,
			""
		},
		/* [40] */
		{139, 184, 156, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [41] */
		{140, 217, 156, 257},
		EditText {
			enabled,
			""
		},
		/* [42] */
		{140, 217, 156, 237},
		EditText {
			enabled,
			""
		},
		/* [43] */
		{139, 248, 156, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [44] */
		{140, 279, 156, 319},
		EditText {
			enabled,
			""
		},
		/* [45] */
		{139, 336, 157, 413},
		UserItem {
			enabled
		},
		/* [46] */
		{177, 16, 195, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [47] */
		{196, 16, 214, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [48] */
		{215, 16, 233, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [49] */
		{42, 17, 172, 453},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (3930) {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{125, 224, 147, 304},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{121, 220, 151, 308},
		UserItem {
			disabled
		},
		/* [3] */
		{127, 138, 147, 204},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{127, 59, 147, 125},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{14, 21, 33, 255},
		StaticText {
			disabled,
			"Hydrodynamic Information for CDOG"
		},
		/* [6] */
		{42, 37, 64, 290},
		UserItem {
			disabled
		},
		/* [7] */
		{73, 29, 106, 304},
		StaticText {
			disabled,
			""
		}
	}
};

resource 'DITL' (3940) {
	{	/* array DITLarray: 30 elements */
		/* [1] */
		{315, 252, 337, 332},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{311, 248, 341, 336},
		UserItem {
			disabled
		},
		/* [3] */
		{315, 152, 337, 232},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{315, 59, 337, 139},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{16, 12, 31, 136},
		StaticText {
			disabled,
			"Diameter of Orifice: "
		},
		/* [6] */
		{16, 143, 32, 201},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{15, 210, 35, 232},
		StaticText {
			disabled,
			"m"
		},
		/* [8] */
		{108, 12, 123, 200},
		StaticText {
			disabled,
			"Temp of  Discharged Mixture: "
		},
		/* [9] */
		{108, 214, 124, 272},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{107, 281, 127, 338},
		StaticText {
			disabled,
			"deg C"
		},
		/* [11] */
		{137, 12, 152, 285},
		StaticText {
			disabled,
			"Density of Product at Average Water Temp"
			":"
		},
		/* [12] */
		{137, 291, 153, 349},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{136, 358, 156, 415},
		StaticText {
			disabled,
			"kg/m**3"
		},
		/* [14] */
		{163, 12, 178, 86},
		StaticText {
			disabled,
			"Gas Type:"
		},
		/* [15] */
		{163, 87, 179, 179},
		UserItem {
			enabled
		},
		/* [16] */
		{219, 235, 239, 372},
		StaticText {
			disabled,
			"kg/m**3    (900-940)"
		},
		/* [17] */
		{47, 12, 62, 146},
		StaticText {
			disabled,
			"Initial bubble radius: "
		},
		/* [18] */
		{47, 153, 63, 211},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{46, 220, 66, 352},
		StaticText {
			disabled,
			"mm     (max 5mm)"
		},
		/* [20] */
		{191, 37, 206, 201},
		StaticText {
			disabled,
			"Molecular weight of gas:"
		},
		/* [21] */
		{191, 204, 207, 262},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{280, 12, 295, 206},
		StaticText {
			disabled,
			"Separation of gas from plume:"
		},
		/* [23] */
		{280, 211, 296, 295},
		UserItem {
			enabled
		},
		/* [24] */
		{220, 37, 235, 161},
		StaticText {
			disabled,
			"Density of Hydrate:"
		},
		/* [25] */
		{220, 168, 236, 226},
		EditText {
			enabled,
			""
		},
		/* [26] */
		{252, 12, 267, 126},
		StaticText {
			disabled,
			"Hydrate process:"
		},
		/* [27] */
		{252, 133, 268, 221},
		UserItem {
			enabled
		},
		/* [28] */
		{75, 12, 90, 86},
		StaticText {
			disabled,
			"Drop size: "
		},
		/* [29] */
		{75, 93, 91, 251},
		UserItem {
			enabled
		},
		/* [30] */
		{190, 273, 206, 330},
		StaticText {
			disabled,
			"kg/mol"
		}
	}
};

resource 'DITL' (3806) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a temperature profile file:"
		}
	}
};

resource 'DITL' (3807) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a salinity profile file:"
		}
	}
};

resource 'DITL' (3960, "CDOG Output Options") {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{132, 237, 154, 317},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{128, 233, 158, 321},
		UserItem {
			disabled
		},
		/* [3] */
		{132, 137, 154, 217},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{132, 44, 154, 124},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{8, 28, 55, 296},
		StaticText {
			disabled,
			"CDOG always outputs the surface oil part"
			"icles for use in GNOME.\nSelect any addit"
			"ional desired output data"
		},
		/* [6] */
		{68, 29, 86, 235},
		CheckBox {
			enabled,
			"Subsurface Particles"
		},
		/* [7] */
		{98, 29, 116, 235},
		CheckBox {
			enabled,
			"Gas Files"
		}
	}
};

resource 'DITL' (3808) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 272},
		StaticText {
			disabled,
			"Select a file from the folder of current"
			"s:"
		}
	}
};

resource 'DITL' (3970) {
	{	/* array DITLarray: 38 elements */
		/* [1] */
		{391, 316, 416, 389},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{387, 312, 420, 393},
		UserItem {
			disabled
		},
		/* [3] */
		{391, 232, 416, 306},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{391, 148, 416, 222},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{201, 209, 226, 329},
		Button {
			enabled,
			"Delete Selected"
		},
		/* [6] */
		{391, 9, 416, 130},
		Button {
			enabled,
			"Delete All"
		},
		/* [7] */
		{201, 63, 226, 188},
		Button {
			enabled,
			"Replace Selected"
		},
		/* [8] */
		{29, 63, 45, 110},
		StaticText {
			enabled,
			"Depth:"
		},
		/* [9] */
		{30, 118, 46, 158},
		EditText {
			disabled,
			""
		},
		/* [10] */
		{30, 170, 46, 330},
		StaticText {
			enabled,
			"meters (z positive down)"
		},
		/* [11] */
		{60, 94, 76, 142},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{60, 188, 76, 236},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{93, 163, 109, 201},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{129, 131, 145, 185},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{171, 237, 187, 271},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{245, 62, 260, 103},
		StaticText {
			disabled,
			"Depth"
		},
		/* [17] */
		{245, 123, 260, 136},
		StaticText {
			disabled,
			"U"
		},
		/* [18] */
		{245, 243, 260, 278},
		StaticText {
			disabled,
			"Temp"
		},
		/* [19] */
		{245, 298, 260, 320},
		StaticText {
			disabled,
			"Sal"
		},
		/* [20] */
		{263, 57, 378, 342},
		UserItem {
			enabled
		},
		/* [21] */
		{450, 534, 660, 744},
		Picture {
			disabled,
			1431
		},
		/* [22] */
		{172, 63, 187, 229},
		StaticText {
			disabled,
			"Auto-increment depth by:"
		},
		/* [23] */
		{510, 323, 527, 328},
		StaticText {
			disabled,
			""
		},
		/* [24] */
		{61, 63, 78, 90},
		StaticText {
			disabled,
			"U:"
		},
		/* [25] */
		{93, 63, 109, 155},
		StaticText {
			disabled,
			"Temperature:"
		},
		/* [26] */
		{129, 63, 148, 118},
		StaticText {
			disabled,
			"Salinity:"
		},
		/* [27] */
		{172, 280, 189, 327},
		StaticText {
			disabled,
			"meters"
		},
		/* [28] */
		{509, 501, 525, 512},
		StaticText {
			disabled,
			"/"
		},
		/* [29] */
		{491, 476, 508, 487},
		StaticText {
			disabled,
			"/"
		},
		/* [30] */
		{61, 157, 78, 184},
		StaticText {
			disabled,
			"V:"
		},
		/* [31] */
		{93, 209, 111, 289},
		StaticText {
			disabled,
			"degrees C"
		},
		/* [32] */
		{493, 380, 511, 459},
		StaticText {
			disabled,
			""
		},
		/* [33] */
		{512, 354, 533, 474},
		Button {
			enabled,
			"Load Wind Data..."
		},
		/* [34] */
		{128, 198, 148, 263},
		StaticText {
			disabled,
			"psu"
		},
		/* [35] */
		{245, 170, 260, 182},
		StaticText {
			disabled,
			"V"
		},
		/* [36] */
		{245, 211, 260, 225},
		StaticText {
			disabled,
			"W"
		},
		/* [37] */
		{61, 251, 77, 291},
		StaticText {
			enabled,
			"m/s"
		},
		/* [38] */
		{11, 57, 233, 342},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (3980, "About CDOG") {
	{	/* array DITLarray: 8 elements */
		/* [1] */
		{339, 215, 363, 300},
		Button {
			enabled,
			"Done"
		},
		/* [2] */
		{336, 211, 366, 304},
		UserItem {
			disabled
		},
		/* [3] */
		{337, 60, 366, 129},
		Button {
			enabled,
			"CANCEL"
		},
		/* [4] */
		{14, 19, 94, 364},
		StaticText {
			disabled,
			"CDOG  (Clarkson Deepwater Oil and Gas Mo"
			"del 2001) was developed at the Departmen"
			"t of Civil and Environmental Engineering"
			", Clarkson University, Potsdam, NY under"
			" the support of  the Minerals Management"
			" Service (MMS) of the U.S. Department"
		},
		/* [5] */
		{205, 19, 270, 373},
		StaticText {
			disabled,
			"Clarkson University, program developers,"
			" MMS, and DSTF make no warranties, expre"
			"ssed or implied, concerning the accuracy"
			", completeness, reliability, usability, "
			"or suitability for any particular purpos"
			"e of the information\n"
		},
		/* [6] */
		{127, 19, 199, 357},
		StaticText {
			disabled,
			"The program was developed under the dire"
			"ction of Dr. Poojitha D. Yapa. Programs "
			"were written by Dr. Li Zheng and Fanghui"
			" Chen. Last day of modification: Novembe"
			"r, 2001.  Copyrights (c) reserved 2001."
		},
		/* [7] */
		{271, 19, 322, 372},
		StaticText {
			disabled,
			"information and data contained in this p"
			"rogram or furnished in connection therew"
			"ith. The user agrees not to assert any p"
			"roprietary right to the program."
		},
		/* [8] */
		{95, 19, 127, 372},
		StaticText {
			disabled,
			"of the Interior, in collaboration with t"
			"he Deep Spill Task Force (DSTF), a conso"
			"rtium of many oil companies."
		}
	}
};

resource 'DITL' (1000, "M10: Model Times Dialog") {
	{	/* array DITLarray: 26 elements */
		/* [1] */
		{350, 270, 371, 347},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{346, 266, 375, 351},
		UserItem {
			disabled
		},
		/* [3] */
		{350, 169, 371, 246},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{350, 75, 371, 152},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{35, 26, 51, 143},
		StaticText {
			disabled,
			"Model Start Date:"
		},
		/* [6] */
		{33, 146, 52, 200},
		UserItem {
			disabled
		},
		/* [7] */
		{36, 290, 52, 319},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{33, 328, 52, 366},
		UserItem {
			disabled
		},
		/* [9] */
		{73, 159, 89, 187},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{73, 205, 89, 233},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{68, 26, 102, 148},
		StaticText {
			disabled,
			"Model Start Time:\n        (24-hour)"
		},
		/* [12] */
		{73, 193, 89, 201},
		StaticText {
			disabled,
			":"
		},
		/* [13] */
		{108, 26, 128, 164},
		StaticText {
			disabled,
			"Model Run Duration:"
		},
		/* [14] */
		{107, 173, 124, 213},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{108, 219, 124, 282},
		StaticText {
			disabled,
			"days and"
		},
		/* [16] */
		{107, 286, 124, 331},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{108, 337, 124, 384},
		StaticText {
			disabled,
			"hours"
		},
		/* [18] */
		{144, 9, 183, 338},
		StaticText {
			disabled,
			"GNOME will automatically run the Best Es"
			"timate (Forecast) solution."
		},
		/* [19] */
		{189, 9, 211, 369},
		CheckBox {
			enabled,
			"Include the Minimum Regret (Uncertainty)"
			" solution.   "
		},
		/* [20] */
		{248, 9, 268, 164},
		StaticText {
			disabled,
			"Computation Time Step:"
		},
		/* [21] */
		{249, 170, 265, 230},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{249, 238, 267, 288},
		StaticText {
			disabled,
			"hours"
		},
		/* [23] */
		{275, 9, 298, 176},
		CheckBox {
			enabled,
			"Prevent Land Jumping"
		},
		/* [24] */
		{16, 10, 135, 438},
		UserItem {
			disabled
		},
		/* [25] */
		{9, 21, 27, 219},
		StaticText {
			disabled,
			"Model Start Time and Duration"
		},
		/* [26] */
		{218, 9, 241, 126},
		CheckBox {
			enabled,
			"Show Currents"
		},
		/* [27] */
		{309, 9, 332, 136},
		CheckBox {
			enabled,
			"Run Backwards"
		}
	}
};

resource 'DITL' (1010, "Model Settings Hindcast") {
	{	/* array DITLarray: 26 elements */
		/* [1] */
		{403, 270, 424, 347},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{399, 266, 428, 351},
		UserItem {
			disabled
		},
		/* [3] */
		{403, 169, 424, 246},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{403, 75, 424, 152},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{35, 26, 51, 143},
		StaticText {
			disabled,
			"Model Start Date:"
		},
		/* [6] */
		{33, 146, 52, 200},
		UserItem {
			disabled
		},
		/* [7] */
		{36, 290, 52, 319},
		EditText {
			enabled,
			""
		},
		/* [8] */
		{33, 328, 52, 366},
		UserItem {
			disabled
		},
		/* [9] */
		{73, 159, 89, 187},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{73, 205, 89, 233},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{68, 26, 102, 148},
		StaticText {
			disabled,
			"Model Start Time:\n        (24-hour)"
		},
		/* [12] */
		{73, 193, 89, 201},
		StaticText {
			disabled,
			":"
		},
		/* [13] */
		{108, 26, 128, 143},
		StaticText {
			disabled,
			"Model End Date:"
		},
		/* [14] */
		{106, 146, 125, 200},
		UserItem {
			disabled
		},
		/* [15] */
		{109, 290, 125, 319},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{106, 328, 125, 366},
		UserItem {
			disabled
		},
		/* [17] */
		{150, 159, 166, 187},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{150, 205, 166, 233},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{145, 26, 179, 148},
		StaticText {
			disabled,
			"Model End Time:\n        (24-hour)"
		},
		/* [20] */
		{150, 193, 166, 201},
		StaticText {
			disabled,
			":"
		},
		/* [21] */
		{195, 9, 234, 338},
		StaticText {
			disabled,
			"GNOME will automatically run the Best Es"
			"timate (Forecast) solution."
		},
		/* [22] */
		{240, 9, 262, 369},
		CheckBox {
			enabled,
			"Include the Minimum Regret (Uncertainty)"
			" solution.   "
		},
		/* [23] */
		{299, 9, 319, 164},
		StaticText {
			disabled,
			"Computation Time Step:"
		},
		/* [24] */
		{300, 170, 316, 230},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{300, 238, 318, 288},
		StaticText {
			disabled,
			"hours"
		},
		/* [26] */
		{326, 9, 349, 176},
		CheckBox {
			enabled,
			"Prevent Land Jumping"
		},
		/* [27] */
		{16, 10, 185, 438},
		UserItem {
			disabled
		},
		/* [28] */
		{9, 21, 27, 219},
		StaticText {
			disabled,
			"Model Start Time and End Time"
		},
		/* [29] */
		{269, 9, 292, 126},
		CheckBox {
			enabled,
			"Show Currents"
		},
		/* [30] */
		{360, 9, 383, 136},
		CheckBox {
			enabled,
			"Run Backwards"
		}
	}
};

resource 'DITL' (1300, "M13: LE Info Dialog") {
	{	/* array DITLarray: 89 elements */
		/* [1] */
		{349, 585, 369, 643},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{345, 581, 373, 647},
		UserItem {
			disabled
		},
		/* [3] */
		{349, 511, 369, 574},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{349, 436, 369, 499},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{43, 390, 59, 465},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{78, 144, 94, 219},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{77, 228, 95, 355},
		UserItem {
			enabled
		},
		/* [8] */
		{183, 333, 199, 408},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{182, 453, 200, 574},
		CheckBox {
			enabled,
			"Spill on bottom"
		},
		/* [10] */
		{41, 88, 59, 165},
		UserItem {
			enabled
		},
		/* [11] */
		{93, 479, 109, 524},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [12] */
		{92, 529, 108, 572},
		StaticText {
			disabled,
			"cm/s"
		},
		/* [13] */
		{140, 23, 156, 135},
		UserItem {
			enabled
		},
		/* [14] */
		{140, 164, 156, 187},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{140, 194, 156, 260},
		UserItem {
			enabled
		},
		/* [16] */
		{167, 101, 183, 129},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{167, 147, 183, 175},
		EditText {
			enabled,
			""
		},
		/* [18] */
		{137, 305, 153, 330},
		StaticText {
			disabled,
			"Lat:"
		},
		/* [19] */
		{138, 335, 154, 415},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{137, 333, 154, 363},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [21] */
		{138, 366, 154, 393},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{137, 404, 154, 434},
		StaticText {
			disabled,
			"Min:"
		},
		/* [23] */
		{138, 437, 154, 477},
		EditText {
			enabled,
			""
		},
		/* [24] */
		{138, 437, 154, 457},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{137, 468, 155, 496},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [26] */
		{138, 499, 154, 539},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{137, 556, 155, 633},
		UserItem {
			enabled
		},
		/* [28] */
		{162, 294, 179, 329},
		StaticText {
			disabled,
			"Long:"
		},
		/* [29] */
		{163, 335, 179, 415},
		EditText {
			enabled,
			""
		},
		/* [30] */
		{162, 333, 179, 363},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [31] */
		{163, 366, 179, 393},
		EditText {
			enabled,
			""
		},
		/* [32] */
		{162, 404, 179, 434},
		StaticText {
			disabled,
			"Min:"
		},
		/* [33] */
		{163, 437, 179, 477},
		EditText {
			enabled,
			""
		},
		/* [34] */
		{163, 437, 179, 457},
		EditText {
			enabled,
			""
		},
		/* [35] */
		{162, 468, 179, 496},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [36] */
		{163, 499, 179, 539},
		EditText {
			enabled,
			""
		},
		/* [37] */
		{162, 556, 180, 633},
		UserItem {
			enabled
		},
		/* [38] */
		{246, 23, 262, 135},
		UserItem {
			enabled
		},
		/* [39] */
		{247, 164, 263, 187},
		EditText {
			enabled,
			""
		},
		/* [40] */
		{246, 194, 262, 260},
		UserItem {
			enabled
		},
		/* [41] */
		{276, 101, 292, 129},
		EditText {
			enabled,
			""
		},
		/* [42] */
		{276, 147, 292, 175},
		EditText {
			enabled,
			""
		},
		/* [43] */
		{246, 312, 262, 338},
		StaticText {
			disabled,
			"Lat:"
		},
		/* [44] */
		{246, 343, 262, 423},
		EditText {
			enabled,
			""
		},
		/* [45] */
		{245, 341, 262, 371},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [46] */
		{246, 374, 262, 401},
		EditText {
			enabled,
			""
		},
		/* [47] */
		{245, 412, 262, 442},
		StaticText {
			disabled,
			"Min:"
		},
		/* [48] */
		{246, 445, 262, 485},
		EditText {
			enabled,
			""
		},
		/* [49] */
		{246, 445, 262, 465},
		EditText {
			enabled,
			""
		},
		/* [50] */
		{245, 476, 263, 504},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [51] */
		{246, 507, 262, 547},
		EditText {
			enabled,
			""
		},
		/* [52] */
		{245, 552, 263, 629},
		UserItem {
			enabled
		},
		/* [53] */
		{271, 303, 288, 340},
		StaticText {
			disabled,
			"Long:"
		},
		/* [54] */
		{271, 343, 287, 423},
		EditText {
			enabled,
			""
		},
		/* [55] */
		{270, 341, 287, 371},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [56] */
		{271, 374, 287, 401},
		EditText {
			enabled,
			""
		},
		/* [57] */
		{270, 412, 287, 442},
		StaticText {
			disabled,
			"Min:"
		},
		/* [58] */
		{271, 445, 287, 485},
		EditText {
			enabled,
			""
		},
		/* [59] */
		{271, 445, 287, 465},
		EditText {
			enabled,
			""
		},
		/* [60] */
		{270, 476, 287, 504},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [61] */
		{271, 507, 287, 547},
		EditText {
			enabled,
			""
		},
		/* [62] */
		{270, 552, 288, 629},
		UserItem {
			enabled
		},
		/* [63] */
		{324, 16, 342, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [64] */
		{343, 16, 361, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [65] */
		{362, 16, 380, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [66] */
		{118, 12, 201, 674},
		UserItem {
			disabled
		},
		/* [67] */
		{224, 11, 312, 296},
		UserItem {
			disabled
		},
		/* [68] */
		{167, 135, 183, 143},
		StaticText {
			disabled,
			":"
		},
		/* [69] */
		{276, 135, 292, 143},
		StaticText {
			disabled,
			":"
		},
		/* [70] */
		{42, 323, 59, 385},
		StaticText {
			disabled,
			"# Splots:"
		},
		/* [71] */
		{78, 20, 96, 139},
		StaticText {
			disabled,
			"Amount Released:"
		},
		/* [72] */
		{42, 19, 58, 87},
		StaticText {
			disabled,
			"Pollutant:"
		},
		/* [73] */
		{93, 388, 109, 474},
		StaticText {
			disabled,
			"rise velocity:"
		},
		/* [74] */
		{110, 26, 126, 122},
		StaticText {
			disabled,
			"Release Start:"
		},
		/* [75] */
		{207, 22, 224, 221},
		CheckBox {
			enabled,
			"Different end release time"
		},
		/* [76] */
		{182, 309, 198, 323},
		StaticText {
			disabled,
			"z:"
		},
		/* [77] */
		{164, 25, 196, 94},
		StaticText {
			disabled,
			"Start Time:\n (24-hour)"
		},
		/* [78] */
		{273, 29, 305, 94},
		StaticText {
			disabled,
			"End Time:\n(24-hour)"
		},
		/* [79] */
		{224, 299, 312, 674},
		UserItem {
			disabled
		},
		/* [80] */
		{207, 302, 224, 521},
		CheckBox {
			enabled,
			"Different end release position"
		},
		/* [81] */
		{69, 391, 85, 497},
		StaticText {
			disabled,
			"Age at Release:"
		},
		/* [82] */
		{69, 499, 85, 574},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [83] */
		{69, 579, 85, 622},
		StaticText {
			disabled,
			"hours"
		},
		/* [84] */
		{350, 205, 373, 375},
		UserItem {
			disabled
		},
		/* [85] */
		{40, 496, 60, 583},
		Button {
			enabled,
			"Windage"
		},
		/* [86] */
		{184, 414, 199, 445},
		StaticText {
			disabled,
			"m"
		},
		/* [87] */
		{321, 258, 341, 345},
		Button {
			enabled,
			"Parameters"
		},
		/* [88] */
		{13, 19, 29, 100},
		StaticText {
			disabled,
			"Spill Name:"
		},
		/* [89] */
		{13, 102, 29, 407},
		EditText {
			enabled,
			"Edit Text"
		}
	}
};

resource 'DITL' (1691, "GMT offsets") {
	{	/* array DITLarray: 6 elements */
		/* [1] */
		{423, 204, 443, 304},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{419, 200, 448, 310},
		UserItem {
			disabled
		},
		/* [3] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [4] */
		{15, 58, 180, 306},
		StaticText {
			disabled,
			"^0"
		},
		/* [5] */
		{181, 58, 346, 306},
		StaticText {
			disabled,
			"^1"
		},
		/* [6] */
		{351, 57, 397, 304},
		StaticText {
			disabled,
			"For international offsets search for \"Wo"
			"rld Time Zones GMT\" on the internet or c"
			"heck your computer clock."
		}
	}
};

resource 'DITL' (3300) {
	{	/* array DITLarray: 42 elements */
		/* [1] */
		{422, 323, 443, 396},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{418, 319, 447, 400},
		UserItem {
			disabled
		},
		/* [3] */
		{422, 227, 443, 300},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{422, 138, 443, 211},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{12, 15, 32, 98},
		StaticText {
			disabled,
			"File Name:"
		},
		/* [6] */
		{11, 98, 29, 398},
		StaticText {
			enabled,
			"name"
		},
		/* [7] */
		{46, 21, 64, 82},
		CheckBox {
			enabled,
			"Active"
		},
		/* [8] */
		{71, 21, 91, 200},
		CheckBox {
			enabled,
			"Show Velocities @ 1 in = "
		},
		/* [9] */
		{73, 207, 89, 282},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{73, 287, 89, 319},
		StaticText {
			disabled,
			"m/s"
		},
		/* [11] */
		{106, 22, 124, 168},
		StaticText {
			disabled,
			"Multiplicative Scalar:"
		},
		/* [12] */
		{104, 207, 120, 282},
		EditText {
			enabled,
			"2000000"
		},
		/* [13] */
		{227, 56, 246, 233},
		StaticText {
			disabled,
			"Along Current Uncertainty:"
		},
		/* [14] */
		{227, 241, 243, 316},
		EditText {
			enabled,
			"val"
		},
		/* [15] */
		{226, 322, 243, 340},
		StaticText {
			disabled,
			"%"
		},
		/* [16] */
		{256, 56, 273, 236},
		StaticText {
			disabled,
			"Cross Current Uncertainty:"
		},
		/* [17] */
		{256, 241, 272, 316},
		EditText {
			enabled,
			"val"
		},
		/* [18] */
		{256, 322, 273, 350},
		StaticText {
			disabled,
			"%"
		},
		/* [19] */
		{283, 56, 301, 180},
		StaticText {
			disabled,
			"Minimum Current:"
		},
		/* [20] */
		{283, 241, 299, 316},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{282, 322, 302, 376},
		StaticText {
			disabled,
			"m/sec"
		},
		/* [22] */
		{284, 57, 300, 132},
		StaticText {
			disabled,
			"Start Time:"
		},
		/* [23] */
		{281, 241, 297, 316},
		EditText {
			enabled,
			"0"
		},
		/* [24] */
		{282, 322, 298, 365},
		StaticText {
			disabled,
			"hours"
		},
		/* [25] */
		{310, 55, 326, 219},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [26] */
		{310, 241, 326, 316},
		EditText {
			enabled,
			"3"
		},
		/* [27] */
		{310, 322, 327, 368},
		StaticText {
			disabled,
			"hours"
		},
		/* [28] */
		{215, 23, 342, 390},
		UserItem {
			disabled
		},
		/* [29] */
		{205, 42, 222, 124},
		StaticText {
			disabled,
			"Uncertainty"
		},
		/* [30] */
		{168, 25, 190, 109},
		UserItem {
			enabled
		},
		/* [31] */
		{170, 168, 188, 294},
		StaticText {
			disabled,
			"Time offset (hours)"
		},
		/* [32] */
		{171, 300, 187, 355},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{169, 365, 189, 463},
		Button {
			enabled,
			"GMT Offsets"
		},
		/* [34] */
		{74, 331, 90, 354},
		StaticText {
			disabled,
			"at"
		},
		/* [35] */
		{73, 364, 90, 409},
		EditText {
			enabled,
			""
		},
		/* [36] */
		{74, 418, 90, 441},
		StaticText {
			disabled,
			"m"
		},
		/* [37] */
		{140, 25, 156, 280},
		StaticText {
			disabled,
			"The time coordinate in this file is in :"
		},
		/* [38] */
		{356, 21, 376, 402},
		CheckBox {
			enabled,
			"Extrapolate Currents Using First and Last Model V"
			"alues"
		},
		/* [39] */
		{400, 21, 420, 362},
		CheckBox {
			enabled,
			"Extrapolate Currents Vertically Using Su"
			"rface Data"
		},
		/* [40] */
		{431, 180, 449, 281},
		StaticText {
			disabled,
			"Extrapolate To:"
		},
		/* [41] */
		{432, 290, 448, 345},
		EditText {
			enabled,
			""
		},
		/* [42] */
		{433, 354, 449, 377},
		StaticText {
			disabled,
			"m"
		},
		/* [43] */
		{386, 21, 397, 134},
		Button {
			enabled,
			"Replace Mover"
		},
		/* [44] */
		{106, 354, 124, 424},
		CheckBox {
			enabled,
			"Bottom"
		},
	}
};

resource 'DITL' (2150, "Load / Create Map") {
	{	/* array DITLarray: 5 elements */
		/* [1] */
		{58, 171, 78, 229},
		Button {
			enabled,
			"Create"
		},
		/* [2] */
		{54, 168, 82, 232},
		UserItem {
			disabled
		},
		/* [3] */
		{58, 239, 78, 297},
		Button {
			enabled,
			"Load"
		},
		/* [4] */
		{58, 305, 78, 363},
		Button {
			enabled,
			"Cancel"
		},
		/* [5] */
		{20, 16, 39, 426},
		StaticText {
			disabled,
			"Load a map file or create water map boun"
			"daries:"
		}
	}
};

resource 'DITL' (5200, "Output Options") {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{111, 237, 133, 317},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{107, 233, 137, 321},
		UserItem {
			disabled
		},
		/* [3] */
		{111, 137, 133, 217},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{111, 44, 133, 124},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{29, 12, 44, 126},
		StaticText {
			disabled,
			"Output Interval:"
		},
		/* [6] */
		{29, 133, 45, 191},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{28, 200, 48, 257},
		StaticText {
			disabled,
			"hours"
		},
		/* [8] */
		{64, 12, 79, 166},
		StaticText {
			disabled,
			"Snapshot series offset:"
		},
		/* [9] */
		{64, 175, 80, 233},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{63, 242, 83, 299},
		StaticText {
			disabled,
			"hours"
		}
	}
};

resource 'DITL' (3985) {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{223, 237, 243, 295},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{219, 233, 247, 299},
		UserItem {
			disabled
		},
		/* [3] */
		{223, 163, 243, 221},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{223, 93, 243, 151},
		Button {
			disabled,
			"Help"
		},
		/* [5] */
		{12, 19, 32, 200},
		Button {
			enabled,
			"Diffusivity and Time Step"
		},
		/* [6] */
		{114, 88, 134, 268},
		Button {
			enabled,
			"Custom Grid and Profiles"
		},
		/* [7] */
		{147, 63, 167, 293},
		Button {
			enabled,
			"Temperature and Salinity Profiles"
		},
		/* [8] */
		{180, 118, 200, 238},
		Button {
			enabled,
			"Hydrodynamics"
		},
		/* [9] */
		{71, 19, 97, 345},
		UserItem {
			enabled
		},
		/* [10] */
		{47, 19, 65, 208},
		StaticText {
			disabled,
			"Circulation Fields : U, V, T, S"
		}
	}
};

resource 'DITL' (3975) {
	{	/* array DITLarray: 62 elements */
		/* [1] */
		{431, 621, 456, 694},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{427, 617, 460, 698},
		UserItem {
			disabled
		},
		/* [3] */
		{431, 537, 456, 611},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{431, 453, 456, 527},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{185, 549, 210, 665},
		Button {
			enabled,
			"Delete Selected"
		},
		/* [6] */
		{185, 675, 210, 759},
		Button {
			enabled,
			"Delete All"
		},
		/* [7] */
		{185, 406, 210, 537},
		Button {
			enabled,
			"Replace Selected"
		},
		/* [8] */
		{61, 10, 77, 112},
		StaticText {
			enabled,
			"Discharge Time:"
		},
		/* [9] */
		{62, 118, 78, 148},
		EditText {
			disabled,
			""
		},
		/* [10] */
		{62, 155, 78, 300},
		StaticText {
			enabled,
			"hours since blowout"
		},
		/* [11] */
		{88, 213, 104, 265},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{114, 206, 129, 254},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{139, 174, 155, 212},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{165, 152, 181, 194},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{14, 591, 30, 625},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{41, 423, 56, 456},
		StaticText {
			disabled,
			"Time"
		},
		/* [17] */
		{41, 479, 56, 515},
		StaticText {
			disabled,
			"Q oil"
		},
		/* [18] */
		{41, 579, 56, 614},
		StaticText {
			disabled,
			"Temp"
		},
		/* [19] */
		{41, 688, 56, 720},
		StaticText {
			disabled,
			"Dens"
		},
		/* [20] */
		{59, 418, 174, 743},
		UserItem {
			enabled
		},
		/* [21] */
		{15, 419, 30, 583},
		StaticText {
			disabled,
			"Auto-increment time by:"
		},
		/* [22] */
		{87, 10, 104, 195},
		UserItem {
			enabled
		},
		/* [23] */
		{139, 10, 155, 168},
		StaticText {
			disabled,
			"Discharge Temperature:"
		},
		/* [24] */
		{165, 10, 184, 145},
		StaticText {
			disabled,
			"Diameter of orifice:"
		},
		/* [25] */
		{15, 636, 32, 679},
		StaticText {
			disabled,
			"hours"
		},
		/* [26] */
		{113, 10, 130, 227},
		StaticText {
			disabled,
			"Gas to Oil Ratio (by Volume):"
		},
		/* [27] */
		{138, 220, 156, 293},
		UserItem {
			disabled
		},
		/* [28] */
		{164, 202, 182, 262},
		UserItem {
			disabled
		},
		/* [29] */
		{41, 530, 56, 559},
		StaticText {
			disabled,
			"GOR"
		},
		/* [30] */
		{41, 635, 56, 668},
		StaticText {
			disabled,
			"Diam"
		},
		/* [31] */
		{194, 10, 211, 85},
		StaticText {
			disabled,
			"Oil Density:"
		},
		/* [32] */
		{193, 92, 209, 130},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{192, 137, 210, 197},
		UserItem {
			disabled
		},
		/* [34] */
		{87, 273, 104, 336},
		UserItem {
			disabled
		},
		/* [35] */
		{12, 12, 38, 203},
		UserItem {
			disabled
		},
		/* [36] */
		{286, 8, 301, 82},
		StaticText {
			disabled,
			"Gas Type:"
		},
		/* [37] */
		{286, 83, 302, 175},
		UserItem {
			enabled
		},
		/* [38] */
		{342, 231, 362, 368},
		StaticText {
			disabled,
			"kg/m**3    (900-940)"
		},
		/* [39] */
		{230, 10, 245, 144},
		StaticText {
			disabled,
			"Initial bubble radius: "
		},
		/* [40] */
		{230, 151, 246, 209},
		EditText {
			enabled,
			""
		},
		/* [41] */
		{229, 218, 249, 350},
		StaticText {
			disabled,
			"mm     (max 10mm)"
		},
		/* [42] */
		{314, 33, 329, 197},
		StaticText {
			disabled,
			"Molecular weight of gas:"
		},
		/* [43] */
		{314, 200, 330, 258},
		EditText {
			enabled,
			""
		},
		/* [44] */
		{403, 8, 418, 202},
		StaticText {
			disabled,
			"Separation of gas from plume:"
		},
		/* [45] */
		{403, 207, 419, 291},
		UserItem {
			enabled
		},
		/* [46] */
		{343, 33, 358, 157},
		StaticText {
			disabled,
			"Density of Hydrate:"
		},
		/* [47] */
		{343, 164, 359, 222},
		EditText {
			enabled,
			""
		},
		/* [48] */
		{375, 8, 390, 122},
		StaticText {
			disabled,
			"Hydrate process:"
		},
		/* [49] */
		{375, 129, 391, 217},
		UserItem {
			enabled
		},
		/* [50] */
		{258, 10, 273, 84},
		StaticText {
			disabled,
			"Drop size: "
		},
		/* [51] */
		{258, 91, 274, 249},
		UserItem {
			enabled
		},
		/* [52] */
		{311, 269, 332, 326},
		UserItem {
			disabled
		},
		/* [53] */
		{433, 249, 454, 313},
		UserItem {
			disabled
		},
		/* [54] */
		{436, 8, 451, 242},
		StaticText {
			disabled,
			"Does Gas Contain Hydrogen Sulfide?"
		},
		/* [55] */
		{61, 11, 76, 75},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [56] */
		{61, 82, 77, 120},
		EditText {
			enabled,
			""
		},
		/* [57] */
		{60, 131, 80, 170},
		StaticText {
			disabled,
			"hours"
		},
		/* [58] */
		{59, 202, 77, 349},
		CheckBox {
			enabled,
			"Continuous release"
		},
		/* [59] */
		{113, 260, 131, 372},
		UserItem {
			disabled
		},
		/* [60] */
		{87, 273, 104, 336},
		UserItem {
			disabled
		},
		/* [61] */
		{47, 9, 209, 380},
		UserItem {
			disabled
		},
		/* [62] */
		{2, 5, 221, 777},
		UserItem {
			enabled
		}
	}
};

resource 'DITL' (1692) {
	{	/* array DITLarray: 6 elements */
		/* [1] */
		{282, 241, 302, 341},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{278, 237, 307, 347},
		UserItem {
			disabled
		},
		/* [3] */
		{7, 6, 39, 38},
		Icon {
			disabled,
			1
		},
		/* [4] */
		{29, 56, 79, 355},
		StaticText {
			disabled,
			"In order to write out the GIS compatible"
			" files the Minimum Regret solution must "
			"be turned on during the GNOME trajectory"
			" simulation.  "
		},
		/* [5] */
		{86, 56, 169, 355},
		StaticText {
			disabled,
			"NOAA requires this because the Uncertain"
			"ty information is an important part of t"
			"he information responders need to make t"
			"he best choices for allocation of limite"
			"d resources during a spill.  "
		},
		/* [6] */
		{187, 56, 237, 355},
		StaticText {
			disabled,
			"Please go back and select the Minimum Re"
			"gret checkbox under \"Model Settings\" and"
			" then run your simulation again."
		}
	}
};

resource 'DITL' (1825, "NetCDF WindMover") {
	{	/* array DITLarray: 29 elements */
		/* [1] */
		{397, 358, 417, 424},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{393, 354, 421, 428},
		UserItem {
			disabled
		},
		/* [3] */
		{397, 273, 417, 339},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{397, 191, 417, 257},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{55, 25, 73, 87},
		CheckBox {
			enabled,
			"Active"
		},
		/* [6] */
		{228, 153, 248, 214},
		EditText {
			enabled,
			"Uncertain"
		},
		/* [7] */
		{255, 153, 271, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{278, 153, 295, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{302, 153, 317, 214},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{301, 221, 317, 302},
		UserItem {
			enabled
		},
		/* [11] */
		{11, 102, 29, 402},
		StaticText {
			enabled,
			"name"
		},
		/* [12] */
		{255, 58, 271, 122},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [13] */
		{256, 221, 272, 262},
		StaticText {
			disabled,
			"hours"
		},
		/* [14] */
		{214, 20, 326, 318},
		UserItem {
			disabled
		},
		/* [15] */
		{278, 58, 295, 143},
		StaticText {
			disabled,
			"Speed Scale:"
		},
		/* [16] */
		{302, 28, 319, 143},
		StaticText {
			disabled,
			"Total Angle Scale:"
		},
		/* [17] */
		{229, 58, 245, 133},
		StaticText {
			disabled,
			"Start Time:"
		},
		/* [18] */
		{230, 222, 246, 261},
		StaticText {
			disabled,
			"hours"
		},
		/* [19] */
		{205, 24, 221, 99},
		StaticText {
			disabled,
			"Uncertainty"
		},
		/* [20] */
		{12, 19, 32, 102},
		StaticText {
			disabled,
			"File Name:"
		},
		/* [21] */
		{347, 21, 367, 397},
		CheckBox {
			enabled,
			"Extrapolate Winds Using First and Last Model Values"
		},
		/* [22] */
		{158, 25, 180, 109},
		UserItem {
			enabled
		},
		/* [23] */
		{160, 173, 178, 299},
		StaticText {
			disabled,
			"Time offset (hours)"
		},
		/* [24] */
		{161, 305, 177, 360},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{159, 370, 179, 468},
		Button {
			enabled,
			"GMT Offsets"
		},
		/* [26] */
		{130, 25, 146, 280},
		StaticText {
			disabled,
			"The time coordinate in this file is in :"
		},
		/* [27] */
		{87, 25, 107, 204},
		CheckBox {
			enabled,
			"Show Velocities @ 1 in = "
		},
		/* [28] */
		{89, 211, 105, 286},
		EditText {
			enabled,
			""
		},
		/* [29] */
		{89, 291, 105, 323},
		StaticText {
			disabled,
			"m/s"
		}
	}
};

resource 'DITL' (1610) {
	{	/* array DITLarray: 38 elements */
		/* [1] */
		{386, 427, 411, 500},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{382, 423, 415, 504},
		UserItem {
			disabled
		},
		/* [3] */
		{386, 331, 411, 405},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{386, 241, 411, 315},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{196, 161, 221, 281},
		Button {
			enabled,
			"Delete Selected"
		},
		/* [6] */
		{386, 9, 411, 130},
		Button {
			enabled,
			"Delete All"
		},
		/* [7] */
		{196, 15, 221, 140},
		Button {
			enabled,
			"Replace Selected"
		},
		/* [8] */
		{11, 15, 27, 52},
		UserItem {
			enabled
		},
		/* [9] */
		{12, 151, 28, 179},
		EditText {
			disabled,
			""
		},
		/* [10] */
		{12, 198, 28, 238},
		UserItem {
			enabled
		},
		/* [11] */
		{42, 122, 58, 150},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{42, 181, 58, 209},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{75, 74, 91, 112},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{111, 176, 127, 230},
		EditText {
			enabled,
			""
		},
		/* [15] */
		{166, 186, 182, 220},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{240, 15, 255, 94},
		StaticText {
			disabled,
			"Date(m/d/y)"
		},
		/* [17] */
		{240, 115, 255, 149},
		StaticText {
			disabled,
			"Time"
		},
		/* [18] */
		{240, 162, 255, 207},
		StaticText {
			disabled,
			"Speed"
		},
		/* [19] */
		{240, 219, 255, 290},
		StaticText {
			disabled,
			"Current To"
		},
		/* [20] */
		{258, 9, 373, 294},
		UserItem {
			enabled
		},
		/* [21] */
		{6, 307, 228, 529},
		UserItem {
			enabled
		},
		/* [22] */
		{450, 534, 660, 744},
		Picture {
			disabled,
			128
		},
		/* [23] */
		{167, 15, 182, 181},
		StaticText {
			disabled,
			"Auto-increment time by:"
		},
		/* [24] */
		{510, 323, 527, 328},
		StaticText {
			disabled,
			""
		},
		/* [25] */
		{43, 15, 60, 112},
		StaticText {
			disabled,
			"Time (24 hour) :"
		},
		/* [26] */
		{75, 15, 91, 67},
		StaticText {
			disabled,
			"Speed:"
		},
		/* [27] */
		{111, 15, 130, 170},
		StaticText {
			disabled,
			"Current Direction is to:"
		},
		/* [28] */
		{167, 232, 184, 279},
		StaticText {
			disabled,
			"hours"
		},
		/* [29] */
		{509, 501, 525, 512},
		StaticText {
			disabled,
			"/"
		},
		/* [30] */
		{491, 476, 508, 487},
		StaticText {
			disabled,
			"/"
		},
		/* [31] */
		{42, 162, 62, 172},
		StaticText {
			disabled,
			":"
		},
		/* [32] */
		{75, 123, 93, 233},
		UserItem {
			disabled
		},
		/* [33] */
		{493, 380, 511, 459},
		StaticText {
			disabled,
			""
		},
		/* [34] */
		{512, 354, 533, 474},
		Button {
			enabled,
			"Load Wind Data..."
		},
		/* [35] */
		{305, 335, 330, 499},
		Button {
			enabled,
			"Current Settings..."
		},
		/* [36] */
		{258, 307, 373, 529},
		UserItem {
			disabled
		},
		/* [37] */
		{6, 9, 228, 294},
		UserItem {
			disabled
		},
		/* [38] */
		{133, 15, 153, 280},
		StaticText {
			disabled,
			"Enter  degrees true or text (e.g. \"NNW\")"
		}
	}
};

resource 'DITL' (2050, "Averaged Winds Parameters") {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{147, 204, 167, 270},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{143, 200, 171, 274},
		UserItem {
			disabled
		},
		/* [3] */
		{147, 125, 167, 191},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{147, 37, 167, 103},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{24, 10, 40, 153},
		StaticText {
			disabled,
			"Time to Average Over:"
		},
		/* [6] */
		{24, 170, 40, 231},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{25, 238, 41, 279},
		StaticText {
			disabled,
			"hours"
		},
		/* [8] */
		{47, 66, 64, 153},
		StaticText {
			disabled,
			"Scale Factor:"
		},
		/* [9] */
		{47, 170, 64, 231},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{71, 28, 88, 153},
		StaticText {
			disabled,
			"Wind Power Factor:"
		},
		/* [11] */
		{71, 170, 86, 231},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [12] */
		{105, 66, 123, 300},
		CheckBox {
			enabled,
			"Use scale factor from main dialog"
		}
	}
};

resource 'DITL' (3250) {
	{	/* array DITLarray: 9 elements */
		/* [1] */
		{158, 188, 178, 254},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{154, 184, 182, 258},
		UserItem {
			disabled
		},
		/* [3] */
		{158, 104, 178, 170},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{158, 21, 178, 87},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{83, 24, 101, 230},
		RadioButton {
			enabled,
			"Standing Wave"
		},
		/* [6] */
		{114, 24, 132, 230},
		RadioButton {
			enabled,
			"Progressive Wave"
		},
		/* [7] */
		{58, 21, 74, 96},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [8] */
		{60, 127, 76, 202},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{23, 22, 54, 348},
		StaticText {
			disabled,
			"Would you like the shio heights file to "
			"be treated as a standing wave or a progr"
			"essive wave?"
		}
	}
};

resource 'DITL' (3400) {
	{	/* array DITLarray: 33 elements */
		/* [1] */
		{341, 691, 362, 764},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{337, 687, 366, 768},
		UserItem {
			disabled
		},
		/* [3] */
		{341, 586, 362, 659},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{341, 488, 362, 561},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{186, 16, 206, 125},
		StaticText {
			disabled,
			"CUR File Path:"
		},
		/* [6] */
		{185, 129, 203, 879},
		StaticText {
			enabled,
			""
		},
		/* [7] */
		{61, 16, 79, 160},
		StaticText {
			disabled,
			"Scale Velocity:"
		},
		/* [8] */
		{63, 196, 79, 233},
		StaticText {
			enabled,
			""
		},
		/* [9] */
		{62, 242, 79, 280},
		StaticText {
			disabled,
			"cm/s"
		},
		/* [10] */
		{93, 16, 110, 193},
		StaticText {
			disabled,
			"Boundary Layer Thickness:"
		},
		/* [11] */
		{94, 196, 110, 241},
		StaticText {
			enabled,
			""
		},
		/* [12] */
		{93, 242, 110, 270},
		StaticText {
			disabled,
			"cm"
		},
		/* [13] */
		{122, 16, 139, 160},
		StaticText {
			disabled,
			"Upper Eddy Viscosity:"
		},
		/* [14] */
		{123, 196, 139, 241},
		StaticText {
			enabled,
			""
		},
		/* [15] */
		{122, 242, 139, 300},
		StaticText {
			disabled,
			"cm2/s"
		},
		/* [16] */
		{151, 16, 169, 160},
		StaticText {
			disabled,
			"Lower Eddy Viscosity:"
		},
		/* [17] */
		{153, 196, 169, 241},
		StaticText {
			enabled,
			""
		},
		/* [18] */
		{151, 242, 169, 296},
		StaticText {
			disabled,
			"cm2/s"
		},
		/* [19] */
		{274, 16, 290, 160},
		StaticText {
			disabled,
			"Upper Level Density:"
		},
		/* [20] */
		{273, 196, 289, 241},
		StaticText {
			enabled,
			""
		},
		/* [21] */
		{272, 242, 289, 298},
		StaticText {
			disabled,
			"gm/cm3"
		},
		/* [22] */
		{303, 16, 319, 160},
		StaticText {
			disabled,
			"Lower Level Density:"
		},
		/* [23] */
		{302, 196, 318, 241},
		StaticText {
			enabled,
			""
		},
		/* [24] */
		{301, 242, 318, 298},
		StaticText {
			disabled,
			"gm/cm3"
		},
		/* [25] */
		{215, 16, 235, 125},
		StaticText {
			disabled,
			"SSH File Path:"
		},
		/* [26] */
		{214, 129, 232, 879},
		StaticText {
			enabled,
			""
		},
		/* [27] */
		{244, 16, 264, 125},
		StaticText {
			disabled,
			"PYC File Path:"
		},
		/* [28] */
		{243, 129, 261, 879},
		StaticText {
			enabled,
			""
		},
		/* [29] */
		{273, 16, 293, 125},
		StaticText {
			disabled,
			"LLD File Path:"
		},
		/* [30] */
		{272, 129, 290, 879},
		StaticText {
			enabled,
			""
		},
		/* [31] */
		{301, 16, 321, 125},
		StaticText {
			disabled,
			"ULD File Path:"
		},
		/* [32] */
		{300, 129, 318, 879},
		StaticText {
			enabled,
			""
		},
		/* [33] */
		{20, 16, 38, 366},
		StaticText {
			enabled,
			""
		}
	}
};

resource 'DITL' (5225) {
	{	/* array DITLarray: 53 elements */
		/* [1] */
		{359, 307, 379, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{355, 303, 383, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{359, 234, 379, 292},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{359, 164, 379, 222},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{20, 18, 55, 426},
		StaticText {
			disabled,
			"To track concentration at a location ent"
			"er the lat/lon below,\nor use the arrow o"
			"r lasso tool to select a region with the"
			" grid on\n"
		},
		/* [6] */
		{137, 17, 154, 236},
		CheckBox {
			enabled,
			"Select Lat/Lon of Point to Track"
		},
		/* [7] */
		{167, 47, 183, 112},
		StaticText {
			disabled,
			"Lat (Top):\n"
		},
		/* [8] */
		{168, 116, 184, 196},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{167, 114, 184, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [10] */
		{168, 147, 184, 174},
		EditText {
			enabled,
			""
		},
		/* [11] */
		{167, 185, 184, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [12] */
		{168, 218, 184, 258},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{168, 218, 184, 238},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{167, 249, 185, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [15] */
		{168, 280, 184, 320},
		EditText {
			enabled,
			""
		},
		/* [16] */
		{167, 336, 185, 413},
		UserItem {
			enabled
		},
		/* [17] */
		{192, 35, 209, 110},
		StaticText {
			disabled,
			"Long (Left):"
		},
		/* [18] */
		{193, 116, 209, 196},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{192, 114, 209, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [20] */
		{193, 147, 209, 174},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{192, 185, 209, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [22] */
		{193, 218, 209, 258},
		EditText {
			enabled,
			""
		},
		/* [23] */
		{193, 218, 209, 238},
		EditText {
			enabled,
			""
		},
		/* [24] */
		{192, 249, 209, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [25] */
		{193, 280, 209, 320},
		EditText {
			enabled,
			""
		},
		/* [26] */
		{192, 336, 210, 413},
		UserItem {
			enabled
		},
		/* [27] */
		{228, 23, 244, 109},
		StaticText {
			disabled,
			"Lat (Bottom):"
		},
		/* [28] */
		{228, 115, 244, 195},
		EditText {
			enabled,
			""
		},
		/* [29] */
		{227, 113, 244, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [30] */
		{228, 146, 244, 173},
		EditText {
			enabled,
			""
		},
		/* [31] */
		{227, 184, 244, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [32] */
		{228, 217, 244, 257},
		EditText {
			enabled,
			""
		},
		/* [33] */
		{228, 217, 244, 237},
		EditText {
			enabled,
			""
		},
		/* [34] */
		{227, 248, 245, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [35] */
		{228, 279, 244, 319},
		EditText {
			enabled,
			""
		},
		/* [36] */
		{227, 336, 245, 413},
		UserItem {
			enabled
		},
		/* [37] */
		{253, 26, 270, 113},
		StaticText {
			disabled,
			"Long (Right):"
		},
		/* [38] */
		{253, 115, 269, 195},
		EditText {
			enabled,
			""
		},
		/* [39] */
		{252, 113, 269, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [40] */
		{253, 146, 269, 173},
		EditText {
			enabled,
			""
		},
		/* [41] */
		{252, 184, 269, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [42] */
		{253, 217, 269, 257},
		EditText {
			enabled,
			""
		},
		/* [43] */
		{253, 217, 269, 237},
		EditText {
			enabled,
			""
		},
		/* [44] */
		{252, 248, 269, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [45] */
		{253, 279, 269, 319},
		EditText {
			enabled,
			""
		},
		/* [46] */
		{252, 336, 270, 413},
		UserItem {
			enabled
		},
		/* [47] */
		{290, 16, 308, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [48] */
		{309, 16, 327, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [49] */
		{328, 16, 346, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [50] */
		{155, 17, 285, 433},
		UserItem {
			disabled
		},
		/* [51] */
		{57, 18, 92, 426},
		StaticText {
			disabled,
			"If no points are selected the concentrat"
			"ion following the plume is tracked."
		},
		/* [52] */
		{94, 18, 129, 426},
		StaticText {
			disabled,
			"To see the output select Show Concentrat"
			"ion Plots from the left hand list after "
			"running the model"
		},
		/* [53] */
		{136, 314, 153, 433},
		CheckBox {
			enabled,
			"Track region"
		}
	}
};

resource 'DITL' (5230) {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{168, 307, 188, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{164, 303, 192, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{168, 233, 188, 296},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{168, 159, 188, 222},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{20, 18, 71, 440},
		StaticText {
			disabled,
			"To track concentration at a location, us"
			"e the arrow or lasso tool to select a re"
			"gion with the triangle grid showing, the"
			"n run the model. When you hit OK to exit"
			" this dialog the grid will be turned on."
		},
		/* [6] */
		{74, 18, 109, 440},
		StaticText {
			disabled,
			"If no points are selected the concentrat"
			"ion following the plume is tracked when "
			"the model is run. "
		},
		/* [7] */
		{111, 18, 146, 440},
		StaticText {
			disabled,
			"To see the output after running the mode"
			"l, select Show Concentration Plots from "
			"the left hand list."
		}
	}
};

resource 'DITL' (5250) {
	{	/* array DITLarray: 15 elements */
		/* [1] */
		{318, 14, 338, 75},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{314, 10, 342, 79},
		UserItem {
			disabled
		},
		/* [3] */
		{318, 96, 338, 157},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{318, 175, 338, 233},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{5, 4, 305, 364},
		UserItem {
			enabled
		},
		/* [6] */
		{14, 365, 30, 540},
		RadioButton {
			enabled,
			"Average Concentration"
		},
		/* [7] */
		{36, 365, 52, 540},
		RadioButton {
			enabled,
			"Maximum Concentration"
		},
		/* [8] */
		{115, 367, 131, 487},
		CheckBox {
			enabled,
			"Show Plot Grid"
		},
		/* [9] */
		{226, 400, 246, 490},
		Button {
			enabled,
			"Data Table"
		},
		/* [10] */
		{255, 400, 275, 490},
		Button {
			enabled,
			"Print Plot..."
		},
		/* [11] */
		{283, 400, 303, 490},
		Button {
			enabled,
			"Save Plot..."
		},
		/* [12] */
		{78, 365, 94, 536},
		RadioButton {
			enabled,
			"Depth Profile at Max Tri"
		},
		/* [13] */
		{147, 390, 167, 542},
		Button {
			enabled,
			"Toxicity Information"
		},
		/* [14] */
		{175, 390, 195, 542},
		Button {
			enabled,
			"Set Axis Endpoints"
		},
		/* [15] */
		{57, 365, 73, 540},
		RadioButton {
			enabled,
			"Average and Max Conc"
		}
	}
};

resource 'DITL' (5275) {
	{	/* array DITLarray: 24 elements */
		/* [1] */
		{259, 212, 281, 292},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{255, 208, 285, 296},
		UserItem {
			disabled
		},
		/* [3] */
		{259, 112, 281, 192},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{259, 19, 281, 99},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{68, 20, 86, 125},
		CheckBox {
			enabled,
			" Adult Fish"
		},
		/* [6] */
		{94, 20, 112, 175},
		CheckBox {
			enabled,
			"Adult Crustaceans"
		},
		/* [7] */
		{121, 20, 139, 185},
		CheckBox {
			enabled,
			"Sensitive Life Stages"
		},
		/* [8] */
		{15, 20, 33, 378},
		StaticText {
			disabled,
			"OverLay Toxicity Thresholds for each sel"
			"ected organism"
		},
		/* [9] */
		{38, 197, 59, 282},
		UserItem {
			enabled
		},
		/* [10] */
		{40, 20, 56, 175},
		UserItem {
			enabled
		},
		/* [11] */
		{68, 201, 86, 331},
		CheckBox {
			enabled,
			" Adult Corals"
		},
		/* [12] */
		{94, 200, 112, 355},
		CheckBox {
			enabled,
			"Stressed Corals"
		},
		/* [13] */
		{121, 200, 139, 365},
		CheckBox {
			enabled,
			"Coral Eggs and Larvae"
		},
		/* [14] */
		{148, 20, 166, 125},
		CheckBox {
			enabled,
			"Sea Grass"
		},
		/* [15] */
		{68, 20, 86, 125},
		CheckBox {
			enabled,
			"Low Concern"
		},
		/* [16] */
		{94, 20, 112, 150},
		CheckBox {
			enabled,
			"Medium Concern"
		},
		/* [17] */
		{121, 20, 139, 150},
		CheckBox {
			enabled,
			"High Concern"
		},
		/* [18] */
		{38, 198, 59, 283},
		UserItem {
			enabled
		},
		/* [19] */
		{212, 20, 228, 130},
		StaticText {
			disabled,
			"Overlay end time"
		},
		/* [20] */
		{212, 143, 228, 200},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [21] */
		{213, 208, 229, 360},
		StaticText {
			disabled,
			"hours after dispersion"
		},
		/* [22] */
		{181, 20, 197, 138},
		StaticText {
			disabled,
			"Overlay start time"
		},
		/* [23] */
		{181, 143, 197, 200},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [24] */
		{182, 208, 198, 360},
		StaticText {
			disabled,
			"hours after dispersion"
		}
	}
};

resource 'DITL' (5280) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{114, 212, 136, 292},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{110, 208, 140, 296},
		UserItem {
			disabled
		},
		/* [3] */
		{114, 112, 136, 192},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{114, 19, 136, 99},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{67, 20, 83, 130},
		StaticText {
			disabled,
			"Min y (depth)"
		},
		/* [6] */
		{67, 143, 83, 200},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{68, 208, 84, 260},
		StaticText {
			disabled,
			"meters"
		},
		/* [8] */
		{36, 20, 52, 128},
		StaticText {
			disabled,
			"Max x (conc)"
		},
		/* [9] */
		{36, 143, 52, 200},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{37, 208, 53, 260},
		StaticText {
			disabled,
			"ppm"
		},
		/* [11] */
		{3, 8, 21, 186},
		CheckBox {
			enabled,
			"Set max/min by hand"
		},
		/* [12] */
		{19, 8, 101, 293},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (5290) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{117, 227, 139, 307},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{113, 223, 143, 311},
		UserItem {
			disabled
		},
		/* [3] */
		{117, 127, 139, 207},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{117, 34, 139, 114},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{70, 35, 86, 145},
		StaticText {
			disabled,
			"Max y (conc)"
		},
		/* [6] */
		{70, 158, 86, 215},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [7] */
		{71, 223, 87, 275},
		StaticText {
			disabled,
			"ppm"
		},
		/* [8] */
		{39, 35, 55, 143},
		StaticText {
			disabled,
			"Max x (time)"
		},
		/* [9] */
		{39, 158, 55, 215},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [10] */
		{40, 223, 56, 275},
		StaticText {
			disabled,
			"hrs"
		},
		/* [11] */
		{6, 23, 24, 201},
		CheckBox {
			enabled,
			"Set max/min by hand"
		},
		/* [12] */
		{22, 23, 104, 308},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (5380) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{318, 14, 338, 75},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{314, 10, 342, 79},
		UserItem {
			disabled
		},
		/* [3] */
		{318, 96, 338, 157},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{318, 175, 338, 233},
		Button {
			enabled,
			"Help..."
		},
		/* [5] */
		{5, 4, 305, 364},
		UserItem {
			enabled
		},
		/* [6] */
		{14, 365, 30, 540},
		RadioButton {
			enabled,
			"Gallons on Shoreline"
		},
		/* [7] */
		{36, 365, 52, 540},
		RadioButton {
			enabled,
			"Miles of Shoreline"
		},
		/* [8] */
		{115, 367, 131, 487},
		CheckBox {
			enabled,
			"Show Plot Grid"
		},
		/* [9] */
		{226, 400, 246, 490},
		Button {
			enabled,
			"Data Table"
		},
		/* [10] */
		{255, 400, 275, 490},
		Button {
			enabled,
			"Print Plot..."
		},
		/* [11] */
		{283, 400, 303, 490},
		Button {
			enabled,
			"Save Plot..."
		},
		/* [12] */
		{175, 400, 195, 542},
		Button {
			enabled,
			"Set Axis Endpoints"
		}
	}
};

resource 'DITL' (5375) {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{107, 226, 129, 306},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{103, 222, 133, 310},
		UserItem {
			disabled
		},
		/* [3] */
		{107, 126, 129, 206},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{107, 33, 129, 113},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{30, 69, 45, 137},
		StaticText {
			disabled,
			"Density:"
		},
		/* [6] */
		{30, 148, 46, 218},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{28, 227, 48, 267},
		StaticText {
			enabled,
			"g/cc"
		},
		/* [8] */
		{61, 69, 76, 137},
		StaticText {
			disabled,
			"Half life:"
		},
		/* [9] */
		{61, 148, 77, 218},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{59, 227, 79, 267},
		StaticText {
			enabled,
			"hours"
		}
	}
};

resource 'DITL' (5350) {
	{	/* array DITLarray: 11 elements */
		/* [1] */
		{151, 237, 173, 317},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{147, 233, 177, 321},
		UserItem {
			disabled
		},
		/* [3] */
		{151, 137, 173, 217},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{151, 44, 173, 124},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{22, 12, 37, 240},
		StaticText {
			disabled,
			"Percentile for max concentration :"
		},
		/* [6] */
		{22, 243, 38, 301},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{83, 12, 103, 132},
		CheckBox {
			enabled,
			"Use smoothing"
		},
		/* [8] */
		{49, 12, 64, 245},
		StaticText {
			disabled,
			"Min. distance for offshore reflection :"
		},
		/* [9] */
		{49, 258, 65, 316},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{49, 323, 64, 351},
		StaticText {
			disabled,
			"km"
		},
		/* [11] */
		{111, 12, 131, 192},
		CheckBox {
			enabled,
			"Use line cross algorithm"
		}
	}
};

resource 'DITL' (1350) {
	{	/* array DITLarray: 69 elements */
		/* [1] */
		{359, 307, 379, 365},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{355, 303, 383, 369},
		UserItem {
			disabled
		},
		/* [3] */
		{359, 234, 379, 292},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{359, 164, 379, 222},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{11, 18, 27, 152},
		UserItem {
			enabled
		},
		/* [6] */
		{12, 214, 28, 264},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{11, 271, 30, 385},
		StaticText {
			disabled,
			"hours (after spill)"
		},
		/* [8] */
		{85, 18, 101, 162},
		StaticText {
			disabled,
			"Dispersant Efficiency:"
		},
		/* [9] */
		{86, 176, 102, 216},
		EditText {
			enabled,
			""
		},
		/* [10] */
		{85, 231, 104, 426},
		StaticText {
			disabled,
			" %  (percentage oil dispersed)"
		},
		/* [11] */
		{111, 18, 126, 62},
		StaticText {
			disabled,
			"API : "
		},
		/* [12] */
		{111, 77, 127, 117},
		EditText {
			enabled,
			""
		},
		/* [13] */
		{110, 130, 130, 187},
		StaticText {
			disabled,
			"meters"
		},
		/* [14] */
		{137, 17, 154, 236},
		CheckBox {
			enabled,
			"Dispersant Application Region"
		},
		/* [15] */
		{167, 47, 183, 112},
		StaticText {
			disabled,
			"Lat (Top):\n"
		},
		/* [16] */
		{168, 116, 184, 196},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{167, 114, 184, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [18] */
		{168, 147, 184, 174},
		EditText {
			enabled,
			""
		},
		/* [19] */
		{167, 185, 184, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [20] */
		{168, 218, 184, 258},
		EditText {
			enabled,
			""
		},
		/* [21] */
		{168, 218, 184, 238},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{167, 249, 185, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [23] */
		{168, 280, 184, 320},
		EditText {
			enabled,
			""
		},
		/* [24] */
		{167, 336, 185, 413},
		UserItem {
			enabled
		},
		/* [25] */
		{192, 35, 209, 110},
		StaticText {
			disabled,
			"Long (Left):"
		},
		/* [26] */
		{193, 116, 209, 196},
		EditText {
			enabled,
			""
		},
		/* [27] */
		{192, 114, 209, 144},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [28] */
		{193, 147, 209, 174},
		EditText {
			enabled,
			""
		},
		/* [29] */
		{192, 185, 209, 215},
		StaticText {
			disabled,
			"Min:"
		},
		/* [30] */
		{193, 218, 209, 258},
		EditText {
			enabled,
			""
		},
		/* [31] */
		{193, 218, 209, 238},
		EditText {
			enabled,
			""
		},
		/* [32] */
		{192, 249, 209, 277},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [33] */
		{193, 280, 209, 320},
		EditText {
			enabled,
			""
		},
		/* [34] */
		{192, 336, 210, 413},
		UserItem {
			enabled
		},
		/* [35] */
		{228, 23, 244, 109},
		StaticText {
			disabled,
			"Lat (Bottom):"
		},
		/* [36] */
		{228, 115, 244, 195},
		EditText {
			enabled,
			""
		},
		/* [37] */
		{227, 113, 244, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [38] */
		{228, 146, 244, 173},
		EditText {
			enabled,
			""
		},
		/* [39] */
		{227, 184, 244, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [40] */
		{228, 217, 244, 257},
		EditText {
			enabled,
			""
		},
		/* [41] */
		{228, 217, 244, 237},
		EditText {
			enabled,
			""
		},
		/* [42] */
		{227, 248, 245, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [43] */
		{228, 279, 244, 319},
		EditText {
			enabled,
			""
		},
		/* [44] */
		{227, 336, 245, 413},
		UserItem {
			enabled
		},
		/* [45] */
		{253, 26, 270, 113},
		StaticText {
			disabled,
			"Long (Right):"
		},
		/* [46] */
		{253, 115, 269, 195},
		EditText {
			enabled,
			""
		},
		/* [47] */
		{252, 113, 269, 143},
		StaticText {
			disabled,
			"Deg:"
		},
		/* [48] */
		{253, 146, 269, 173},
		EditText {
			enabled,
			""
		},
		/* [49] */
		{252, 184, 269, 214},
		StaticText {
			disabled,
			"Min:"
		},
		/* [50] */
		{253, 217, 269, 257},
		EditText {
			enabled,
			""
		},
		/* [51] */
		{253, 217, 269, 237},
		EditText {
			enabled,
			""
		},
		/* [52] */
		{252, 248, 269, 276},
		StaticText {
			disabled,
			"Sec:"
		},
		/* [53] */
		{253, 279, 269, 319},
		EditText {
			enabled,
			""
		},
		/* [54] */
		{252, 336, 270, 413},
		UserItem {
			enabled
		},
		/* [55] */
		{290, 16, 308, 145},
		RadioButton {
			enabled,
			"decimal degrees"
		},
		/* [56] */
		{309, 16, 327, 148},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [57] */
		{328, 16, 346, 208},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [58] */
		{155, 17, 285, 453},
		UserItem {
			disabled
		},
		/* [59] */
		{58, 18, 74, 80},
		StaticText {
			disabled,
			"Duration:"
		},
		/* [60] */
		{59, 89, 75, 119},
		EditText {
			enabled,
			""
		},
		/* [61] */
		{58, 129, 77, 168},
		StaticText {
			disabled,
			"hours"
		},
		/* [62] */
		{11, 217, 27, 329},
		UserItem {
			enabled
		},
		/* [63] */
		{11, 362, 27, 385},
		EditText {
			enabled,
			""
		},
		/* [64] */
		{11, 396, 27, 462},
		UserItem {
			enabled
		},
		/* [65] */
		{39, 354, 55, 382},
		EditText {
			enabled,
			""
		},
		/* [66] */
		{39, 400, 55, 428},
		EditText {
			enabled,
			""
		},
		/* [67] */
		{39, 388, 55, 396},
		StaticText {
			disabled,
			":"
		},
		/* [68] */
		{39, 246, 55, 343},
		StaticText {
			disabled,
			"Time (24-hr) :"
		},
		/* [69] */
		{3, 205, 66, 502},
		UserItem {
			disabled
		}
	}
};

resource 'DITL' (1355) {
	{	/* array DITLarray: 10 elements */
		/* [1] */
		{220, 199, 240, 257},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{216, 195, 244, 261},
		UserItem {
			disabled
		},
		/* [3] */
		{216, 112, 242, 172},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{77, 15, 199, 268},
		UserItem {
			enabled
		},
		/* [5] */
		{60, 34, 76, 109},
		StaticText {
			disabled,
			"Probability"
		},
		/* [6] */
		{59, 147, 76, 234},
		StaticText {
			disabled,
			"Droplet size"
		},
		/* [7] */
		{12, 22, 28, 207},
		StaticText {
			disabled,
			"Volume probability table"
		},
		/* [8] */
		{79, 284, 105, 369},
		Button {
			enabled,
			"Import File"
		},
		/* [9] */
		{121, 276, 147, 381},
		Button {
			enabled,
			"Default Values"
		},
		/* [10] */
		{161, 285, 187, 370},
		Button {
			enabled,
			"Save to File"
		}
	}
};

resource 'DITL' (1375) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{211, 176, 231, 234},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{207, 172, 235, 238},
		UserItem {
			disabled
		},
		/* [3] */
		{209, 244, 235, 404},
		Button {
			enabled,
			"Save Table to File"
		},
		/* [4] */
		{77, 15, 200, 422},
		UserItem {
			enabled
		},
		/* [5] */
		{59, 47, 76, 79},
		StaticText {
			disabled,
			"Time"
		},
		/* [6] */
		{60, 138, 76, 213},
		StaticText {
			disabled,
			"Evaporated"
		},
		/* [7] */
		{59, 251, 76, 318},
		StaticText {
			disabled,
			"Dispersed"
		},
		/* [8] */
		{12, 22, 28, 97},
		StaticText {
			disabled,
			"Oil Type :"
		},
		/* [9] */
		{12, 102, 28, 227},
		StaticText {
			disabled,
			""
		},
		/* [10] */
		{33, 22, 49, 67},
		StaticText {
			disabled,
			"API :"
		},
		/* [11] */
		{33, 77, 49, 152},
		StaticText {
			disabled,
			""
		},
		/* [12] */
		{59, 351, 76, 418},
		StaticText {
			disabled,
			"Removed"
		}
	}
};

resource 'DITL' (1380) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{158, 99, 178, 157},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{154, 95, 182, 161},
		UserItem {
			disabled
		},
		/* [3] */
		{156, 167, 182, 327},
		Button {
			enabled,
			"Save Table to File"
		},
		/* [4] */
		{24, 15, 147, 723},
		UserItem {
			enabled
		},
		/* [5] */
		{6, 18, 23, 59},
		StaticText {
			disabled,
			"Hour"
		},
		/* [6] */
		{7, 84, 23, 149},
		StaticText {
			disabled,
			"Released"
		},
		/* [7] */
		{6, 172, 23, 241},
		StaticText {
			disabled,
			"Floating"
		},
		/* [8] */
		{6, 262, 23, 341},
		StaticText {
			disabled,
			"Evaporated"
		},
		/* [9] */
		{6, 366, 23, 435},
		StaticText {
			disabled,
			"Dispersed"
		},
		/* [10] */
		{6, 460, 23, 529},
		StaticText {
			disabled,
			"Beached"
		},
		/* [11] */
		{6, 552, 23, 621},
		StaticText {
			disabled,
			"Off Map"
		},
		/* [12] */
		{6, 642, 23, 711},
		StaticText {
			disabled,
			"Removed"
		}
	}
};

resource 'DITL' (1390) {
	{	/* array DITLarray: 11 elements */
		/* [1] */
		{191, 99, 211, 157},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{187, 95, 215, 161},
		UserItem {
			disabled
		},
		/* [3] */
		{189, 167, 215, 327},
		Button {
			enabled,
			"Save Table to File"
		},
		/* [4] */
		{24, 15, 175, 733},
		UserItem {
			enabled
		},
		/* [5] */
		{6, 18, 23, 59},
		StaticText {
			disabled,
			"SegNo"
		},
		/* [6] */
		{7, 84, 23, 149},
		StaticText {
			disabled,
			"Start Pt"
		},
		/* [7] */
		{6, 172, 23, 241},
		StaticText {
			disabled,
			"End Point"
		},
		/* [8] */
		{6, 262, 23, 341},
		StaticText {
			disabled,
			"NumBeached"
		},
		/* [9] */
		{6, 366, 23, 435},
		StaticText {
			disabled,
			"LenInMiles"
		},
		/* [10] */
		{6, 460, 23, 529},
		StaticText {
			disabled,
			"Gallons"
		},
		/* [11] */
		{6, 552, 23, 621},
		StaticText {
			disabled,
			"Gal/Mile"
		},
		/* [12] */
		{6, 644, 23, 713},
		StaticText {
			disabled,
			"Gal/Foot"
		}
	}
};

resource 'DITL' (2850) {
	{	/* array DITLarray: 33 elements */
		/* [1] */
		{362, 241, 382, 307},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{358, 237, 386, 311},
		UserItem {
			disabled
		},
		/* [3] */
		{362, 164, 382, 230},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{362, 92, 382, 158},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{18, 63, 34, 244},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [6] */
		{44, 16, 62, 79},
		CheckBox {
			enabled,
			"Active"
		},
		/* [7] */
		{132, 183, 148, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [8] */
		{163, 183, 179, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [9] */
		{115, 16, 196, 361},
		UserItem {
			disabled
		},
		/* [10] */
		{132, 24, 149, 173},
		StaticText {
			disabled,
			"Diffusion Coefficient:"
		},
		/* [11] */
		{18, 17, 34, 60},
		StaticText {
			disabled,
			"Name:"
		},
		/* [12] */
		{132, 270, 148, 352},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [13] */
		{162, 24, 178, 166},
		StaticText {
			disabled,
			"Uncertainty Factor:"
		},
		/* [14] */
		{234, 183, 250, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [15] */
		{265, 183, 281, 258},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [16] */
		{217, 16, 298, 361},
		UserItem {
			disabled
		},
		/* [17] */
		{234, 24, 251, 173},
		StaticText {
			disabled,
			"Diffusion Coefficient:"
		},
		/* [18] */
		{234, 270, 250, 352},
		StaticText {
			disabled,
			"cm**2/sec"
		},
		/* [19] */
		{264, 24, 280, 166},
		StaticText {
			disabled,
			"Uncertainty Factor:"
		},
		/* [20] */
		{209, 21, 225, 78},
		StaticText {
			disabled,
			" Vertical"
		},
		/* [21] */
		{74, 16, 97, 361},
		UserItem {
			disabled
		},
		/* [22] */
		{162, 24, 178, 136},
		StaticText {
			disabled,
			"Wind Speed:"
		},
		/* [23] */
		{163, 153, 179, 228},
		EditText {
			enabled,
			""
		},
		/* [24] */
		{132, 24, 149, 143},
		StaticText {
			disabled,
			"Current Speed:"
		},
		/* [25] */
		{132, 153, 148, 228},
		EditText {
			enabled,
			""
		},
		/* [26] */
		{132, 240, 148, 302},
		StaticText {
			disabled,
			"m/sec"
		},
		/* [27] */
		{164, 240, 180, 302},
		StaticText {
			disabled,
			"m/sec"
		},
		/* [28] */
		{115, 16, 196, 361},
		UserItem {
			enabled
		},
		/* [29] */
		{108, 21, 124, 94},
		StaticText {
			disabled,
			" Horizontal"
		},
		/* [30] */
		{44, 180, 62, 383},
		CheckBox {
			enabled,
			"Depth Dependent Vertical"
		},
		/* [31] */
		{320, 24, 337, 193},
		StaticText {
			disabled,
			"Kz through the pycnocline:"
		},
		/* [32] */
		{320, 205, 336, 260},
		EditText {
			enabled,
			"Edit Text"
		},
		/* [33] */
		{320, 270, 336, 352},
		StaticText {
			disabled,
			"cm**2/sec"
		}
	}
};

resource 'DITL' (3809) {
	{	/* array DITLarray: 12 elements */
		/* [1] */
		{118, 279, 138, 350},
		Button {
			enabled,
			"Open"
		},
		/* [2] */
		{1152, 59, 1232, 77},
		Button {
			enabled,
			""
		},
		/* [3] */
		{180, 279, 200, 350},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{29, 271, 49, 357},
		UserItem {
			disabled
		},
		/* [5] */
		{55, 279, 75, 350},
		Button {
			enabled,
			"Eject"
		},
		/* [6] */
		{80, 279, 100, 350},
		Button {
			enabled,
			"Drive"
		},
		/* [7] */
		{54, 12, 200, 230},
		UserItem {
			enabled
		},
		/* [8] */
		{54, 229, 200, 246},
		UserItem {
			enabled
		},
		/* [9] */
		{106, 270, 107, 358},
		UserItem {
			disabled
		},
		/* [10] */
		{1044, 20, 1145, 116},
		StaticText {
			disabled,
			""
		},
		/* [11] */
		{149, 279, 169, 350},
		Button {
			enabled,
			"Help…"
		},
		/* [12] */
		{14, 16, 30, 242},
		StaticText {
			disabled,
			"Select a budget table file:"
		}
	}
};

resource 'DITL' (5100) {
	{	/* array DITLarray: 30 elements */
		/* [1] */
		{355, 263, 377, 343},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{351, 259, 381, 347},
		UserItem {
			disabled
		},
		/* [3] */
		{355, 163, 377, 243},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{355, 70, 377, 150},
		Button {
			enabled,
			"Help"
		},
		/* [5] */
		{18, 12, 34, 56},
		StaticText {
			disabled,
			"Name:"
		},
		/* [6] */
		{18, 63, 53, 354},
		StaticText {
			disabled,
			"map name"
		},
		/* [7] */
		{94, 10, 109, 124},
		StaticText {
			disabled,
			"Refloat Half Life:"
		},
		/* [8] */
		{94, 131, 110, 189},
		EditText {
			enabled,
			""
		},
		/* [9] */
		{93, 198, 113, 255},
		StaticText {
			disabled,
			"hours"
		},
		/* [10] */
		{126, 10, 141, 154},
		StaticText {
			disabled,
			"Contour depth range : "
		},
		/* [11] */
		{126, 164, 143, 211},
		EditText {
			enabled,
			""
		},
		/* [12] */
		{126, 220, 146, 243},
		StaticText {
			disabled,
			"to"
		},
		/* [13] */
		{126, 252, 143, 293},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{125, 305, 146, 333},
		StaticText {
			disabled,
			"m"
		},
		/* [15] */
		{58, 10, 78, 120},
		Button {
			enabled,
			"Replace Map"
		},
		/* [16] */
		{155, 10, 175, 110},
		Button {
			enabled,
			"Set Contours"
		},
		/* [17] */
		{197, 163, 216, 316},
		UserItem {
			disabled
		},
		/* [18] */
		{198, 11, 214, 160},
		StaticText {
			disabled,
			"Water Density (kg/m3):"
		},
		/* [19] */
		{200, 344, 216, 399},
		EditText {
			enabled,
			""
		},
		/* [20] */
		{277, 11, 296, 185},
		UserItem {
			disabled
		},
		/* [21] */
		{279, 246, 295, 286},
		EditText {
			enabled,
			""
		},
		/* [22] */
		{278, 296, 298, 353},
		UserItem {
			disabled
		},
		/* [23] */
		{308, 11, 327, 155},
		UserItem {
			enabled
		},
		/* [24] */
		{310, 212, 326, 252},
		EditText {
			enabled,
			""
		},
		/* [25] */
		{309, 262, 329, 319},
		UserItem {
			disabled
		},
		/* [26] */
		{156, 122, 175, 368},
		UserItem {
			disabled
		},
		/* [27] */
		{93, 282, 111, 440},
		CheckBox {
			enabled,
			"Contour Bottom"
		},
		/* [28] */
		{126, 161, 141, 260},
		StaticText {
			disabled,
			"Bottom Layer"
		},
		/* [29] */
		{240, 121, 259, 295},
		UserItem {
			disabled
		},
		/* [30] */
		{242, 10, 257, 114},
		StaticText {
			disabled,
			"Wave Height :"
		},
		/* [31] */
		{126, 271, 141, 315},
		EditText {
			enabled,
			""
		},
		/* [32] */
		{126, 325, 141, 345},
		StaticText {
			disabled,
			"m"
		}
	}
};

resource 'DITL' (5150) {
	{	/* array DITLarray: 21 elements */
		/* [1] */
		{287, 19, 307, 77},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{283, 15, 311, 81},
		UserItem {
			disabled
		},
		/* [3] */
		{287, 89, 307, 157},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{5, 6, 221, 185},
		UserItem {
			enabled
		},
		/* [5] */
		{11, 278, 27, 353},
		EditText {
			enabled,
			""
		},
		/* [6] */
		{36, 279, 54, 354},
		EditText {
			enabled,
			""
		},
		/* [7] */
		{90, 233, 114, 344},
		Button {
			enabled,
			"Make Levels"
		},
		/* [8] */
		{237, 209, 257, 269},
		Button {
			enabled,
			"Delete"
		},
		/* [9] */
		{238, 283, 258, 364},
		Button {
			enabled,
			"Delete All"
		},
		/* [10] */
		{12, 225, 29, 270},
		StaticText {
			disabled,
			"Initial:"
		},
		/* [11] */
		{65, 200, 81, 277},
		StaticText {
			disabled,
			"Increment:"
		},
		/* [12] */
		{39, 232, 55, 273},
		StaticText {
			disabled,
			"Final:"
		},
		/* [13] */
		{65, 279, 82, 353},
		EditText {
			enabled,
			""
		},
		/* [14] */
		{4, 192, 118, 366},
		UserItem {
			disabled
		},
		/* [15] */
		{146, 222, 163, 329},
		StaticText {
			disabled,
			"Contour Level"
		},
		/* [16] */
		{170, 227, 186, 302},
		EditText {
			enabled,
			""
		},
		/* [17] */
		{194, 207, 215, 355},
		Button {
			enabled,
			"Add Contour Level"
		},
		/* [18] */
		{136, 192, 220, 366},
		UserItem {
			disabled
		},
		/* [19] */
		{228, 7, 246, 185},
		StaticText {
			disabled,
			"Static Text"
		},
		/* [20] */
		{287, 165, 307, 223},
		Button {
			enabled,
			"Help…"
		},
		/* [21] */
		{286, 247, 306, 365},
		Button {
			enabled,
			"Default Values"
		}
	}
};

resource 'DITL' (5175) {
	{	/* array DITLarray: 7 elements */
		/* [1] */
		{168, 99, 188, 157},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{164, 95, 192, 161},
		UserItem {
			disabled
		},
		/* [3] */
		{166, 167, 192, 327},
		Button {
			enabled,
			"Save Table to File"
		},
		/* [4] */
		{24, 15, 147, 342},
		UserItem {
			enabled
		},
		/* [5] */
		{6, 18, 23, 79},
		StaticText {
			disabled,
			"Time"
		},
		/* [6] */
		{7, 145, 23, 200},
		StaticText {
			disabled,
			"Av Conc"
		},
		/* [7] */
		{6, 251, 23, 320},
		StaticText {
			disabled,
			"Max Conc"
		}
	}
};

resource 'MENU' (163, "P North-South 1", preload) {
	163,
	textMenuProc,
	allEnabled,
	enabled,
	"P North South 1",
	{	/* array: 2 elements */
		/* [1] */
		"North", noIcon, noKey, noMark, plain,
		/* [2] */
		"South", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (164, "P East-West 1", preload) {
	164,
	textMenuProc,
	allEnabled,
	enabled,
	"P East West 1",
	{	/* array: 2 elements */
		/* [1] */
		"East", noIcon, noKey, noMark, plain,
		/* [2] */
		"West", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (165, "P North-South 2", preload) {
	165,
	textMenuProc,
	allEnabled,
	enabled,
	"P North South 2",
	{	/* array: 2 elements */
		/* [1] */
		"North", noIcon, noKey, noMark, plain,
		/* [2] */
		"South", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (166, "P East-West 2", preload) {
	166,
	textMenuProc,
	allEnabled,
	enabled,
	"P East West 2",
	{	/* array: 2 elements */
		/* [1] */
		"East", noIcon, noKey, noMark, plain,
		/* [2] */
		"West", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (168, "Pollutants") {
	168,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 7 elements */
		/* [1] */
		"gasoline", noIcon, noKey, noMark, plain,
		/* [2] */
		"kerosene / jet fuels", noIcon, noKey, noMark, plain,
		/* [3] */
		"diesel", noIcon, noKey, noMark, plain,
		/* [4] */
		"fuel oil # 4", noIcon, noKey, noMark, plain,
		/* [5] */
		"medium crude", noIcon, noKey, noMark, plain,
		/* [6] */
		"fuel oil # 6", noIcon, noKey, noMark, plain,
		/* [7] */
		"non-weathering", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (181, "Map Types Menu") {
	181,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 2 elements */
		/* [1] */
		"OSSM", noIcon, noKey, noMark, plain,
		/* [2] */
		"Vector", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (182, "Novice LE Types Menu") {
	182,
	textMenuProc,
	allEnabled,
	enabled,
	"pNoviceLETypes",
	{	/* array: 2 elements */
		/* [1] */
		"Point/Line Source Splots", noIcon, noKey, noMark, plain,
		/* [2] */
		"Sprayed Splots", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (174, "pSPEEDUNITS") {
	174,
	textMenuProc,
	allEnabled,
	enabled,
	"pSPEEDUNITS",
	{	/* array: 3 elements */
		/* [1] */
		"knots", noIcon, noKey, noMark, plain,
		/* [2] */
		"meters / sec", noIcon, noKey, noMark, plain,
		/* [3] */
		"miles / hour", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (169, "Mover Types") {
	169,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 5 elements */
		/* [1] */
		"Currents", noIcon, noKey, noMark, plain,
		/* [2] */
		"Winds-Variable", noIcon, noKey, noMark, plain,
		/* [3] */
		"Winds-Constant", noIcon, noKey, noMark, plain,
		/* [4] */
		"Diffusion", noIcon, noKey, noMark, plain,
		/* [5] */
		"Component Mover", noIcon, noKey, noMark, plain,
		/* [6] */
		"Compound Current", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (175, "Directions") {
	175,
	textMenuProc,
	allEnabled,
	enabled,
	"Directions",
	{	/* array: 16 elements */
		/* [1] */
		"N", noIcon, noKey, noMark, plain,
		/* [2] */
		"NNE", noIcon, noKey, noMark, plain,
		/* [3] */
		"NE", noIcon, noKey, noMark, plain,
		/* [4] */
		"ENE", noIcon, noKey, noMark, plain,
		/* [5] */
		"E", noIcon, noKey, noMark, plain,
		/* [6] */
		"ESE", noIcon, noKey, noMark, plain,
		/* [7] */
		"SE", noIcon, noKey, noMark, plain,
		/* [8] */
		"SSE", noIcon, noKey, noMark, plain,
		/* [9] */
		"S", noIcon, noKey, noMark, plain,
		/* [10] */
		"SSW", noIcon, noKey, noMark, plain,
		/* [11] */
		"SW", noIcon, noKey, noMark, plain,
		/* [12] */
		"WSW", noIcon, noKey, noMark, plain,
		/* [13] */
		"W", noIcon, noKey, noMark, plain,
		/* [14] */
		"WNW", noIcon, noKey, noMark, plain,
		/* [15] */
		"NW", noIcon, noKey, noMark, plain,
		/* [16] */
		"NNW", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (159, "Days") {
	159,
	textMenuProc,
	allEnabled,
	enabled,
	"Days",
	{	/* array: 16 elements */
		/* [1] */
		"0", noIcon, noKey, noMark, plain,
		/* [2] */
		"1", noIcon, noKey, noMark, plain,
		/* [3] */
		"2", noIcon, noKey, noMark, plain,
		/* [4] */
		"3", noIcon, noKey, noMark, plain,
		/* [5] */
		"4", noIcon, noKey, noMark, plain,
		/* [6] */
		"5", noIcon, noKey, noMark, plain,
		/* [7] */
		"6", noIcon, noKey, noMark, plain,
		/* [8] */
		"7", noIcon, noKey, noMark, plain,
		/* [9] */
		"8", noIcon, noKey, noMark, plain,
		/* [10] */
		"9", noIcon, noKey, noMark, plain,
		/* [11] */
		"10", noIcon, noKey, noMark, plain,
		/* [12] */
		"11", noIcon, noKey, noMark, plain,
		/* [13] */
		"12", noIcon, noKey, noMark, plain,
		/* [14] */
		"13", noIcon, noKey, noMark, plain,
		/* [15] */
		"14", noIcon, noKey, noMark, plain,
		/* [16] */
		"15", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (176, "Month Menu") {
	176,
	textMenuProc,
	allEnabled,
	enabled,
	"Month",
	{	/* array: 12 elements */
		/* [1] */
		"January", noIcon, noKey, noMark, plain,
		/* [2] */
		"February", noIcon, noKey, noMark, plain,
		/* [3] */
		"March", noIcon, noKey, noMark, plain,
		/* [4] */
		"April", noIcon, noKey, noMark, plain,
		/* [5] */
		"May", noIcon, noKey, noMark, plain,
		/* [6] */
		"June", noIcon, noKey, noMark, plain,
		/* [7] */
		"July", noIcon, noKey, noMark, plain,
		/* [8] */
		"August", noIcon, noKey, noMark, plain,
		/* [9] */
		"September", noIcon, noKey, noMark, plain,
		/* [10] */
		"October", noIcon, noKey, noMark, plain,
		/* [11] */
		"November", noIcon, noKey, noMark, plain,
		/* [12] */
		"December", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (177, "Years") {
	177,
	textMenuProc,
	allEnabled,
	enabled,
	"Year",
	{	/* array: 13 elements */
		/* [1] */
		"1998", noIcon, noKey, noMark, plain,
		/* [2] */
		"1999", noIcon, noKey, noMark, plain,
		/* [3] */
		"2000", noIcon, noKey, noMark, plain,
		/* [4] */
		"2001", noIcon, noKey, noMark, plain,
		/* [5] */
		"2002", noIcon, noKey, noMark, plain,
		/* [6] */
		"2003", noIcon, noKey, noMark, plain,
		/* [7] */
		"2004", noIcon, noKey, noMark, plain,
		/* [8] */
		"2005", noIcon, noKey, noMark, plain,
		/* [9] */
		"2006", noIcon, noKey, noMark, plain,
		/* [10] */
		"2007", noIcon, noKey, noMark, plain,
		/* [11] */
		"2008", noIcon, noKey, noMark, plain,
		/* [12] */
		"2009", noIcon, noKey, noMark, plain,
		/* [13] */
		"2010", noIcon, noKey, noMark, plain,
		/* [14] */
		"2011", noIcon, noKey, noMark, plain,
		/* [15] */
		"2012", noIcon, noKey, noMark, plain,
		/* [16] */
		"2013", noIcon, noKey, noMark, plain,
		/* [17] */
		"2014", noIcon, noKey, noMark, plain,
		/* [18] */
		"2015", noIcon, noKey, noMark, plain,
		/* [19] */
		"2016", noIcon, noKey, noMark, plain,
		/* [20] */
		"2017", noIcon, noKey, noMark, plain,
		/* [21] */
		"2018", noIcon, noKey, noMark, plain,
		/* [22] */
		"2019", noIcon, noKey, noMark, plain,
		/* [23] */
		"2020", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (178, "Month Menu") {
	178,
	textMenuProc,
	allEnabled,
	enabled,
	"Month",
	{	/* array: 12 elements */
		/* [1] */
		"January", noIcon, noKey, noMark, plain,
		/* [2] */
		"February", noIcon, noKey, noMark, plain,
		/* [3] */
		"March", noIcon, noKey, noMark, plain,
		/* [4] */
		"April", noIcon, noKey, noMark, plain,
		/* [5] */
		"May", noIcon, noKey, noMark, plain,
		/* [6] */
		"June", noIcon, noKey, noMark, plain,
		/* [7] */
		"July", noIcon, noKey, noMark, plain,
		/* [8] */
		"August", noIcon, noKey, noMark, plain,
		/* [9] */
		"September", noIcon, noKey, noMark, plain,
		/* [10] */
		"October", noIcon, noKey, noMark, plain,
		/* [11] */
		"November", noIcon, noKey, noMark, plain,
		/* [12] */
		"December", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (179, "Years") {
	179,
	textMenuProc,
	allEnabled,
	enabled,
	"Year",
	{	/* array: 13 elements */
		/* [1] */
		"1998", noIcon, noKey, noMark, plain,
		/* [2] */
		"1999", noIcon, noKey, noMark, plain,
		/* [3] */
		"2000", noIcon, noKey, noMark, plain,
		/* [4] */
		"2001", noIcon, noKey, noMark, plain,
		/* [5] */
		"2002", noIcon, noKey, noMark, plain,
		/* [6] */
		"2003", noIcon, noKey, noMark, plain,
		/* [7] */
		"2004", noIcon, noKey, noMark, plain,
		/* [8] */
		"2005", noIcon, noKey, noMark, plain,
		/* [9] */
		"2006", noIcon, noKey, noMark, plain,
		/* [10] */
		"2007", noIcon, noKey, noMark, plain,
		/* [11] */
		"2008", noIcon, noKey, noMark, plain,
		/* [12] */
		"2009", noIcon, noKey, noMark, plain,
		/* [13] */
		"2010", noIcon, noKey, noMark, plain,
		/* [14] */
		"2011", noIcon, noKey, noMark, plain,
		/* [15] */
		"2012", noIcon, noKey, noMark, plain,
		/* [16] */
		"2013", noIcon, noKey, noMark, plain,
		/* [17] */
		"2014", noIcon, noKey, noMark, plain,
		/* [18] */
		"2015", noIcon, noKey, noMark, plain,
		/* [19] */
		"2016", noIcon, noKey, noMark, plain,
		/* [20] */
		"2017", noIcon, noKey, noMark, plain,
		/* [21] */
		"2018", noIcon, noKey, noMark, plain,
		/* [22] */
		"2019", noIcon, noKey, noMark, plain,
		/* [23] */
		"2020", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (180, "pSPEEDUNITS2") {
	180,
	textMenuProc,
	allEnabled,
	enabled,
	"pSPEEDUNITS2",
	{	/* array: 3 elements */
		/* [1] */
		"knots", noIcon, noKey, noMark, plain,
		/* [2] */
		"meters / sec", noIcon, noKey, noMark, plain,
		/* [3] */
		"miles / hour", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (183, "pMoverTypes") {
	183,
	textMenuProc,
	allEnabled,
	enabled,
	"pMoverTypes",
	{	/* array: 3 elements */
		/* [1] */
		"None", noIcon, noKey, noMark, plain,
		/* [2] */
		"Time File", noIcon, noKey, noMark, plain,
		/* [3] */
		"Wind Mover", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (184, "Intermediate LE Types Menu") {
	184,
	textMenuProc,
	allEnabled,
	enabled,
	"pIntermediateLETypes",
	{	/* array: 2 elements */
		/* [1] */
		"Point/Line Source Splots", noIcon, noKey, noMark, plain,
		/* [2] */
		"Sprayed Splots", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (185, "Advanced LE Types Menu") {
	185,
	textMenuProc,
	allEnabled,
	enabled,
	"pAdvancedLETypes",
	{	/* array: 5 elements */
		/* [1] */
		"Point/Line Source Splots", noIcon, noKey, noMark, plain,
		/* [2] */
		"Sprayed Splots", noIcon, noKey, noMark, plain,
		/* [3] */
		"Splots From GNOME Splots File", noIcon, noKey, noMark, plain,
		/* [4] */
		"Splot Grid", noIcon, noKey, noMark, plain,
		/* [5] */
		"Deep Well Blowout (CDOG)", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (186, "Time of Amount") {
	186,
	textMenuProc,
	allEnabled,
	enabled,
	"pTimeofAmount",
	{	/* array: 2 elements */
		/* [1] */
		"Time of Spill", noIcon, noKey, noMark, plain,
		/* [2] */
		"Overflight Time", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (187, "Scale Type") {
	187,
	textMenuProc,
	allEnabled,
	enabled,
	"pScaleByTypes",
	{	/* array: 2 elements */
		/* [1] */
		"Wind Speed", noIcon, noKey, noMark, plain,
		/* [2] */
		"Wind Stress", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (188, "Years Extended") {
	188,
	textMenuProc,
	allEnabled,
	enabled,
	"Year",
	{	/* array: 31 elements */
		/* [1] */
		"1980", noIcon, noKey, noMark, plain,
		/* [2] */
		"1981", noIcon, noKey, noMark, plain,
		/* [3] */
		"1982", noIcon, noKey, noMark, plain,
		/* [4] */
		"1983", noIcon, noKey, noMark, plain,
		/* [5] */
		"1984", noIcon, noKey, noMark, plain,
		/* [6] */
		"1985", noIcon, noKey, noMark, plain,
		/* [7] */
		"1986", noIcon, noKey, noMark, plain,
		/* [8] */
		"1987", noIcon, noKey, noMark, plain,
		/* [9] */
		"1988", noIcon, noKey, noMark, plain,
		/* [10] */
		"1989", noIcon, noKey, noMark, plain,
		/* [11] */
		"1990", noIcon, noKey, noMark, plain,
		/* [12] */
		"1991", noIcon, noKey, noMark, plain,
		/* [13] */
		"1992", noIcon, noKey, noMark, plain,
		/* [14] */
		"1993", noIcon, noKey, noMark, plain,
		/* [15] */
		"1994", noIcon, noKey, noMark, plain,
		/* [16] */
		"1995", noIcon, noKey, noMark, plain,
		/* [17] */
		"1996", noIcon, noKey, noMark, plain,
		/* [18] */
		"1997", noIcon, noKey, noMark, plain,
		/* [19] */
		"1998", noIcon, noKey, noMark, plain,
		/* [20] */
		"1999", noIcon, noKey, noMark, plain,
		/* [21] */
		"2000", noIcon, noKey, noMark, plain,
		/* [22] */
		"2001", noIcon, noKey, noMark, plain,
		/* [23] */
		"2002", noIcon, noKey, noMark, plain,
		/* [24] */
		"2003", noIcon, noKey, noMark, plain,
		/* [25] */
		"2004", noIcon, noKey, noMark, plain,
		/* [26] */
		"2005", noIcon, noKey, noMark, plain,
		/* [27] */
		"2006", noIcon, noKey, noMark, plain,
		/* [28] */
		"2007", noIcon, noKey, noMark, plain,
		/* [29] */
		"2008", noIcon, noKey, noMark, plain,
		/* [30] */
		"2009", noIcon, noKey, noMark, plain,
		/* [31] */
		"2010", noIcon, noKey, noMark, plain,
		/* [32] */
		"2011", noIcon, noKey, noMark, plain,
		/* [33] */
		"2012", noIcon, noKey, noMark, plain,
		/* [34] */
		"2013", noIcon, noKey, noMark, plain,
		/* [35] */
		"2014", noIcon, noKey, noMark, plain,
		/* [36] */
		"2015", noIcon, noKey, noMark, plain,
		/* [37] */
		"2016", noIcon, noKey, noMark, plain,
		/* [38] */
		"2017", noIcon, noKey, noMark, plain,
		/* [39] */
		"2018", noIcon, noKey, noMark, plain,
		/* [40] */
		"2019", noIcon, noKey, noMark, plain,
		/* [41] */
		"2020", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (189, "Years2 Extended") {
	189,
	textMenuProc,
	allEnabled,
	enabled,
	"Year",
	{	/* array: 31 elements */
		/* [1] */
		"1980", noIcon, noKey, noMark, plain,
		/* [2] */
		"1981", noIcon, noKey, noMark, plain,
		/* [3] */
		"1982", noIcon, noKey, noMark, plain,
		/* [4] */
		"1983", noIcon, noKey, noMark, plain,
		/* [5] */
		"1984", noIcon, noKey, noMark, plain,
		/* [6] */
		"1985", noIcon, noKey, noMark, plain,
		/* [7] */
		"1986", noIcon, noKey, noMark, plain,
		/* [8] */
		"1987", noIcon, noKey, noMark, plain,
		/* [9] */
		"1988", noIcon, noKey, noMark, plain,
		/* [10] */
		"1989", noIcon, noKey, noMark, plain,
		/* [11] */
		"1990", noIcon, noKey, noMark, plain,
		/* [12] */
		"1991", noIcon, noKey, noMark, plain,
		/* [13] */
		"1992", noIcon, noKey, noMark, plain,
		/* [14] */
		"1993", noIcon, noKey, noMark, plain,
		/* [15] */
		"1994", noIcon, noKey, noMark, plain,
		/* [16] */
		"1995", noIcon, noKey, noMark, plain,
		/* [17] */
		"1996", noIcon, noKey, noMark, plain,
		/* [18] */
		"1997", noIcon, noKey, noMark, plain,
		/* [19] */
		"1998", noIcon, noKey, noMark, plain,
		/* [20] */
		"1999", noIcon, noKey, noMark, plain,
		/* [21] */
		"2000", noIcon, noKey, noMark, plain,
		/* [22] */
		"2001", noIcon, noKey, noMark, plain,
		/* [23] */
		"2002", noIcon, noKey, noMark, plain,
		/* [24] */
		"2003", noIcon, noKey, noMark, plain,
		/* [25] */
		"2004", noIcon, noKey, noMark, plain,
		/* [26] */
		"2005", noIcon, noKey, noMark, plain,
		/* [27] */
		"2006", noIcon, noKey, noMark, plain,
		/* [28] */
		"2007", noIcon, noKey, noMark, plain,
		/* [29] */
		"2008", noIcon, noKey, noMark, plain,
		/* [30] */
		"2009", noIcon, noKey, noMark, plain,
		/* [31] */
		"2010", noIcon, noKey, noMark, plain,
		/* [32] */
		"2011", noIcon, noKey, noMark, plain,
		/* [33] */
		"2012", noIcon, noKey, noMark, plain,
		/* [34] */
		"2013", noIcon, noKey, noMark, plain,
		/* [35] */
		"2014", noIcon, noKey, noMark, plain,
		/* [36] */
		"2015", noIcon, noKey, noMark, plain,
		/* [37] */
		"2016", noIcon, noKey, noMark, plain,
		/* [38] */
		"2017", noIcon, noKey, noMark, plain,
		/* [39] */
		"2018", noIcon, noKey, noMark, plain,
		/* [40] */
		"2019", noIcon, noKey, noMark, plain,
		/* [41] */
		"2020", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (190, "Mover Types") {
	190,
	textMenuProc,
	allEnabled,
	enabled,
	"Time File Type",
	{	/* array: 5 elements */
		/* [1] */
		"No Time Series", noIcon, noKey, noMark, plain,
		/* [2] */
		"Shio Currents Coefficients", noIcon, noKey, noMark, plain,
		/* [3] */
		"Shio Heights Coefficients", noIcon, noKey, noMark, plain,
		/* [4] */
		"Current Time Series", noIcon, noKey, noMark, plain,
		/* [5] */
		"Hydrology Time Series", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (192, "pHydrologyInfo") {
	192,
	textMenuProc,
	allEnabled,
	enabled,
	"pHydrologyInfo",
	{	/* array: 2 elements */
		/* [1] */
		"Integrated Transport", noIcon, noKey, noMark, plain,
		/* [2] */
		"Velocity at Ref Pt for Specific Transpor"
		"t Value", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (193, "pTransportUnits") {
	193,
	textMenuProc,
	allEnabled,
	enabled,
	"pTransportUnits",
	{	/* array: 4 elements */
		/* [1] */
		"m3/s", noIcon, noKey, noMark, plain,
		/* [2] */
		"k(m3/s)", noIcon, noKey, noMark, plain,
		/* [3] */
		"CFS", noIcon, noKey, noMark, plain,
		/* [4] */
		"KCFS", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (194, "pPersistence") {
	194,
	textMenuProc,
	allEnabled,
	enabled,
	"pPersistence",
	{	/* array: 2 elements */
		/* [1] */
		"15 minutes", noIcon, noKey, noMark, plain,
		/* [2] */
		"Infinite", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (210, "Hydrate Process") {
	210,
	textMenuProc,
	allEnabled,
	enabled,
	"Hydrate Process",
	{	/* array: 2 elements */
		/* [1] */
		"do not suppress", noIcon, noKey, noMark, plain,
		/* [2] */
		"suppress", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (211, "Equilibrium Curves") {
	211,
	textMenuProc,
	allEnabled,
	enabled,
	"Equilibrium Curves",
	{	/* array: 2 elements */
		/* [1] */
		"Methane", noIcon, noKey, noMark, plain,
		/* [2] */
		"Natural Gas", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (212, "Separation of Gas From Plume") {
	212,
	textMenuProc,
	allEnabled,
	enabled,
	"Separation From Plume",
	{	/* array: 2 elements */
		/* [1] */
		"No Separation", noIcon, noKey, noMark, plain,
		/* [2] */
		"Separation Possible", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (213, "Temperature and Salinity Profiles") {
	213,
	textMenuProc,
	allEnabled,
	enabled,
	"Temperature & Salinity Profiles",
	{	/* array: 2 elements */
		/* [1] */
		"My files are already in the CDOG input f"
		"older", noIcon, noKey, noMark, plain,
		/* [2] */
		"Select files to move to CDOG input folde"
		"r", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (214) {
	214,
	textMenuProc,
	allEnabled,
	enabled,
	"Droplet Size",
	{	/* array: 2 elements */
		/* [1] */
		"Use CDOG default", noIcon, noKey, noMark, plain,
		/* [2] */
		"User supplied file", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (215) {
	215,
	textMenuProc,
	allEnabled,
	enabled,
	"Discharge Rate",
	{	/* array: 2 elements */
		/* [1] */
		"Oil Discharge Rate", noIcon, noKey, noMark, plain,
		/* [2] */
		"Gas Discharge Rate", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (216, "Hydrodynamic Options") {
	216,
	textMenuProc,
	allEnabled,
	enabled,
	"Hydrodynamic Options",
	{	/* array: 3 elements */
		/* [1] */
		"Select Folder with Currents", noIcon, noKey, noMark, plain,
		/* [2] */
		"Export Gnome currents with Profile", noIcon, noKey, noMark, plain,
		/* [3] */
		"My files are already in the CDOG input f"
		"older", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (195, "pAngleUnits") {
	195,
	textMenuProc,
	allEnabled,
	enabled,
	"pAngleUnits",
	{	/* array: 2 elements */
		/* [1] */
		"rad", noIcon, noKey, noMark, plain,
		/* [2] */
		"deg", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (196, "pTimeZone") {
	196,
	textMenuProc,
	allEnabled,
	enabled,
	"pTimeZones",
	{	/* array: 2 elements */
		/* [1] */
		"Local Time", noIcon, noKey, noMark, plain,
		/* [2] */
		"GMT / UTC", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (217, "pDischargeType") {
	217,
	textMenuProc,
	allEnabled,
	enabled,
	"DischargeType",
	{	/* array: 2 elements */
		/* [1] */
		"Constant Discharge", noIcon, noKey, noMark, plain,
		/* [2] */
		"Variable Discharge", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (218, "pCirculationInfo") {
	218,
	textMenuProc,
	allEnabled,
	enabled,
	"CirculationInfo",
	{	/* array: 4 elements */
		/* [1] */
		"I'll construct my own profile (constant "
		"in x,y)", noIcon, noKey, noMark, plain,
		/* [2] */
		"Export profile from GNOME hydrodynamic f"
		"ile", noIcon, noKey, noMark, plain,
		/* [3] */
		"My files are already in CDOG folder", noIcon, noKey, noMark, plain,
		/* [4] */
		"Move hydrodynamic files to CDOG input fo"
		"lder", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (219, "pYesNo") {
	219,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 2 elements */
		/* [1] */
		"Yes", noIcon, noKey, noMark, plain,
		/* [2] */
		"No", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (220, "pDiameterUnits") {
	220,
	textMenuProc,
	allEnabled,
	enabled,
	"pDiameterUnits",
	{	/* array: 3 elements */
		/* [1] */
		"m", noIcon, noKey, noMark, plain,
		/* [2] */
		"cm", noIcon, noKey, noMark, plain,
		/* [3] */
		"in", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (221, "pTempUnits") {
	221,
	textMenuProc,
	allEnabled,
	enabled,
	"pTempUnits",
	{	/* array: 2 elements */
		/* [1] */
		"deg C", noIcon, noKey, noMark, plain,
		/* [2] */
		"deg F", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (222, "pDensityUnits") {
	222,
	textMenuProc,
	allEnabled,
	enabled,
	"pDensityUnits",
	{	/* array: 2 elements */
		/* [1] */
		"kg/m3 at STP", noIcon, noKey, noMark, plain,
		/* [2] */
		"API", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (223, "pGOR") {
	223,
	textMenuProc,
	allEnabled,
	enabled,
	"pGOR",
	{	/* array: 3 elements */
		/* [1] */
		"SI Units", noIcon, noKey, noMark, plain,
		/* [2] */
		"SCFD/BOPD", noIcon, noKey, noMark, plain,
		/* [3] */
		"MSCF/BOPD", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (224, "pOilDischarge") {
	224,
	textMenuProc,
	allEnabled,
	enabled,
	"pOilDischarge",
	{	/* array: 2 elements */
		/* [1] */
		"m3/s", noIcon, noKey, noMark, plain,
		/* [2] */
		"BOPD", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (225, "pGasDischarge") {
	225,
	textMenuProc,
	allEnabled,
	enabled,
	"pGasDischarge",
	{	/* array: 2 elements */
		/* [1] */
		"m3/s", noIcon, noKey, noMark, plain,
		/* [2] */
		"MSCF", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (226, "pDepthUnits2") {
	226,
	textMenuProc,
	allEnabled,
	enabled,
	"pDepthUnits2",
	{	/* array: 3 elements */
		/* [1] */
		"meters", noIcon, noKey, noMark, plain,
		/* [2] */
		"feet", noIcon, noKey, noMark, plain,
		/* [3] */
		"fathoms", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (227, "pMolWt") {
	227,
	textMenuProc,
	allEnabled,
	enabled,
	"pMolWt",
	{	/* array: 2 elements */
		/* [1] */
		"g/mol", noIcon, noKey, noMark, plain,
		/* [2] */
		"kg/mol", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (228, "Advanced LE Types Menu2") {
	228,
	textMenuProc,
	allEnabled,
	enabled,
	"pAdvancedLETypes2",
	{	/* array: 4 elements */
		/* [1] */
		"Point/Line Source Splots", noIcon, noKey, noMark, plain,
		/* [2] */
		"Sprayed Splots", noIcon, noKey, noMark, plain,
		/* [3] */
		"Splots From GNOME Splots File", noIcon, noKey, noMark, plain,
		/* [4] */
		"Splot Grid", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (191, "pDispersantMode") {
	191,
	textMenuProc,
	allEnabled,
	enabled,
	"pDispersantMode",
	{	/* array: 5 elements */
		/* [1] */
		"No Dispersants", noIcon, noKey, noMark, plain,
		/* [2] */
		"Chemical Dispersants", noIcon, noKey, noMark, plain,
		/* [3] */
		"Natural Dispersion", noIcon, noKey, noMark, plain,
		/* [4] */
		"Chemical & Natural", noIcon, noKey, noMark, plain,
		/* [5] */
		"Natural & Removal", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (197, "pRandomInputType") {
	197,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 3 elements */
		/* [1] */
		"Horizontal and Vertical Eddy Diffusion", noIcon, noKey, noMark, plain,
		/* [2] */
		"Horizontal Eddy Diffusion", noIcon, noKey, noMark, plain,
		/* [3] */
		"Current and Wind Speed (m/s)", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (198, "pDISPTIMETYPE") {
	198,
	textMenuProc,
	allEnabled,
	enabled,
	"pDISPTIMETYPE",
	{	/* array: 2 elements */
		/* [1] */
		"Time To Disperse", noIcon, noKey, noMark, plain,
		/* [2] */
		"Hours After Spill", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (199) {
	199,
	textMenuProc,
	allEnabled,
	enabled,
	"pWaveHeight",
	{	/* array: 2 elements */
		/* [1] */
		"Breaking Wave Height", noIcon, noKey, noMark, plain,
		/* [2] */
		"Significant  Wave Height", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (201, "pMixedLayerDepth") {
	201,
	textMenuProc,
	allEnabled,
	enabled,
	"pMIXEDLAYERDEPTH",
	{	/* array: 2 elements */
		/* [1] */
		"Mixed Layer Depth", noIcon, noKey, noMark, plain,
		/* [2] */
		"Windrow Width", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (202, "pWaterDensity") {
	202,
	textMenuProc,
	allEnabled,
	enabled,
	"pWaterDensity",
	{	/* array: 4 elements */
		/* [1] */
		"1020 (Oceanic)", noIcon, noKey, noMark, plain,
		/* [2] */
		"1010 (Estuary)", noIcon, noKey, noMark, plain,
		/* [3] */
		"1000 (Fresh Water)", noIcon, noKey, noMark, plain,
		/* [4] */
		"Other", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (203, "pDepthUnits") {
	203,
	textMenuProc,
	allEnabled,
	enabled,
	"pDepthUnits",
	{	/* array: 2 elements */
		/* [1] */
		"meters", noIcon, noKey, noMark, plain,
		/* [2] */
		"feet", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (204, "pHeightUnits") {
	204,
	textMenuProc,
	allEnabled,
	enabled,
	"pHeightUnits",
	{	/* array: 2 elements */
		/* [1] */
		"meters", noIcon, noKey, noMark, plain,
		/* [2] */
		"feet", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (205, "pDiagnosticStrType") {
	205,
	textMenuProc,
	allEnabled,
	enabled,
	"pDiagnosticStrType",
	{	/* array: 6 elements */
		/* [1] */
		"No Grid Diagnostics", noIcon, noKey, noMark, plain,
		/* [2] */
		"Show Triangle Areas", noIcon, noKey, noMark, plain,
		/* [3] */
		"Show Number LEs in Triangles", noIcon, noKey, noMark, plain,
		/* [4] */
		"Show Concentration Levels", noIcon, noKey, noMark, plain,
		/* [5] */
		"Show Depth at Centers", noIcon, noKey, noMark, plain,
		/* [6] */
		"Show Subsurface Particles", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (206, "Level of Concern") {
	206,
	textMenuProc,
	allEnabled,
	enabled,
	"pLevelofConcern",
	{	/* array: 3 elements */
		/* [1] */
		"Low", noIcon, noKey, noMark, plain,
		/* [2] */
		"Medium", noIcon, noKey, noMark, plain,
		/* [3] */
		"High", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (207, "Species") {
	207,
	textMenuProc,
	allEnabled,
	enabled,
	"pSpecies",
	{	/* array: 7 elements */
		/* [1] */
		"Adult Fish", noIcon, noKey, noMark, plain,
		/* [2] */
		"Crustaceans", noIcon, noKey, noMark, plain,
		/* [3] */
		"Sens. Life Stages", noIcon, noKey, noMark, plain,
		/* [4] */
		"Adult Coral", noIcon, noKey, noMark, plain,
		/* [5] */
		"Stressed Coral", noIcon, noKey, noMark, plain,
		/* [6] */
		"Coral Eggs", noIcon, noKey, noMark, plain,
		/* [7] */
		"Sea Grass", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (208, "Check Box Type") {
	208,
	textMenuProc,
	allEnabled,
	enabled,
	"pCheckBoxType",
	{	/* array: 2 elements */
		/* [1] */
		"Level of Concern", noIcon, noKey, noMark, plain,
		/* [2] */
		"Species", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (209, "pWaveHtInput") {
	209,
	textMenuProc,
	allEnabled,
	enabled,
	"pWaveHtInput",
	{	/* array: 2 elements */
		/* [1] */
		"Compute from Wind Speed", noIcon, noKey, noMark, plain,
		/* [2] */
		"Enter by hand", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (229, "Pollutants") {
	229,
	textMenuProc,
	allEnabled,
	enabled,
	"Title",
	{	/* array: 8 elements */
		/* [1] */
		"gasoline", noIcon, noKey, noMark, plain,
		/* [2] */
		"kerosene / jet fuels", noIcon, noKey, noMark, plain,
		/* [3] */
		"diesel", noIcon, noKey, noMark, plain,
		/* [4] */
		"fuel oil # 4", noIcon, noKey, noMark, plain,
		/* [5] */
		"medium crude", noIcon, noKey, noMark, plain,
		/* [6] */
		"fuel oil # 6", noIcon, noKey, noMark, plain,
		/* [7] */
		"non-weathering", noIcon, noKey, noMark, plain,
		/* [8] */
		"chemical", noIcon, noKey, noMark, plain
	}
};

resource 'MENU' (10050, "Wind Types") {
	10050,
	textMenuProc,
	allEnabled,
	enabled,
	"Wind Types",
	{	/* array: 2 elements */
		/* [1] */
		"constant", noIcon, noKey, noMark, plain,
		/* [2] */
		"variable", noIcon, noKey, noMark, plain
	}
};

data 'pltt' (1000, "Main Window Color Palette") {
	$"0100 0000 0000 2004 0000 0000 0083 A428"            /* ...... ......É§( */
	$"FFFF FFFF FFFF 1502 0000 9502 0000 0000"            /* ˇˇˇˇˇˇ....ï..... */
	$"0000 0000 0000 1502 0000 1502 0000 0000"            /* ................ */
	$"FFFF FFFF 9999 1502 0000 1502 0000 0000"            /* ˇˇˇˇôô.......... */
	$"FFFF FFFF 6666 1502 0000 1502 0000 0000"            /* ˇˇˇˇff.......... */
	$"FFFF FFFF 3333 1502 0000 1502 0000 0000"            /* ˇˇˇˇ33.......... */
	$"FFFF FFFF 0000 1502 0000 1502 0000 0000"            /* ˇˇˇˇ............ */
	$"FFFF CCCC FFFF 1502 0000 1502 0000 0000"            /* ˇˇÃÃˇˇ.......... */
	$"FFFF CCCC CCCC 1502 0000 1502 0000 0000"            /* ˇˇÃÃÃÃ.......... */
	$"FFFF CCCC 9999 1502 0000 1502 0000 0000"            /* ˇˇÃÃôô.......... */
	$"FFFF CCCC 6666 1502 0000 1502 0000 0000"            /* ˇˇÃÃff.......... */
	$"FFFF CCCC 3333 1502 0000 1502 0000 0000"            /* ˇˇÃÃ33.......... */
	$"FFFF CCCC 0000 1502 0000 1502 0000 0000"            /* ˇˇÃÃ............ */
	$"FFFF 9999 FFFF 1502 0000 1502 0000 0000"            /* ˇˇôôˇˇ.......... */
	$"FFFF 9999 CCCC 1502 0000 1502 0000 0000"            /* ˇˇôôÃÃ.......... */
	$"FFFF 9999 9999 1502 0000 1502 0000 0000"            /* ˇˇôôôô.......... */
	$"FFFF 9999 6666 1502 0000 1502 0000 0000"            /* ˇˇôôff.......... */
	$"FFFF 9999 3333 1502 0000 1502 0000 0000"            /* ˇˇôô33.......... */
	$"FFFF 9999 0000 1502 0000 1502 0000 0000"            /* ˇˇôô............ */
	$"FFFF 6666 FFFF 1502 0000 1502 0000 0000"            /* ˇˇffˇˇ.......... */
	$"FFFF 6666 CCCC 1502 0000 1502 0000 0000"            /* ˇˇffÃÃ.......... */
	$"FFFF 6666 9999 1502 0000 1502 0000 0000"            /* ˇˇffôô.......... */
	$"FFFF 6666 6666 1502 0000 1502 0000 0000"            /* ˇˇffff.......... */
	$"FFFF 6666 3333 1502 0000 1502 0000 0000"            /* ˇˇff33.......... */
	$"FFFF 6666 0000 1502 0000 1502 0000 0000"            /* ˇˇff............ */
	$"FFFF 3333 FFFF 1502 0000 1502 0000 0000"            /* ˇˇ33ˇˇ.......... */
	$"FFFF 3333 CCCC 1502 0000 1502 0000 0000"            /* ˇˇ33ÃÃ.......... */
	$"FFFF 3333 9999 1502 0000 1502 0000 0000"            /* ˇˇ33ôô.......... */
	$"FFFF 3333 6666 1502 0000 1502 0000 0000"            /* ˇˇ33ff.......... */
	$"FFFF 3333 3333 1502 0000 1502 0000 0000"            /* ˇˇ3333.......... */
	$"FFFF 3333 0000 1502 0000 1502 0000 0000"            /* ˇˇ33............ */
	$"FFFF 0000 FFFF 1502 0000 1502 0000 0000"            /* ˇˇ..ˇˇ.......... */
	$"FFFF 0000 CCCC 1502 0000 1502 0000 0000"            /* ˇˇ..ÃÃ.......... */
	$"FFFF 0000 9999 1502 0000 1502 0000 0000"            /* ˇˇ..ôô.......... */
	$"FFFF 0000 6666 1502 0000 1502 0000 0000"            /* ˇˇ..ff.......... */
	$"FFFF 0000 3333 1502 0000 1502 0000 0000"            /* ˇˇ..33.......... */
	$"FFFF 0000 0000 1502 0000 1502 0000 0000"            /* ˇˇ.............. */
	$"CCCC FFFF FFFF 1502 0000 1502 0000 0000"            /* ÃÃˇˇˇˇ.......... */
	$"CCCC FFFF CCCC 1502 0000 1502 0000 0000"            /* ÃÃˇˇÃÃ.......... */
	$"CCCC FFFF 9999 1502 0000 1502 0000 0000"            /* ÃÃˇˇôô.......... */
	$"CCCC FFFF 6666 1502 0000 1502 0000 0000"            /* ÃÃˇˇff.......... */
	$"CCCC FFFF 3333 1502 0000 1502 0000 0000"            /* ÃÃˇˇ33.......... */
	$"CCCC FFFF 0000 1502 0000 1502 0000 0000"            /* ÃÃˇˇ............ */
	$"CCCC CCCC FFFF 1502 0000 1502 0000 0000"            /* ÃÃÃÃˇˇ.......... */
	$"CCCC CCCC CCCC 1502 0000 1502 0000 0000"            /* ÃÃÃÃÃÃ.......... */
	$"CCCC CCCC 9999 1502 0000 1502 0000 0000"            /* ÃÃÃÃôô.......... */
	$"CCCC CCCC 6666 1502 0000 1502 0000 0000"            /* ÃÃÃÃff.......... */
	$"CCCC CCCC 3333 1502 0000 1502 0000 0000"            /* ÃÃÃÃ33.......... */
	$"CCCC CCCC 0000 1502 0000 1502 0000 0000"            /* ÃÃÃÃ............ */
	$"CCCC 9999 FFFF 1502 0000 1502 0000 0000"            /* ÃÃôôˇˇ.......... */
	$"CCCC 9999 CCCC 1502 0000 1502 0000 0000"            /* ÃÃôôÃÃ.......... */
	$"CCCC 9999 9999 1502 0000 1502 0000 0000"            /* ÃÃôôôô.......... */
	$"CCCC 9999 6666 1502 0000 1502 0000 0000"            /* ÃÃôôff.......... */
	$"CCCC 9999 3333 1502 0000 1502 0000 0000"            /* ÃÃôô33.......... */
	$"CCCC 9999 0000 1502 0000 1502 0000 0000"            /* ÃÃôô............ */
	$"CCCC 6666 FFFF 1502 0000 1502 0000 0000"            /* ÃÃffˇˇ.......... */
	$"CCCC 6666 CCCC 1502 0000 1502 0000 0000"            /* ÃÃffÃÃ.......... */
	$"CCCC 6666 9999 1502 0000 1502 0000 0000"            /* ÃÃffôô.......... */
	$"CCCC 6666 6666 1502 0000 1502 0000 0000"            /* ÃÃffff.......... */
	$"CCCC 6666 3333 1502 0000 1502 0000 0000"            /* ÃÃff33.......... */
	$"CCCC 6666 0000 1502 0000 1502 0000 0000"            /* ÃÃff............ */
	$"CCCC 3333 FFFF 1502 0000 1502 0000 0000"            /* ÃÃ33ˇˇ.......... */
	$"CCCC 3333 CCCC 1502 0000 1502 0000 0000"            /* ÃÃ33ÃÃ.......... */
	$"CCCC 3333 9999 1502 0000 1502 0000 0000"            /* ÃÃ33ôô.......... */
	$"CCCC 3333 6666 1502 0000 1502 0000 0000"            /* ÃÃ33ff.......... */
	$"CCCC 3333 3333 1502 0000 1502 0000 0000"            /* ÃÃ3333.......... */
	$"CCCC 3333 0000 1502 0000 1502 0000 0000"            /* ÃÃ33............ */
	$"CCCC 0000 FFFF 1502 0000 1502 0000 0000"            /* ÃÃ..ˇˇ.......... */
	$"CCCC 0000 CCCC 1502 0000 1502 0000 0000"            /* ÃÃ..ÃÃ.......... */
	$"CCCC 0000 9999 1502 0000 1502 0000 0000"            /* ÃÃ..ôô.......... */
	$"CCCC 0000 6666 1502 0000 1502 0000 0000"            /* ÃÃ..ff.......... */
	$"CCCC 0000 3333 1502 0000 1502 0000 0000"            /* ÃÃ..33.......... */
	$"CCCC 0000 0000 1502 0000 1502 0000 0000"            /* ÃÃ.............. */
	$"9999 FFFF FFFF 1502 0000 1502 0000 0000"            /* ôôˇˇˇˇ.......... */
	$"9999 FFFF CCCC 1502 0000 1502 0000 0000"            /* ôôˇˇÃÃ.......... */
	$"9999 FFFF 9999 1502 0000 1502 0000 0000"            /* ôôˇˇôô.......... */
	$"9999 FFFF 6666 1502 0000 1502 0000 0000"            /* ôôˇˇff.......... */
	$"9999 FFFF 3333 1502 0000 1502 0000 0000"            /* ôôˇˇ33.......... */
	$"9999 FFFF 0000 1502 0000 1502 0000 0000"            /* ôôˇˇ............ */
	$"9DFA BD66 FFFF 1502 0000 1502 0000 0000"            /* ù˙Ωfˇˇ.......... */
	$"9999 CCCC CCCC 1502 0000 1502 0000 0000"            /* ôôÃÃÃÃ.......... */
	$"9999 CCCC 9999 1502 0000 1502 0000 0000"            /* ôôÃÃôô.......... */
	$"9999 CCCC 6666 1502 0000 1502 0000 0000"            /* ôôÃÃff.......... */
	$"9999 CCCC 3333 1502 0000 1502 0000 0000"            /* ôôÃÃ33.......... */
	$"9999 CCCC 0000 1502 0000 1502 0000 0000"            /* ôôÃÃ............ */
	$"9999 9999 FFFF 1502 0000 1502 0000 0000"            /* ôôôôˇˇ.......... */
	$"9999 9999 CCCC 1502 0000 1502 0000 0000"            /* ôôôôÃÃ.......... */
	$"9999 9999 9999 1502 0000 1502 0000 0000"            /* ôôôôôô.......... */
	$"9999 9999 6666 1502 0000 1502 0000 0000"            /* ôôôôff.......... */
	$"9999 9999 3333 1502 0000 1502 0000 0000"            /* ôôôô33.......... */
	$"9999 9999 0000 1502 0000 1502 0000 0000"            /* ôôôô............ */
	$"9999 6666 FFFF 1502 0000 1502 0000 0000"            /* ôôffˇˇ.......... */
	$"9999 6666 CCCC 1502 0000 1502 0000 0000"            /* ôôffÃÃ.......... */
	$"9999 6666 9999 1502 0000 1502 0000 0000"            /* ôôffôô.......... */
	$"9999 6666 6666 1502 0000 1502 0000 0000"            /* ôôffff.......... */
	$"9999 6666 3333 1502 0000 1502 0000 0000"            /* ôôff33.......... */
	$"9999 6666 0000 1502 0000 1502 0000 0000"            /* ôôff............ */
	$"9999 3333 FFFF 1502 0000 1502 0000 0000"            /* ôô33ˇˇ.......... */
	$"9999 3333 CCCC 1502 0000 1502 0000 0000"            /* ôô33ÃÃ.......... */
	$"9999 3333 9999 1502 0000 1502 0000 0000"            /* ôô33ôô.......... */
	$"9999 3333 6666 1502 0000 1502 0000 0000"            /* ôô33ff.......... */
	$"9999 3333 3333 1502 0000 1502 0000 0000"            /* ôô3333.......... */
	$"9999 3333 0000 1502 0000 1502 0000 0000"            /* ôô33............ */
	$"9999 0000 FFFF 1502 0000 1502 0000 0000"            /* ôô..ˇˇ.......... */
	$"9999 0000 CCCC 1502 0000 1502 0000 0000"            /* ôô..ÃÃ.......... */
	$"9999 0000 9999 1502 0000 1502 0000 0000"            /* ôô..ôô.......... */
	$"9999 0000 6666 1502 0000 1502 0000 0000"            /* ôô..ff.......... */
	$"9999 0000 3333 1502 0000 1502 0000 0000"            /* ôô..33.......... */
	$"9999 0000 0000 1502 0000 1502 0000 0000"            /* ôô.............. */
	$"6666 FFFF FFFF 1502 0000 1502 0000 0000"            /* ffˇˇˇˇ.......... */
	$"6666 FFFF CCCC 1502 0000 1502 0000 0000"            /* ffˇˇÃÃ.......... */
	$"6666 FFFF 9999 1502 0000 1502 0000 0000"            /* ffˇˇôô.......... */
	$"6666 FFFF 6666 1502 0000 1502 0000 0000"            /* ffˇˇff.......... */
	$"6666 FFFF 3333 1502 0000 1502 0000 0000"            /* ffˇˇ33.......... */
	$"6666 FFFF 0000 1502 0000 1502 0000 0000"            /* ffˇˇ............ */
	$"6666 CCCC FFFF 1502 0000 1502 0000 0000"            /* ffÃÃˇˇ.......... */
	$"6666 CCCC CCCC 1502 0000 1502 0000 0000"            /* ffÃÃÃÃ.......... */
	$"6666 CCCC 9999 1502 0000 1502 0000 0000"            /* ffÃÃôô.......... */
	$"6666 CCCC 6666 1502 0000 1502 0000 0000"            /* ffÃÃff.......... */
	$"6666 CCCC 3333 1502 0000 1502 0000 0000"            /* ffÃÃ33.......... */
	$"6666 CCCC 0000 1502 0000 1502 0000 0000"            /* ffÃÃ............ */
	$"600C 95E3 FFFF 1502 0000 1502 0000 0000"            /* `.ï„ˇˇ.......... */
	$"6666 9999 CCCC 1502 0000 1502 0000 0000"            /* ffôôÃÃ.......... */
	$"6666 9999 9999 1502 0000 1502 0000 0000"            /* ffôôôô.......... */
	$"6666 9999 6666 1502 0000 1502 0000 0000"            /* ffôôff.......... */
	$"6666 9999 3333 1502 0000 1502 0000 0000"            /* ffôô33.......... */
	$"6666 9999 0000 1502 0000 1502 0000 0000"            /* ffôô............ */
	$"6666 6666 FFFF 1502 0000 1502 0000 0000"            /* ffffˇˇ.......... */
	$"6666 6666 CCCC 1502 0000 1502 0000 0000"            /* ffffÃÃ.......... */
	$"6666 6666 9999 1502 0000 1502 0000 0000"            /* ffffôô.......... */
	$"6666 6666 6666 1502 0000 1502 0000 0000"            /* ffffff.......... */
	$"6666 6666 3333 1502 0000 1502 0000 0000"            /* ffff33.......... */
	$"6666 6666 0000 1502 0000 1502 0000 0000"            /* ffff............ */
	$"6666 3333 FFFF 1502 0000 1502 0000 0000"            /* ff33ˇˇ.......... */
	$"6666 3333 CCCC 1502 0000 1502 0000 0000"            /* ff33ÃÃ.......... */
	$"6666 3333 9999 1502 0000 1502 0000 0000"            /* ff33ôô.......... */
	$"6666 3333 6666 1502 0000 1502 0000 0000"            /* ff33ff.......... */
	$"6666 3333 3333 1502 0000 1502 0000 0000"            /* ff3333.......... */
	$"6666 3333 0000 1502 0000 1502 0000 0000"            /* ff33............ */
	$"6666 0000 FFFF 1502 0000 1502 0000 0000"            /* ff..ˇˇ.......... */
	$"6666 0000 CCCC 1502 0000 1502 0000 0000"            /* ff..ÃÃ.......... */
	$"6666 0000 9999 1502 0000 1502 0000 0000"            /* ff..ôô.......... */
	$"6666 0000 6666 1502 0000 1502 0000 0000"            /* ff..ff.......... */
	$"6666 0000 3333 1502 0000 1502 0000 0000"            /* ff..33.......... */
	$"6666 0000 0000 1502 0000 1502 0000 0000"            /* ff.............. */
	$"3333 FFFF FFFF 1502 0000 1502 0000 0000"            /* 33ˇˇˇˇ.......... */
	$"3333 FFFF CCCC 1502 0000 1502 0000 0000"            /* 33ˇˇÃÃ.......... */
	$"3333 FFFF 9999 1502 0000 1502 0000 0000"            /* 33ˇˇôô.......... */
	$"3333 FFFF 6666 1502 0000 1502 0000 0000"            /* 33ˇˇff.......... */
	$"3333 FFFF 3333 1502 0000 1502 0000 0000"            /* 33ˇˇ33.......... */
	$"3333 FFFF 0000 1502 0000 1502 0000 0000"            /* 33ˇˇ............ */
	$"3333 CCCC FFFF 1502 0000 1502 0000 0000"            /* 33ÃÃˇˇ.......... */
	$"3333 CCCC CCCC 1502 0000 1502 0000 0000"            /* 33ÃÃÃÃ.......... */
	$"3333 CCCC 9999 1502 0000 1502 0000 0000"            /* 33ÃÃôô.......... */
	$"3333 CCCC 6666 1502 0000 1502 0000 0000"            /* 33ÃÃff.......... */
	$"3333 CCCC 3333 1502 0000 1502 0000 0000"            /* 33ÃÃ33.......... */
	$"3333 CCCC 0000 1502 0000 1502 0000 0000"            /* 33ÃÃ............ */
	$"3333 9999 FFFF 1502 0000 1502 0000 0000"            /* 33ôôˇˇ.......... */
	$"3333 9999 CCCC 1502 0000 1502 0000 0000"            /* 33ôôÃÃ.......... */
	$"3333 9999 9999 1502 0000 1502 0000 0000"            /* 33ôôôô.......... */
	$"3333 9999 6666 1502 0000 1502 0000 0000"            /* 33ôôff.......... */
	$"3333 9999 3333 1502 0000 1502 0000 0000"            /* 33ôô33.......... */
	$"3333 9999 0000 1502 0000 1502 0000 0000"            /* 33ôô............ */
	$"3333 6666 FFFF 1502 0000 1502 0000 0000"            /* 33ffˇˇ.......... */
	$"3333 6666 CCCC 1502 0000 1502 0000 0000"            /* 33ffÃÃ.......... */
	$"3333 6666 9999 1502 0000 1502 0000 0000"            /* 33ffôô.......... */
	$"3333 6666 6666 1502 0000 1502 0000 0000"            /* 33ffff.......... */
	$"3333 6666 3333 1502 0000 1502 0000 0000"            /* 33ff33.......... */
	$"3333 6666 0000 1502 0000 1502 0000 0000"            /* 33ff............ */
	$"3333 3333 FFFF 1502 0000 1502 0000 0000"            /* 3333ˇˇ.......... */
	$"3333 3333 CCCC 1502 0000 1502 0000 0000"            /* 3333ÃÃ.......... */
	$"3333 3333 9999 1502 0000 1502 0000 0000"            /* 3333ôô.......... */
	$"3333 3333 6666 1502 0000 1502 0000 0000"            /* 3333ff.......... */
	$"3333 3333 3333 1502 0000 1502 0000 0000"            /* 333333.......... */
	$"3333 3333 0000 1502 0000 1502 0000 0000"            /* 3333............ */
	$"3333 0000 FFFF 1502 0000 1502 0000 0000"            /* 33..ˇˇ.......... */
	$"3333 0000 CCCC 1502 0000 1502 0000 0000"            /* 33..ÃÃ.......... */
	$"3333 0000 9999 1502 0000 1502 0000 0000"            /* 33..ôô.......... */
	$"3333 0000 6666 1502 0000 1502 0000 0000"            /* 33..ff.......... */
	$"3333 0000 3333 1502 0000 1502 0000 0000"            /* 33..33.......... */
	$"3333 0000 0000 1502 0000 1502 0000 0000"            /* 33.............. */
	$"0000 FFFF FFFF 1502 0000 1502 0000 0000"            /* ..ˇˇˇˇ.......... */
	$"0000 FFFF CCCC 1502 0000 1502 0000 0000"            /* ..ˇˇÃÃ.......... */
	$"0000 FFFF 9999 1502 0000 1502 0000 0000"            /* ..ˇˇôô.......... */
	$"0000 FFFF 6666 1502 0000 1502 0000 0000"            /* ..ˇˇff.......... */
	$"0000 FFFF 3333 1502 0000 1502 0000 0000"            /* ..ˇˇ33.......... */
	$"0000 FFFF 0000 1502 0000 1502 0000 0000"            /* ..ˇˇ............ */
	$"0000 CCCC FFFF 1502 0000 1502 0000 0000"            /* ..ÃÃˇˇ.......... */
	$"0000 CCCC CCCC 1502 0000 1502 0000 0000"            /* ..ÃÃÃÃ.......... */
	$"0000 CCCC 9999 1502 0000 1502 0000 0000"            /* ..ÃÃôô.......... */
	$"0000 CCCC 6666 1502 0000 1502 0000 0000"            /* ..ÃÃff.......... */
	$"0000 CCCC 3333 1502 0000 1502 0000 0000"            /* ..ÃÃ33.......... */
	$"0000 CCCC 0000 1502 0000 1502 0000 0000"            /* ..ÃÃ............ */
	$"0000 9999 FFFF 1502 0000 1502 0000 0000"            /* ..ôôˇˇ.......... */
	$"0000 9999 CCCC 1502 0000 1502 0000 0000"            /* ..ôôÃÃ.......... */
	$"0000 9999 9999 1502 0000 1502 0000 0000"            /* ..ôôôô.......... */
	$"0000 9999 6666 1502 0000 1502 0000 0000"            /* ..ôôff.......... */
	$"0000 9999 3333 1502 0000 1502 0000 0000"            /* ..ôô33.......... */
	$"0000 9999 0000 1502 0000 1502 0000 0000"            /* ..ôô............ */
	$"15A3 6342 FFFF 1502 0000 1502 0000 0000"            /* .£cBˇˇ.......... */
	$"0000 6666 CCCC 1502 0000 1502 0000 0000"            /* ..ffÃÃ.......... */
	$"0000 6666 9999 1502 0000 1502 0000 0000"            /* ..ffôô.......... */
	$"0000 6666 6666 1502 0000 1502 0000 0000"            /* ..ffff.......... */
	$"0000 6666 3333 1502 0000 1502 0000 0000"            /* ..ff33.......... */
	$"0000 6666 0000 1502 0000 1502 0000 0000"            /* ..ff............ */
	$"0000 3333 FFFF 1502 0000 1502 0000 0000"            /* ..33ˇˇ.......... */
	$"0000 3333 CCCC 1502 0000 1502 0000 0000"            /* ..33ÃÃ.......... */
	$"0000 3333 9999 1502 0000 1502 0000 0000"            /* ..33ôô.......... */
	$"0000 3333 6666 1502 0000 1502 0000 0000"            /* ..33ff.......... */
	$"0000 3333 3333 1502 0000 1502 0000 0000"            /* ..3333.......... */
	$"0000 3333 0000 1502 0000 1502 0000 0000"            /* ..33............ */
	$"0000 0000 FFFF 1502 0000 1502 0000 0000"            /* ....ˇˇ.......... */
	$"0000 0000 CCCC 1502 0000 1502 0000 0000"            /* ....ÃÃ.......... */
	$"0000 0000 9999 1502 0000 1502 0000 0000"            /* ....ôô.......... */
	$"0000 0000 6666 1502 0000 1502 0000 0000"            /* ....ff.......... */
	$"0000 0000 3333 1502 0000 1502 0000 0000"            /* ....33.......... */
	$"EEEE 0000 0000 1502 0000 1502 0000 0000"            /* ÓÓ.............. */
	$"DDDD 0000 0000 1502 0000 1502 0000 0000"            /* ››.............. */
	$"BBBB 0000 0000 1502 0000 1502 0000 0000"            /* ªª.............. */
	$"AAAA 0000 0000 1502 0000 1502 0000 0000"            /* ™™.............. */
	$"8888 0000 0000 1502 0000 1502 0000 0000"            /* àà.............. */
	$"7777 0000 0000 1502 0000 1502 0000 0000"            /* ww.............. */
	$"5555 0000 0000 1502 0000 1502 0000 0000"            /* UU.............. */
	$"4444 0000 0000 1502 0000 1502 0000 0000"            /* DD.............. */
	$"2222 0000 0000 1502 0000 1502 0000 0000"            /* "".............. */
	$"1111 0000 0000 1502 0000 1502 0000 0000"            /* ................ */
	$"0000 EEEE 0000 1502 0000 1502 0000 0000"            /* ..ÓÓ............ */
	$"0000 DDDD 0000 1502 0000 1502 0000 0000"            /* ..››............ */
	$"0000 BBBB 0000 1502 0000 1502 0000 0000"            /* ..ªª............ */
	$"0000 AAAA 0000 1502 0000 1502 0000 0000"            /* ..™™............ */
	$"0000 8888 0000 1502 0000 1502 0000 0000"            /* ..àà............ */
	$"0000 7777 0000 1502 0000 1502 0000 0000"            /* ..ww............ */
	$"0000 5555 0000 1502 0000 1502 0000 0000"            /* ..UU............ */
	$"0000 4444 0000 1502 0000 1502 0000 0000"            /* ..DD............ */
	$"0000 2222 0000 1502 0000 1502 0000 0000"            /* ..""............ */
	$"0000 1111 0000 1502 0000 1502 0000 0000"            /* ................ */
	$"0000 0000 EEEE 1502 0000 1502 0000 0000"            /* ....ÓÓ.......... */
	$"0000 0000 DDDD 1502 0000 1502 0000 0000"            /* ....››.......... */
	$"0000 0000 BBBB 1502 0000 1502 0000 0000"            /* ....ªª.......... */
	$"0000 0000 AAAA 1502 0000 1502 0000 0000"            /* ....™™.......... */
	$"0000 0000 8888 1502 0000 1502 0000 0000"            /* ....àà.......... */
	$"0000 0000 7777 1502 0000 1502 0000 0000"            /* ....ww.......... */
	$"0000 0000 5555 1502 0000 1502 0000 0000"            /* ....UU.......... */
	$"0000 0000 4444 1502 0000 1502 0000 0000"            /* ....DD.......... */
	$"0000 0000 2222 1502 0000 1502 0000 0000"            /* ...."".......... */
	$"0000 0000 1111 1502 0000 1502 0000 0000"            /* ................ */
	$"EEEE EEEE EEEE 1502 0000 1502 0000 0000"            /* ÓÓÓÓÓÓ.......... */
	$"DDDD DDDD DDDD 1502 0000 1502 0000 0000"            /* ››››››.......... */
	$"BBBB BBBB BBBB 1502 0000 1502 0000 0000"            /* ªªªªªª.......... */
	$"AAAA AAAA AAAA 1502 0000 1502 0000 0000"            /* ™™™™™™.......... */
	$"8888 8888 8888 1502 0000 1502 0000 0000"            /* àààààà.......... */
	$"7777 7777 7777 1502 0000 1502 0000 0000"            /* wwwwww.......... */
	$"5555 5555 5555 1502 0000 1502 0000 0000"            /* UUUUUU.......... */
	$"4444 4444 4444 1502 0000 1502 0000 0000"            /* DDDDDD.......... */
	$"2222 2222 2222 1502 0000 1502 0000 0000"            /* """""".......... */
	$"1111 1111 1111 1502 0000 1502 0000 0000"            /* ................ */
	$"FFFF FFFF CCCC 1502 0000 9502 0000 0000"            /* ˇˇˇˇÃÃ....ï..... */
};

resource 'dctb' (1660, "Edit Winds") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1665, "Wind Type") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1800, "M18: Wind Mover Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1600, "M16: CATS Mover Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1000, "M10: Model Times Dialog") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};
resource 'dctb' (1010, "Model Settings Hindcast") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};


resource 'dctb' (1200, "M12: Load / Define LEs") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1300, "M13: LE Info Dialog") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1400, "M14: Random Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1650, "Current Uncertainty") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1700, "M17: Choose Scaling Grid") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1900, "M19: Format selection") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2100, "M21: Load / Define Movers") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2200, "M22:  Save File Type") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2300, "M23: Load / Define Weatherers") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2400, "M24: Weatherer Name") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2600, "M26: Weather Info") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2700, "M27: Constant Mover Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2800, "M28: Simple Random Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3800, "M38: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3801, "M38b: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3802, "M38c: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3803, "M38d: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3804, "M38e: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (7900, "Time File Delete Dialog") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1060, "SELECTUNITS") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1050, "RUNUNTIL") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3805, "M38f: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5301, "M53C:Caveat") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5300, "M53") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1670, "Use Shio Ref Pt?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2000, "M20: Component Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2075, "Compound Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3150, "ADCP Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1500, "M15: Overflight Info Dialog") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5000, "M50") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3000, "M30: PtCurMover Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3100, "M31") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1680, "Change Model Start Time ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1682, "Change Model Start Time ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1681, "Change Model Start Time ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1690, "Are You Sure ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1685, "Exit Editing Mode ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1750, "Choose BNA Map") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2500, "M25: Vector Map Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1687, "Location File Expired") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3200, "M32: Hydrology") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1850) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3300, "M33: PtCurMover Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1688, "Have Topology File ?") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3900, "CDOG Spill") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3920, "CDOG Temp and Salinity Profiles") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3950, "Map Box Bounds") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3940, "CDOG Spill Settings") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3910, "CDOG Hydrodynamics") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3806) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3807, "M38i: SFGetFile") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3960, "CDOG Output Options") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3808, "Select a current file") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3930, "CDOG Hydrodynamics") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3970) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1691) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2150, "Load / Create Map") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3975, "CDOG Variable Discharge") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5200, "Movie Frame Interval") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3985, "CDOG Circulation Information") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1825) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2050) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3250, "Shio Heights File") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (128) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5225, "Anaylsis Options") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5275, "Toxicity Thresholds") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5280, "Set Axis Values") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5290, "Set Plot Axes") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5375, "Chemical parameters") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5350, "Smoothing parameters") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1350) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1355, "Droplet Table") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1375, "Adios Table") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1380, "Budget Table") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (1390, "Oiled Shoreline Table") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (2850, "M28b: 3D Random Mover") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (3809, "M38h: GetAdiosBudgetTable") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5100) {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5150, "Set Contour Levels") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

resource 'dctb' (5175, "Concentrations") {
	{	/* array ColorSpec: 5 elements */
		/* [1] */
		wContentColor, 56797, 56797, 56797,
		/* [2] */
		wFrameColor, 0, 0, 0,
		/* [3] */
		wTextColor, 0, 0, 0,
		/* [4] */
		wHiliteColor, 0, 0, 0,
		/* [5] */
		wTitleBarColor, 65535, 65535, 65535
	}
};

data 'TEXT' (128) {
	$"506C 6561 7365 2067 6F20 6261 636B 2061"            /* Please go back a */
	$"6E64 2073 656C 6563 7420 7468 6520 4D69"            /* nd select the Mi */
	$"6E69 6D75 6D20 5265 6772 6574 2063 6865"            /* nimum Regret che */
	$"636B 626F 7820 756E 6465 7220 5C22 4D6F"            /* ckbox under \"Mo */
	$"6465 6C20 5365 7474 696E 6773 5C22 2061"            /* del Settings\" a */
	$"6E64 2074 6865 6E20 7275 6E20 796F 7572"            /* nd then run your */
	$"2073 696D 756C 6174 696F 6E20 6167 6169"            /*  simulation agai */
	$"6E2E"                                               /* n. */
};

