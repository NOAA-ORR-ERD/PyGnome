//#include "types.r"

#ifndef CODEWARRIOR
	#include <Carbon/Carbon.r>
#endif



resource 'DITL' (100) {
	{
		/* [1] */
		{198, 10, 218, 68},
		Button {
			enabled,
			"OK"
		},
		/* [2] */
		{194, 6, 222, 72},
		UserItem {
			disabled
		},
		/* [3] */
		{198, 80, 218, 138},
		Button {
			enabled,
			"Cancel"
		},
		/* [4] */
		{198, 145, 218, 203},
		Button {
			enabled,
			"Help…"
		},
		/* [5] */
		{33, 6, 190, 7},
		UserItem {
			disabled
		},
		/* [6] */
		{33, 7, 34, 473},
		UserItem {
			disabled
		},
		/* [7] */
		{33, 472, 190, 473},
		UserItem {
			disabled
		},
		/* [8] */
		{189, 7, 190, 473},
		UserItem {
			disabled
		},
		/* [9] */
		{6, 17, 34, 77},
		UserItem {
			enabled
		},
		/* [10] */
		{6, 87, 34, 147},
		UserItem {
			enabled
		},
		/* [11] */
		{6, 157, 34, 217},
		UserItem {
			enabled
		},
		/* [12] */
		{6, 227, 34, 287},
		UserItem {
			enabled
		},
		/* [13] */
		{6, 297, 34, 357},
		UserItem {
			enabled
		},
		/* [1] */
		{76, 14, 94, 103},
		RadioButton {
			enabled,
			"degrees"
		},
		/* [2] */
		{94, 14, 111, 155},
		RadioButton {
			enabled,
			"degrees/minutes"
		},
		/* [5] */
		{111, 14, 128, 217},
		RadioButton {
			enabled,
			"degrees/minutes/seconds"
		},
		/* [6] */
		{132, 81, 150, 122},
		RadioButton {
			enabled,
			"top"
		},
		/* [7] */
		{132, 127, 150, 194},
		RadioButton {
			enabled,
			"bottom"
		},
		/* [8] */
		{56, 10, 74, 159},
		CheckBox {
			enabled,
			"show lat/long grid"
		},
		/* [11] */
		{133, 14, 149, 72},
		StaticText {
			disabled,
			"position:"
		},
		/* [12] */
		{155, 14, 173, 193},
		CheckBox {
			enabled,
			"show intermediate lines"
		},
		/* [13] */
		{95, 237, 111, 337},
		StaticText {
			disabled,
			"long line every"
		},
		/* [14] */
		{118, 229, 134, 337},
		StaticText {
			disabled,
			"long label every"
		},
		/* [15] */
		{141, 247, 157, 337},
		StaticText {
			disabled,
			"lat line every"
		},
		/* [16] */
		{164, 239, 180, 337},
		StaticText {
			disabled,
			"lat label every"
		},
		/* [17] */
		{95, 345, 111, 367},
		EditText {
			enabled,
			"00"
		},
		/* [18] */
		{118, 345, 134, 367},
		EditText {
			enabled,
			"00"
		},
		/* [19] */
		{141, 345, 157, 367},
		EditText {
			enabled,
			"00"
		},
		/* [20] */
		{164, 345, 180, 367},
		EditText {
			enabled,
			"00"
		},
		/* [21] */
		{94, 375, 112, 448},
		UserItem {
			disabled
		},
		/* [22] */
		{117, 375, 135, 448},
		UserItem {
			disabled
		},
		/* [23] */
		{140, 375, 158, 448},
		UserItem {
			disabled
		},
		/* [24] */
		{163, 375, 181, 448},
		UserItem {
			disabled
		},
		/* [25] */
		{73, 224, 91, 321},
		CheckBox {
			enabled,
			"custom grid"
		},
		/* [26] */
		{73, 10, 187, 469},
		UserItem {
			disabled
		},
		/* [27] */
		{90, 224, 185, 467},
		UserItem {
			disabled
		},
		/* [28] */
		{284, 312, 300, 387},
		EditText {
			enabled,
			"dummy"
		},
		/* [5] */
		{68, 38, 84, 150},
		StaticText {
			disabled,
			"Model Mode:"
		},
		/* [6] */
		{91, 49, 109, 155},
		RadioButton {
			enabled,
			"Standard"
		},
		/* [7] */
		/*{116, 49, 134, 155},
		RadioButton {
			enabled,
			"GIS Output"
		},*/
		/* [8] */
		/*{141, 49, 159, 155},
		RadioButton {
			enabled,
			"Diagnostic"
		},*/
		{116, 49, 134, 155},
		RadioButton {
			enabled,
			"Diagnostic"
		},
		/* [9] */
		{68, 263, 86, 404},
		StaticText {
			disabled,
			"Startup Model Mode:"
		},
		/* [10] */
		{91, 279, 109, 385},
		RadioButton {
			enabled,
			"Standard"
		},
		/* [11] */
		/*{116, 279, 134, 385},
		RadioButton {
			enabled,
			"GIS Output"
		},*/
		/* [12] */
		{116, 279, 134, 385},
		RadioButton {
			enabled,
			"Diagnostic"
		},
		/* [5] */
		{73, 14, 91, 350},
		CheckBox {
			enabled,
			"Disable Transition to U.S. Daylight Savings Time"
		},
		/* [9] */
		{116, 14, 152, 454},
		StaticText {
			disabled,
			"If you have already loaded a location or save file you will need to reload it, or restart GNOME for a change in this setting to take effect."
		}
	}
};

resource 'DLOG' (100, "M1") {
	{60, 20, 285, 499},
	documentProc,
	invisible,
	noGoAway,
	0x0,
	100,
	"Preferences"
	//#ifndef MPW
	, centerMainScreen
	//#endif
};

resource 'dctb' (100, "M1") {
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
