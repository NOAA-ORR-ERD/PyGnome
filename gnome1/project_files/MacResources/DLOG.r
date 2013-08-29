#ifndef CODEWARRIOR
	#include <Carbon/Carbon.r>
#endif

resource 'DLOG' (3801, "M38b: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3801,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1200, "M12: Load / Define LEs", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1200,
	"Add New Spill"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3800, "M38: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3800,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1300, "M13: LE Info Dialog") {
	{53, 14, 463, 700},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1300,
	"Spill Information"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1350, "Dispersant Data", purgeable) {
	{46, 82, 450, 595},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1350,
	"Dispersant Data"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1355, "Droplet Table", purgeable) {
	{46, 82, 304, 477},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1355,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1375, "Adios Table", purgeable) {
	{46, 82, 294, 512},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1375,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1380, "Budget Table", purgeable) {
	{46, 82, 244, 827},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1380,
	"Budget Table"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1390, "Oiled Shoreline Table", purgeable) {
	{46, 82, 274, 800},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1390,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3802, "M38c: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3802,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3803, "M38d: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3803,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

data 'DLOG' (1900, "M19: Format selection") {
	$"0035 002A 0175 0170 0005 0000 0000 0000"            /* .5.*.u.p........ */
	$"0000 076C 00"                                       /* ...l. */
};

resource 'DLOG' (1600, "M16: CATS Mover Settings") {
	{40, 40, 481, 479},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1600,
	"Current Mover Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1400, "M14: Random Mover") {
	{53, 14, 303, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1400,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

data 'DLOG' (1700, "M17: Choose Scaling Grid") {
	$"0028 0028 00D2 0141 0005 0000 0000 0000"            /* .(.(.Ò.A........ */
	$"0000 06A4 00"                                       /* ...¤. */
};

resource 'DLOG' (3804, "M38e: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3804,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2050, "Averaged Winds Parameters") {
	{53, 14, 243, 335},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2050,
	"Averaged Winds Parameters"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2100, "M21: Load / Define Movers") {
	{53, 14, 163, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2100,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2200, "M22:  Save File Type") {
	{53, 14, 203, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2200,
	"Save"
#ifndef MPW
	, centerMainScreen
#endif
};

data 'DLOG' (2300, "M23: Load / Define Weatherers") {
	$"0028 0028 0099 01C4 0005 0000 0000 0000"            /* .(.(.™.Ä........ */
	$"0000 08FC 00"                                       /* ...ü. */
};

data 'DLOG' (2400, "M24: Weatherer Name") {
	$"0028 0028 0092 01AA 0005 0000 0000 0000"            /* .(.(.’.ª........ */
	$"0000 0960 00"                                       /* ..Æ`. */
};

data 'DLOG' (2600, "M26: Weather Info") {
	$"0028 0028 01AC 0208 0000 0000 0000 0000"            /* .(.(.¬.......... */
	$"0000 0A28 00"                                       /* ...(. */
};

resource 'DLOG' (1800, "M18: Wind Settings") {
	{53, 14, 353, 355},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1800,
	"Wind Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (7900, "Time File Delete Dialog", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	7900,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2700, "M27: Constant Mover Settings") {
	{53, 14, 303, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2700,
	"Constant Wind"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1650, "Current Uncertainty") {
	{53, 14, 373, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1650,
	"Current Uncertainty"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1660, "Edit Winds") {
	{53, 14, 483, 570},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1660,
	"Wind"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1665, "Wind Type") {
	{53, 14, 288, 455},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1665,
	"Wind Type"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1050, "RUNUNTIL", purgeable, preload) {
	{42, 2, 260, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1050,
	"Run Until"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1060, "SELECTUNITS", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1060,
	"Specify Units"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3805, "M38g: General SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3805,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5301, "M53C:Caveat", purgeable, preload) {
	{42, 2, 330, 405},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5301,
	"Caveat"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5300, "M53 Output Header", purgeable, preload) {
	{42, 2, 340, 535},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5300,
	"Output Header"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1000, "M10: Model Settings") {
	{53, 14, 443, 470},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1000,
	"Model Settings"
#ifndef MPW
	, centerMainScreen
#endif
};
resource 'DLOG' (1010, "Model Settings Hindcast") {
	{53, 14, 493, 470},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1010,
	"Model Settings"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (2800, "M28: Simple Random Mover") {
	{53, 14, 283, 405},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2800,
	"Diffusion Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2850, "M28b: 3D Random Mover") {
	{53, 14, 463, 415},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2850,
	"3D Diffusion Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1670, "Use Shio Ref Pt ?", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1670,
	"Use Reference Pt ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5000, "M50: TMap Settings") {
	{53, 14, 423, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5000,
	"Map Settings"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (5100, "M51: PtCur Map Settings") {
	{53, 14, 453, 445},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5100,
	"Bathymetry Map Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5150, "Set Contour Levels") {
	{53, 14, 378, 405},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5150,
	"Set Contour Levels"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5175, "Concentrations") {
	{53, 14, 265, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5175,
	"Concentration"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3000, "M30: PtCurMover Settings") {
	{53, 14, 443, 440},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3000,
	"External Current Mover Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3100, "M31: Spray Can Settings") {
	{53, 14, 173, 275},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3100,
	"Spray Can Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1680, "Change Model Start Time ?", purgeable, preload) {
	{42, 2, 180, 425},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1680,
	"Change Model Start Time ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1681, "Change Model Start Time ?", purgeable, preload) {
	{42, 2, 180, 435},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1681,
	"Change Model Start Time ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1690, "Are You Sure ?", purgeable, preload) {
	{42, 2, 180, 425},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1690,
	"Are You Sure ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1500, "M15: Overflight Info Dialog") {
	{53, 14, 473, 605},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1500,
	"Overflight Information"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1685, "Exit Editing Mode ?", purgeable, preload) {
	{42, 2, 180, 425},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1685,
	"Exit Editing Mode ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1750, "Choose BNA Map", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1750,
	"Choose Map"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2500, "M25: Vector Map Settings") {
	{53, 14, 423, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2500,
	"Vector Map Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1682, "Change Model Start Time ?", purgeable, preload) {
	{42, 2, 180, 425},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1682,
	"Change Model Start Time ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1683, "Change Model Start Time ?", purgeable, preload) {
	{42, 2, 180, 425},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1683,
	"Change Model Start Time ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2000, "M20: Component Mover") {
	{53, 14, 661, 535},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2000,
	"Component Mover Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2075, "Compound Mover") {
	{53, 14, 643, 535},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2075,
	"Compound Current"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3150, "ADCP Mover") {
	{53, 14, 643, 535},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3150,
	"ADCP Current"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1687, "Download Location File", purgeable, preload) {
	{42, 2, 180, 455},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1687,
	"Download Location File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3200, "M32: Hydrology", purgeable, preload) {
	{42, 2, 380, 455},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3200,
	"Hydrology File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1850, "Windage Settings", purgeable, preload) {
	{42, 2, 210, 335},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1850,
	"Windage Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1825, "NetCDF WindMover Settings") {
	{53, 14, 493, 505},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1825,
	"NetCDF WindMover Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3300, "M33: NetCDFMover Settings") {
	{53, 14, 523, 485},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3300,
	"External Current Mover Settings"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1688, "Have Topology File ?", purgeable, preload) {
	{42, 2, 180, 435},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1688,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3900, "CDOG Spill Information") {
	{53, 14, 383, 700},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3900,
	"CDOG Spill Information"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3920, "CDOG Temp and Salinity Profiles", purgeable) {
	{46, 82, 254, 537},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3920,
	"CDOG Temp and Salinity Profiles"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3910, "CDOG Diffusivity", purgeable) {
	{46, 82, 254, 437},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3910,
	"CDOG Diffusivity and Time Step"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3950, "Map Box Bounds", purgeable) {
	{46, 82, 334, 567},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3950,
	"Map Box Bounds"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3930, "CDOG Hydrodynamics", purgeable) {
	{46, 82, 204, 527},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3930,
	"CDOG Hydrodynamics"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3940, "CDOG Settings") {
	{53, 14, 503, 525},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3940,
	"CDOG Model Parameters"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3806, "M38h: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3806,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3807, "M38i: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3807,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3960, "CDOG Output Options", purgeable) {
	{46, 82, 234, 427},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3960,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3975, "CDOG Variable Discharge") {
	{53, 14, 528, 810},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3975,
	"Variable Discharge"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3985, "CDOG Circulation Information", purgeable) {
	{46, 82, 314, 517},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3985,
	"CDOG Circulation Information"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3808, "M38j: SFGetFile", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3808,
	"Select a current file"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3809, "M38h: GetAdiosBudgetTable", purgeable) {
	{46, 82, 254, 487},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3809,
	"Select a current file"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3970, "Input CDOG Profiles") {
	{53, 14, 483, 420},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3970,
	"Input CDOG Profiles"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (3980, "About CDOG") {
	{53, 14, 443, 400},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	3980,
	"About CDOG"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (1691, "GMT Offsets", purgeable, preload) {
	{42, 2, 510, 355},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	1691,
	"GMT Offsets"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2150, "M21: Load / Create Map") {
	{53, 14, 183, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2150,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5200, "Movie Frame Interval", purgeable, preload) {
	{42, 2, 200, 345},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5200,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5230, "Analysis Options b", purgeable, preload) {
	{42, 2, 250, 470},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5230,
	"Analysis Options"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5380, "Oiled Shoreline Plot", purgeable, preload) {
	{42, 2, 400, 560},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5380,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5250, "Concentration Plot", purgeable, preload) {
	{42, 2, 395, 570},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5250,
	"Concentration vs Time"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5275, "Toxicity Thresholds", purgeable, preload) {
	{42, 2, 340, 405},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5275,
	"Toxicity Levels"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5280, "Set Axis Values", purgeable, preload) {
	{42, 2, 195, 315},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5280,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5290, "Set Plot Axes", purgeable, preload) {
	{42, 2, 200, 345},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5290,
	"Set Plot Axes"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5350, "Smoothing parameters", purgeable, preload) {
	{42, 2, 240, 360},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5350,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5375, "Chemical parameters", purgeable, preload) {
	{42, 2, 380, 495},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5375,
	"Chemical parameters"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (11700, "MESSAGE") {
	{62, 56, 162, 421},
#ifdef MPW
	dBoxProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	11700,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (11701, "CONFIRM") {
	{62, 56, 162, 421},
#ifdef MPW
	dBoxProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	11701,
	"Confirm"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (11702, "REQUEST") {
	{62, 56, 162, 421},
#ifdef MPW
	dBoxProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	11702,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (11703, "CHOICE") {
	{62, 56, 162, 421},
#ifdef MPW
	dBoxProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	11703,
	"Choice"
#ifndef MPW
	, centerMainScreen
#endif
};

/*****/

resource 'DLOG' (134, "Help", purgeable) {
	{0, 0, 301, 515},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	134,
	"GNOMEª Help"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (135, "Topics", purgeable) {
	{40, 52, 307, 364},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	135,
	"GNOMEª Help Topics"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (7500, "M75") {
	{42, 2, 119, 321},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	7500,
	"Exit ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (2900, "M29") {
	{56, 12, 157, 512},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	2900,
	"Please wait..."
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9001, "STOP") {
	{40, 40, 196, 380},
#ifdef MPW
	dBoxProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9001,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5800, "M58", purgeable) {
	{46, 82, 244, 452},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5800,
	"Open a File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (7600, "M76") {
	{42, 2, 119, 321},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	7600,
	"Delete ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (5500, "M55 Save file", purgeable, preload) {
	{46, 74, 246, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5500,
	"Save As"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (5400, "M54") {
	{64, 41, 289, 398},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	5400,
	"Movie Frames"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (7700, "M77C Save before close?", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	7700,
	"Save ?"
#ifndef MPW
	, centerMainScreen
#endif
};


resource 'DLOG' (7750, "M77Q Save before quit?", purgeable, preload) {
	{42, 2, 180, 395},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	7750,
	"Save ?"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9900, "Start Screen", purgeable, preload) {
	{38, 6, 380, 493},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9900,
	"Welcome to GNOME"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9901, "Welcome", purgeable, preload) {
	{35, 6, 265, 475},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9901,
	"Selecting a Location File"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9920, "Welcome", purgeable, preload) {
	{35, 6, 250, 510},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9920,
	"Location File Welcome"
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9950, "Enter Diagnostic Mode", purgeable, preload) {
	{40, 40, 345, 405},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9950,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9960, "Leave Diagnostic Mode", purgeable, preload) {
	{40, 40, 340, 460},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9960,
	""
#ifndef MPW
	, centerMainScreen
#endif
};

resource 'DLOG' (9990, "Almost Done", purgeable, preload) {
	{40, 4, 398, 501},
#ifdef MPW
	noGrowDocProc,
#else
	kWindowMovableModalDialogProc,
#endif
	invisible,
	noGoAway,
	0x0,
	9990,
	"Almost Done"
#ifndef MPW
	, centerMainScreen
#endif
};

data 'DLOG' (128) {
	$"0028 0028 00F0 0118 0000 0100 0100 0000"            /* .(.(.ð.......... */
	$"0000 0080 00"                                       /* ...€. */
};

