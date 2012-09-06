#ifdef WIN32
	#ifdef LIB_GNOMEDLL_EXPORTS
		#define  GNOMEDLL_API __declspec(dllexport)
	#else
		#ifdef IBM
			#define GNOMEDLL_API	// for gui_gnome to work correctly
		#else
			#define  GNOMEDLL_API __declspec(dllimport) 
		#endif // IBM
	#endif // LIB_GNOMEDLL_EXPORTS
#else 
	// for linux/mac
	#define GNOMEDLL_API
#endif //WIN32
