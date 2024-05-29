
#ifdef _MSC_VER
	#if defined _EXPORTS
		#define DLL_API __declspec(dllexport)
	#elif defined _IMPORTS
		#define DLL_API __declspec(dllimport)
	#else
		#define DLL_API	// for standalone apps that don't use .dll
	#endif
#else 
	// for linux/mac
	#define DLL_API
#endif