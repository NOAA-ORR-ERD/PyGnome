
#ifndef __MakeMovie__
#define __MakeMovie__

#ifdef MAC
	#include <MoviesFormat.h>
    #include "Cross.h"
	pascal_ifMac ComponentInstance OPENSTDCOMPRESSION (void);
	#define	OpenStdCompression	OPENSTDCOMPRESSION
	#ifdef MPW
		enum{createMovieFileDontCreateResFile = 1L << 28};
	#endif
#endif

#ifdef MAC
Boolean CanMakeMovie(void);
#else
BOOL CanMakeMovie(void);
#endif

long InitMovies (void);
void CleanupMovieStuff(void);

long PICStoMovie (	char *moviePath,
					char *frameformatStr,short startIndex,short endIndex,
					short frameTop,short frameLeft, short frameBottom, short frameRight);

//	Convenient macros for error checking

#define	BailOnError(result)	if (result) goto error
#define	BailOnNil(p)		if (!p) { result = -1; goto error;}


#endif	// __MakeMovie__
