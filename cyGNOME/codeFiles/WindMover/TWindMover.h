#ifndef __TWindMover__
#define __TWindMover__

#include "WindMover_c.h"
#include "WindMover_g.h"
#include "Mover/TMover.h"

class TWindMover : virtual public WindMover_c, virtual public WindMover_g, virtual public TMover
{

public:
	TWindMover (TMap *owner, char* name);
	virtual			   ~TWindMover () { Dispose (); }
	virtual void		Dispose ();

};

#endif