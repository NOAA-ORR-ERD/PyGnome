
#include "MyRandom.h"

/////////////////////////////////////////////////

void ResetAllRandomSeeds(void)
{
	ResetMyRandomSeed();
	srand(1);
}
/////////////////////////////////////////////////


static long iv [NTAB];
static long idumForMyRandom = 1;
static long iy = 0; 

void ResetMyRandomSeed(void) // JLM 1/4/98
{
	long i;
	for(i = 0; i< NTAB; i++) iv[i] = 0;
	idumForMyRandom = 1;// JLM 11/12/99
	iy = 0; // JLM 11/12/99
}


float MyRandom (void)
{
	long 		j, k;
	float		temp;

	if (idumForMyRandom <= 0 || !iy)
	{
		if (-(idumForMyRandom) < 1)
			idumForMyRandom=1;
		else
			idumForMyRandom = -(idumForMyRandom);

		for (j=NTAB+7;j>=0;j--)
		{
			k=(idumForMyRandom)/IQ;
			idumForMyRandom=IA*(idumForMyRandom-k*IQ)-IR*k;
			if (idumForMyRandom < 0) idumForMyRandom += IM;
			if (j < NTAB) iv[j] = idumForMyRandom;
		}
		iy=iv[0];
	}
	
	k=(idumForMyRandom)/IQ;
	idumForMyRandom=IA*(idumForMyRandom-k*IQ)-IR*k;
	if (idumForMyRandom < 0) idumForMyRandom += IM;
	j=iy/NDIV;
	iy=iv[j];
	iv[j] = idumForMyRandom;
	
	temp=(float) (AM*iy);
	if (temp > RNMX)
		return RNMX;
	else
		return temp;
}

