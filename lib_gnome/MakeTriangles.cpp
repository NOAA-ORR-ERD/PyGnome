#include "Basics.h"
#include "TypeDefs.h"
#include "MemUtils.h"
#include "DagTreeIO.h"
#include "my_build_list.h"

#ifndef pyGNOME
#include "CROSS.H"
#else
#include "Replacements.h"
#endif

typedef struct MySegment{
	long a;
	long b;
	long flg;
} MySegment;

long *g_v1=0,*g_v2=0,*g_v3=0,*g_n1=0,*g_n2=0,*g_n3=0;
double *gTrival=0;

#define SF 30000

Boolean CROSS(long x1,long y1,long x2,long y2,long x3,long y3,long x4,long y4)
{
	long p1Left,p2Left,p3Left,p4Left;
	Boolean rv;
	
	if((p3Left = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)) == 0)
	{
		rv = false;
		goto RET;
	}
	
	if((p4Left = (x2-x1)*(y4-y1)-(x4-x1)*(y2-y1)) == 0)
	{
		rv = false;
		goto RET;
	}
	
	if((p3Left > 0 && p4Left > 0) || (p3Left < 0 && p4Left < 0))
	{
		rv = false;
		goto RET;
	}
	
	if((p1Left = (x4-x3)*(y1-y3)-(x1-x3)*(y4-y3)) == 0)
	{
		rv = false;
		goto RET;
	}
	
	if((p2Left = (x4-x3)*(y2-y3)-(x2-x3)*(y4-y3)) == 0)
	{
		rv = false;
		goto RET;
	}
	rv = !((p1Left > 0 && p2Left > 0) || (p1Left < 0 && p2Left < 0));

RET:	
	return rv;
}


long HYPOT(long dx, long dy)
{
	register long x1;
	register long a;
	if(dx == 0)
	{
		x1 =  dy < 0 ? -dy :dy;
	}
	else if(dy == 0)
	{
		x1 =  dx < 0 ? -dx : dx;
	}
	else
	{
		a = dx * dx + dy * dy;
		x1 = 10;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
		x1 = (x1 + a/x1) >> 1;
	}
	return x1;
}

//double  GOOD(long x1,long y1,long x2,long y2,long x3,long y3)
double  GOOD(double x1,double y1,double x2,double y2,double x3,double y3)
{
	// Scale the triangle formed by p1,p2,p3 so that the perimeter
	// has length one then return the area of the resulting triangle.
	// The closer the triangle is to being equilateral - the larger the area

	//float a,b,c,s;
	//a = hypot(x1-x2,y1-y2);
	//b = hypot(x2-x3,y2-y3);
	//c = hypot(x3-x1,y3-y1);
	//s=(a+b+c)/2;
	//return ((s-a)*(s-b)*(s-c))/(s*s*s);
	
	// A better way would be to note that the area of the scaled triangle
	// is just 1/p^2 * the area of the original triangle where p is the perimeter
	// of the original triangle
	double p = sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)) +
							 sqrt((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3)) +
							sqrt((x3-x1)*(x3-x1)+(y3-y1)*(y3-y1));
	return fabs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))/(p * p);
} 

Boolean FIX(long badtri,long neig,long *x,long *y)
{
	long j1v,j2v,ver1,ver2,ver3,ver4,j3,j4,j5,j6;
	double fac;
	double x1,y1,x2,y2,x3,y3;
	double good1,good2;

	
	/* check to see if triangles are real and neighbors */
	
	if((badtri<0)||(neig<0)) return false;
	if((badtri!=g_n1[neig])&&(badtri!=g_n2[neig])&&(badtri!=g_n3[neig]))return false;
	
	/* two non common vertices are identified */
	
	if(g_n1[badtri]==neig)j1v=1;
	if(g_n2[badtri]==neig)j1v=2;
	if(g_n3[badtri]==neig)j1v=3;
	if(g_n1[neig]==badtri)j2v=1;
	if(g_n2[neig]==badtri)j2v=2;
	if(g_n3[neig]==badtri)j2v=3;
	
	/* common rectangle is labeled */
	
	if(j1v==1){
		ver1=g_v1[badtri];
		ver2=g_v2[badtri];
		ver3=g_v3[badtri];
		j3=g_n2[badtri];
		j4=g_n3[badtri];
	}
	if(j1v==2){
		ver1=g_v2[badtri];
		ver2=g_v3[badtri];
		ver3=g_v1[badtri];
		j3=g_n3[badtri];
		j4=g_n1[badtri];
	}
	if(j1v==3){
		ver1=g_v3[badtri];
		ver2=g_v1[badtri];
		ver3=g_v2[badtri];
		j3=g_n1[badtri];
		j4=g_n2[badtri];
	}
	if(j2v==1){
		ver4=g_v1[neig];
		j5=g_n2[neig];
		j6=g_n3[neig];
	}
	if(j2v==2){
		ver4=g_v2[neig];
		j5=g_n3[neig];
		j6=g_n1[neig];
	}
	if(j2v==3){
		ver4=g_v3[neig];
		j5=g_n1[neig];
		j6=g_n2[neig];
	}
	
	/* goodness of the first triangle is checked */
	
	x1=x[ver1];
	y1=y[ver1];
	x2=x[ver2];
	y2=y[ver2];
	x3=x[ver4];
	y3=y[ver4];
	good1=GOOD(x1,y1,x2,y2,x3,y3);
	
	/* check if first triangle is concave */
	
	fac=(x3-x1)*(y2-y1)-(x2-x1)*(y3-y1);
	if(fac<0) return false;
	
	/* goodness of the second triangle is checked */
	
	x1= x[ver4];
	y1= y[ver4];
	x2= x[ver3];
	y2= y[ver3];
	x3= x[ver1];
	y3= y[ver1];
	good2=GOOD(x1,y1,x2,y2,x3,y3);
	
	/* check if second triangle is concave */
	
	fac=(x3-x1)*(y2-y1)-(x2-x1)*(y3-y1);
	if(fac<0) return false;

	/* check to see if switch is appropriate */
	
	if((good1<=gTrival[badtri])||(good2<=gTrival[badtri])) return false;

	/* switch triangles */
	
	g_v1[badtri]=ver1;
	g_v2[badtri]=ver2;
	g_v3[badtri]=ver4;
	g_n1[badtri]=j5;
	g_n2[badtri]=neig;
	g_n3[badtri]=j4;
	gTrival[badtri]=good1;
	g_v1[neig]=ver4;
	g_v2[neig]=ver3;
	g_v3[neig]=ver1;
	g_n1[neig]=j3;
	g_n2[neig]=badtri;
	g_n3[neig]=j6;
	gTrival[neig]=good2;
	
	/* correct neighbor triangles */
	
	if(j5!=-1){
		if(g_n1[j5]==neig)g_n1[j5]=badtri;
		if(g_n2[j5]==neig)g_n2[j5]=badtri;
		if(g_n3[j5]==neig)g_n3[j5]=badtri;
	}
	if(j3!=-1){
		if(g_n1[j3]==badtri)g_n1[j3]=neig;
		if(g_n2[j3]==badtri)g_n2[j3]=neig;
		if(g_n3[j3]==badtri)g_n3[j3]=neig;
	}

	return true;
}

/*void InitCoordinates1(short *x, short *y, LongPointHdl ptsHdl, long nv)
{
	long nv = GetNumVertices();
	long i;
	for(i=0; i < nv; i++)
	{
		x[i] = round(SF * (*gCoord)[i].pLong);
		y[i] = round(SF * (*gCoord)[i].pLat);
	}
}

void InitCoordinates2(short *x, short *y, LongPointHdl ptsHdl, long nv)
{
	long nv = GetNumVertices();
	long i;
	for(i=0; i < nv; i++)
	{
		x[i] = round(SF * (*gCoord)[i].pLong);
		y[i] = round(SF * (1-(*gCoord)[i].pLat));
	}
}*/

// merged CATS functions LatLongTransform and InitCoordinates 1 and 2 into InitCoordinates, with a type 1 or 2 option, note sf and SF
//OSErr LatLongTransform(/*WorldRect *wr, */WORLDPOINTDH vertices, WORLDPOINTDH *theCoord)
//void InitCoordinates2(long *x, long *y)
// code goes here, throw in the checkboundary code for cw/ccw boundary errors
OSErr InitCoordinates(long *x, long *y, LongPointHdl vertices, long nv, short type)
{
	long i,npoints = _GetHandleSize((Handle)vertices)/sizeof(LongPoint);	
	double deg2rad = 3.14159/180;
	double R = 8000,dLat,dLong,Height,Width,scalefactor,tx,ty;
	double xmin=1e6,xmax=-1e6,ymin=1e6,ymax=-1e6;
	WORLDPOINTDH coord=0;
	
	if (!(coord = (WORLDPOINTDH) _NewHandleClear(sizeof(WorldPointD) * npoints))) {
		printError("Not enough memory in InitCoordinates");
		return -1;
	}

	for(i=0;i<npoints;i++)
	{
		(*coord)[i].pLat = (*vertices)[i].v / 1e6;
		(*coord)[i].pLong = (*vertices)[i].h / 1e6;
	}
	//GetVertexRange(vertices, &xmin,&xmax,&ymin,&ymax);	// defined in Iocats.c
	for(i=0;i<npoints;i++)
	{
		if((*coord)[i].pLat < ymin) ymin = (*coord)[i].pLat;
		if((*coord)[i].pLat > ymax) ymax = (*coord)[i].pLat;
		if((*coord)[i].pLong < xmin) xmin = (*coord)[i].pLong;
		if((*coord)[i].pLong > xmax) xmax = (*coord)[i].pLong;
	}
	dLat = ymax - ymin;
	dLong = xmax - xmin;
	Height = dLat * deg2rad * R;
	Width = dLong * deg2rad * R * cos((ymax + ymin)*deg2rad/2.0);
	scalefactor = (Height > Width) ? 1/Height : 1/Width;
	tx = scalefactor * Width/dLong;
	ty = scalefactor * Height/dLat;
	
	//wr->hiLat = 1e6 * ymax; wr->loLat = 1e6 * ymin;
	//wr->hiLong = 1e6 * xmax; wr->loLong = 1e6 * xmin;

	for(i= 0; i < npoints ; i++)
	{
		(*coord)[i].pLong = tx * ((*coord)[i].pLong - xmin);
		 (*coord)[i].pLat = ty * ((*coord)[i].pLat - ymin);
	}
	//MySetHandle((Handle *)theCoord,(Handle)coord);	// probably don't need to set handle if merge
	for(i=0; i < nv; i++)
	{
		x[i] = round(SF * (*coord)[i].pLong);
		if (type==1)
			y[i] = round(SF * (*coord)[i].pLat);
		else
			y[i] = round(SF * (1-(*coord)[i].pLat));
	}
	if (coord) {
		DisposeHandle((Handle)coord);
		coord = 0;
	}

	return 0;
}

Boolean maketriangles(TopologyHdl *topoHdl, LongPointHdl ptsH, long nv, LONGH boundarySegs, long nbounds) 
{
	long triscale = 2;
	char from[20],to[40],msg[256];
	//Boolean memerr = true;
	Boolean userCancel = false, morepoints;
	short err = 1; //AH 03/19/2012
	long i,j,k, nowlines,pt=0,flag1,flag2;
	//short i,j,k, nowlines,pt=0,flag1,flag2;	// if more than 32768 triangles crashes out
	long *x=0,*y=0;
	double s1,s2,s3;
	long x1,y1,x2,y2,x3,y3,length,side,xia,xib,yia,yib,dxi,dyi,xj,yj,xja,yja,xjb,yjb;
	long *p=0,ntri;
	Boolean linesCross;
	short neig, nlines=0,segstart=0, segCount = 0,changeflag, *changelist=0;
	//long nv = GetNumVertices();	// defined in Topology.c
	long nzerobytes = nv * sizeof(long);
	MySegment	*l=0;	
	//Rect r=MapDrawingRect();
	TopologyHdl tempTopoHdl = 0;

	//long nbounds = GetNumBoundaries();	// defined in Topology.c
	/*if(!(p =(long *) _NewPtrClear(sizeof(long)*nv)))goto errRecovery;
	if(!(l = (MySegment *)_NewPtr(8*nv *sizeof(MySegment))))goto errRecovery;
	if(!(g_v1 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;	
	if(!(g_v2 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_v3 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n1 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n2 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n3 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(x = (long *)_NewPtrClear(sizeof(long) * nv)))goto errRecovery;
	if(!(y = (long *)_NewPtrClear(sizeof(long) * nv)))goto errRecovery;*/

	p = (long *)calloc(nv,sizeof(long));
	if (p==NULL) {err = memFullErr; goto errRecovery;}
	l = (MySegment *)calloc(8*nv,sizeof(MySegment));
	if (l==NULL) {err = memFullErr; goto errRecovery;}
	g_v1 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v1==NULL) {err = memFullErr; goto errRecovery;}
	g_v2 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v2==NULL) {err = memFullErr; goto errRecovery;}
	g_v3 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v3==NULL) {err = memFullErr; goto errRecovery;}
	g_n1 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n1==NULL) {err = memFullErr; goto errRecovery;}
	g_n2 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n2==NULL) {err = memFullErr; goto errRecovery;}
	g_n3 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n3==NULL) {err = memFullErr; goto errRecovery;}
	x = (long *)calloc(nv,sizeof(long));
	if (x==NULL) {err = memFullErr; goto errRecovery;}
	y = (long *)calloc(nv,sizeof(long));
	if (y==NULL) {err = memFullErr; goto errRecovery;}
	
	//memerr = false;

	//PenNormal();
	cerr << "nbounds = " << nbounds << endl;
	for (;;) {
		if (segCount == nbounds)
			break;
		l[nlines].a = pt;
		l[nlines].flg = -1;

		if (pt == (*boundarySegs)[segCount]) {
			segCount++;
			l[nlines].b = segstart;
			segstart = pt + 1;
		}
		else{
			l[nlines].b = pt + 1;
		}		
		pt++;
		nlines++;
	}

	//InitCoordinates1(x,y,ptsH,nv);
	err = InitCoordinates(x, y, ptsH, nv, 1);
	if (err != noErr)
		goto errRecovery;

	/* enter triangle generation loop */
	// make sure triangles don't have 3 points in the same array row/col in netcdf curvilinear case
	cerr << "entering triangle generation loop..."<< endl;

	ntri = -1;
	for (;;) {
		if (nlines <= 0)
			break;

		nowlines = nlines;
		for (i = 0; i < nowlines; i++)
		{
			if(l[i].flg==-2)continue;
			xia = x[l[i].a]; yia = y[l[i].a];
			xib = x[l[i].b]; yib = y[l[i].b];
			dxi = xib - xia; dyi = yib - yia;
			memset(p,1,nzerobytes); //equivalent to for(j=0;j<nv;j++)p[j]=1;
			p[l[i].a]=0; p[l[i].b]=0;

			for(;;)
			{
				length=10000000;
				morepoints = false;
				pt = -1;
				for(j=0; j<nv; j++)
				{
					if(CmdPeriod())
					{
						userCancel= true;
						goto errRecovery;
					}
					if(j>1000 && j%1000 == 0) 
						MySpinCursor(); 
					xj = x[j] ; yj = y[j];
					//cerr << "maketriangles():"
					//	 << " xj = " << xj
					//	 << " yj = " << yj << endl;

					if (dxi*(yj-yia) > dyi*(xj-xia) &&
						p[j] &&
						(side = HYPOT(xj - xia, yj - yia) + HYPOT(xj - xib, yj - yib)) < length)
					{
								length=side;
								pt=j;
					}
					morepoints = morepoints || p[j] != 0;
				}
				
				if(pt == -1)
				{
					err = true;
					if(morepoints) 	// something is wrong
					{
						sprintf(from,"%ld",l[i].a);
						sprintf(to,"%ld",l[i].b);
						strcpy(msg,"Problem at segment with endpoints [");
						strcat(msg,from); strcat(msg," ,"); strcat(msg,to);strcat(msg,"]");
						strcat(msg," There may be a problem with boundary orientation.");
						printError(msg);	
						SysBeep(5);
					}
					else printError("Could not generate triangles.");
					goto errRecovery;
				}
				
				linesCross=false;
				x2=x[pt];
				y2=y[pt];
				for(j=0;j<nlines;j++)
				{
					xja = x[l[j].a]; yja = y[l[j].a];
					xjb = x[l[j].b]; yjb = y[l[j].b];
					if(CROSS(xia,yia,x2,y2,xja,yja,xjb,yjb) || CROSS(xib,yib,x2,y2,xja,yja,xjb,yjb))	
					{
						linesCross=true;
						p[pt]=0;
						break;
					}
				}				
				if(linesCross)continue;
				break;
			}
			/* add triangle to triangle list */
			ntri=ntri+1;
			if(ntri > triscale*nv)
			{
				printError("Problem generating triangles. There may be a problem with "
				"boundary orientation.");
				goto errRecovery;
			}
			
			g_v1[ntri]=l[i].a;
			g_v2[ntri]=l[i].b;
			g_v3[ntri]=pt;
			k=l[i].flg;
			g_n3[ntri]=k;
			if(k>=0){
				if((g_v1[k]!=l[i].a)&&(g_v1[k]!=l[i].b))g_n1[k]=ntri;
				if((g_v2[k]!=l[i].a)&&(g_v2[k]!=l[i].b))g_n2[k]=ntri;
			}
			
			/* check if lines are on list already */
			
			flag1=-1;
			flag2=-1;
			for(k=0;k<nlines;k++){
				if(((l[k].a==l[i].a)&&(l[k].b==pt))||
					 ((l[k].b==l[i].a)&&(l[k].a==pt)))flag1=k;
				if(((l[k].a==l[i].b)&&(l[k].b==pt))||
					 ((l[k].b==l[i].b)&&(l[k].a==pt)))flag2=k;
			}
			
			/* add first new line segment */
			
			if(flag1==-1){
				l[nlines].a=l[i].a;
				l[nlines].b=pt;
				l[nlines].flg=ntri;	
				nlines=nlines+1;
			}
			else{
			  k=l[flag1].flg;
				g_n2[ntri]=k;
				if(k>=0){
					if((g_v1[k]!=l[i].a)&&(g_v1[k]!=pt))g_n1[k]=ntri;
					if((g_v2[k]!=l[i].a)&&(g_v2[k]!=pt))g_n2[k]=ntri;
					if((g_v3[k]!=l[i].a)&&(g_v3[k]!=pt))g_n3[k]=ntri;
				}
				l[flag1].flg=-2;
			}
			
			/* add second new line segment */
			
			if(flag2==-1){
				l[nlines].a=pt;
				l[nlines].b=l[i].b;
				l[nlines].flg=ntri;
				nlines=nlines+1;
			}
			else{
				k=l[flag2].flg;
				g_n1[ntri]=k;
				if(k>=0){
					if((g_v1[k]!=pt)&&(g_v1[k]!=l[i].b))g_n1[k]=ntri;
					if((g_v2[k]!=pt)&&(g_v2[k]!=l[i].b))g_n2[k]=ntri;
					if((g_v3[k]!=pt)&&(g_v3[k]!=l[i].b))g_n3[k]=ntri;
				}
				l[flag2].flg=-2;
			}
			
			/* remove used line segment*/ 
			
			l[i].flg=-2;
		}
		
		/* compress line list */
		
		j=0;
		for(i=0;i<nlines;i++){
			if(l[i].flg==-2){
				j++;
			}
			else{
				l[i-j]=l[i];
			}
		}	
		nlines -= j;
	}
	
	ntri++;
	
	
	/* calculate goodness of triangles */
	//float gTrival[TRIANGLES];
	//short changeflag,changelist[TRIANGLES];

	//if(!(gTrival = (double *)_NewPtrClear(sizeof(double)*ntri)))goto errRecovery;	
	//if(!(changelist = (short *)_NewPtrClear(sizeof(short) *ntri)))goto errRecovery;
	gTrival = (double *)calloc(ntri,sizeof(double));
	if (gTrival==NULL) {err = memFullErr; goto errRecovery;}
	changelist = (short *)calloc(ntri,sizeof(short));
	if (changelist==NULL) {err = memFullErr; goto errRecovery;}

	//InitCoordinates2(x,y,ptsH,nv); 	
	if ((err = InitCoordinates(x, y, ptsH, nv, 2)) != 0) goto errRecovery;

	for(i=0;i<ntri;i++){
		x1= x[g_v1[i]];
		y1= y[g_v1[i]];
		x2= x[g_v2[i]];
		y2= y[g_v2[i]];
		x3= x[g_v3[i]];
		y3= y[g_v3[i]];
		gTrival[i]=GOOD(x1,y1,x2,y2,x3,y3);
		changelist[i]=1;
	}
	
	changeflag=0;
	for(j=0;j<10;j++){
		for(i=0;i< ntri;i++){
			if(changelist[i]==0)continue;
			
			
			x1= x[g_v1[i]];
			y1= y[g_v1[i]];
			x2= x[g_v2[i]];
			y2= y[g_v2[i]];
			x3= x[g_v3[i]];
			y3= y[g_v3[i]];
			
			s1 = labs(x2-x3) + labs(y2-y3);
			s2 = labs(x3-x1) + labs(y3 - y1);
			s3 = labs(x1-x2) + labs(y1-y2);
			if(s1>s2 && s1>s3)
			{
				side=s1;
			}
			else
			{
				if(s2>s3)
				{
					side=s2;
				}
				else
				{
					side=s3;
				}
			}
			if(side==s1)neig=g_n1[i];
			if(side==s2)neig=g_n2[i];
			if(side==s3)neig=g_n3[i];
			if(!FIX(i,neig,x,y))		
			{
				changelist[i]=0;
				changeflag=changeflag+1;
			}
			else
			{
				if(i!=-1)changelist[i]=1;
				if(g_n1[i]!=-1)changelist[g_n1[i]]=1;
				if(g_n2[i]!=-1)changelist[g_n2[i]]=1;
				if(g_n3[i]!=-1)changelist[g_n3[i]]=1;
				if(neig!=-1)changelist[neig]=1;
				if(g_n1[neig]!=-1)changelist[g_n1[neig]]=1;
				if(g_n2[neig]!=-1)changelist[g_n2[neig]]=1;
				if(g_n3[neig]!=-1)changelist[g_n3[neig]]=1;
			}
		}
		if(changeflag==0)break;
	}
	
	cerr << "num triangles to add = " << ntri << endl;
	if ((tempTopoHdl = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))) == 0)
		goto errRecovery;	// declared in System.c

	for(i = 0; i < ntri; i ++)
	{
		(*tempTopoHdl)[i].vertex1 = g_v1[i];
		(*tempTopoHdl)[i].vertex2 = g_v2[i];
		(*tempTopoHdl)[i].vertex3 = g_v3[i];
		(*tempTopoHdl)[i].adjTri1 = g_n1[i];
		(*tempTopoHdl)[i].adjTri2 = g_n2[i];
		(*tempTopoHdl)[i].adjTri3 = g_n3[i];
	}
	//InitBooleanHandle(ntri,&gTriSelected);	// defined in CatUtils.c, declared in System.c
	err = false;
	*topoHdl = tempTopoHdl;

errRecovery:
	if (err==memFullErr)
	//if(memerr)
	{
		printError("Not enough memory to generate triangles.");
	}
	//if(gTrival)_DisposePtr((Ptr)gTrival);
	//gTrival = 0;
	//if(changelist)_DisposePtr((Ptr)changelist);
	//if(p)_DisposePtr((Ptr)p);
	//if(l)_DisposePtr((Ptr)l);
	//if(g_v1) _DisposePtr((Ptr)g_v1); 
	//g_v1 = 0;
	//if(g_v2)_DisposePtr((Ptr)g_v2); 
	//g_v2 = 0;
	//if(g_v3)_DisposePtr((Ptr)g_v3);
	//g_v3 = 0;
	//if(g_n1)_DisposePtr((Ptr)g_n1);
	//g_n1 = 0;
	//if(g_n2)_DisposePtr((Ptr)g_n2);
	//g_n2 = 0;
	//if(g_n3)_DisposePtr((Ptr)g_n3);
	//g_n3 = 0;
	//if(x)_DisposePtr((Ptr)x);
	//if(y)_DisposePtr((Ptr)y);

	if(gTrival) {free(gTrival); gTrival = NULL;}
	if(changelist) {free(changelist); changelist = NULL;}
	if(p) {free(p); p = NULL;}
	if(l) {free(l); l = NULL;}
	if(g_v1) {free(g_v1); g_v1 = NULL;}
	if(g_v2) {free(g_v2); g_v2 = NULL;}
	if(g_v3) {free(g_v3); g_v3 = NULL;}
	if(g_n1) {free(g_n1); g_n1 = NULL;}
	if(g_n2) {free(g_n2); g_n2 = NULL;}
	if(g_n3) {free(g_n3); g_n3 = NULL;}
	if(x) {free(x); x = NULL;}
	if(y) {free(y); y = NULL;}

	if (err)
		if(tempTopoHdl) {DisposeHandle((Handle)tempTopoHdl); tempTopoHdl = 0;}
	return  err;
}

Boolean maketriangles2(TopologyHdl *topoHdl, LongPointHdl ptsH, long nv, LONGH boundarySegs, long nbounds, LONGH ptrVerdatToNetCDFH, long numCols_ext) 
{
	long triscale = 2;
	char from[20],to[40],msg[256];
	//Boolean memerr = true;
	Boolean userCancel = false, morepoints;
	short err = 1; // AH 03/19/2012
	long i,j,k, nowlines,pt=0,flag1,flag2;
	//short i,j,k, nowlines,pt=0,flag1,flag2;	// if more than 32768 triangles crashes out
	long *x=0,*y=0;
	double s1,s2,s3;
	long x1,y1,x2,y2,x3,y3,length,side,xia,xib,yia,yib,dxi,dyi,xj,yj,xja,yja,xjb,yjb;
	long *p=0,ntri;
	Boolean linesCross;
	short neig, nlines=0,segstart=0, segCount = 0,changeflag, *changelist=0;
	//long nv = GetNumVertices();	// defined in Topology.c
	long nzerobytes = nv * sizeof(long);
	MySegment	*l=0;	
	//Rect r=MapDrawingRect();
	TopologyHdl tempTopoHdl=0;
	//long nbounds = GetNumBoundaries();	// defined in Topology.c
	/*if(!(p =(long *) _NewPtrClear(sizeof(long)*nv)))goto errRecovery;
	if(!(l = (MySegment *)_NewPtr(8*nv *sizeof(MySegment))))goto errRecovery;
	if(!(g_v1 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;	
	if(!(g_v2 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_v3 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n1 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n2 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(g_n3 = (long *)_NewPtrClear(sizeof(long) * triscale * nv)))goto errRecovery;
	if(!(x = (long *)_NewPtrClear(sizeof(long) * nv)))goto errRecovery;
	if(!(y = (long *)_NewPtrClear(sizeof(long) * nv)))goto errRecovery;*/
	
	p = (long *)calloc(nv,sizeof(long));
	if (p==NULL) {err = memFullErr; goto errRecovery;}
	l = (MySegment *)calloc(8*nv,sizeof(MySegment));
	if (l==NULL) {err = memFullErr; goto errRecovery;}
	g_v1 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v1==NULL) {err = memFullErr; goto errRecovery;}
	g_v2 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v2==NULL) {err = memFullErr; goto errRecovery;}
	g_v3 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_v3==NULL) {err = memFullErr; goto errRecovery;}
	g_n1 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n1==NULL) {err = memFullErr; goto errRecovery;}
	g_n2 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n2==NULL) {err = memFullErr; goto errRecovery;}
	g_n3 = (long *)calloc(triscale * nv,sizeof(long));
	if (g_n3==NULL) {err = memFullErr; goto errRecovery;}
	x = (long *)calloc(nv,sizeof(long));
	if (x==NULL) {err = memFullErr; goto errRecovery;}
	y = (long *)calloc(nv,sizeof(long));
	if (y==NULL) {err = memFullErr; goto errRecovery;}
	
	//memerr = false;

	//PenNormal();
	for(;;)
	{
		if(segCount==nbounds)break;
		l[nlines].a=pt;
		l[nlines].flg=-1;
		//if(pt== (*gSegs)[segCount]){
		if(pt == (*boundarySegs)[segCount]){
			segCount++;
			l[nlines].b=segstart;
			segstart=pt+1;
		}
		else{
			l[nlines].b=pt+1;
		}		
		pt++;
		nlines++;
	} 
	//InitCoordinates1(x,y,ptsH,nv); 	
	if ((err = InitCoordinates(x, y, ptsH, nv, 1)) != 0) goto errRecovery;

	/* enter triangle generation loop */
	// make sure triangles don't have 3 points in the same array row/col in netcdf curvilinear case
	ntri=-1;
	for(;;){
		if(nlines<=0)break;
		nowlines=nlines;
		for(i=0;i<nowlines;i++)
		{
			long vertex1, vertex2, vertex3, iIndex[3], jIndex[3];
			if(l[i].flg==-2)continue;
			xia = x[l[i].a]; yia = y[l[i].a];
			xib = x[l[i].b]; yib = y[l[i].b];
			dxi = xib - xia; dyi = yib - yia;
			memset(p,1,nzerobytes); //equivalent to for(j=0;j<nv;j++)p[j]=1;
			p[l[i].a]=0; p[l[i].b]=0;
					vertex1 = INDEXH(ptrVerdatToNetCDFH,l[i].a);
					vertex2 = INDEXH(ptrVerdatToNetCDFH,l[i].b);
					iIndex[0] = vertex1/numCols_ext;
					jIndex[0] = vertex1%numCols_ext;
					iIndex[1] = vertex2/numCols_ext;
					jIndex[1] = vertex2%numCols_ext;

			for(;;)
			{
				length=10000000;
				morepoints = false;
				pt = -1;
				for(j=0; j<nv; j++)
				{
					if(CmdPeriod())
					{
						userCancel= true;
						goto errRecovery;
					}
					xj = x[j] ; yj = y[j];
					vertex3 = INDEXH(ptrVerdatToNetCDFH,j);
					iIndex[2] = vertex3/numCols_ext;
					jIndex[2] = vertex3%numCols_ext;
					if ((iIndex[0]==iIndex[1] && iIndex[1]==iIndex[2]) || (jIndex[0]==jIndex[1] && jIndex[1]==jIndex[2])) continue;
					if(dxi*(yj-yia) > dyi*(xj-xia) && p[j] && 
								(side=HYPOT(xj-xia,yj-yia) + HYPOT(xj-xib,yj-yib)) < length)	
					{
								length=side;
								pt=j;
					}
					morepoints = morepoints || p[j] != 0;
				}
				
				if(pt == -1)
				{
					err = true;
					if(morepoints) 	// something is wrong
					{
						sprintf(from,"%ld",l[i].a);
						sprintf(to,"%ld",l[i].b);
						strcpy(msg,"Problem at segment with endpoints [");
						strcat(msg,from); strcat(msg," ,"); strcat(msg,to);strcat(msg,"]");
						strcat(msg," There may be a problem with boundary orientation.");
						printError(msg);	
						SysBeep(5);
					}
					else printError("Could not generate triangles.");
					goto errRecovery;
				}
				
				linesCross=false;
				x2=x[pt];
				y2=y[pt];
				for(j=0;j<nlines;j++)
				{
					xja = x[l[j].a]; yja = y[l[j].a];
					xjb = x[l[j].b]; yjb = y[l[j].b];
					if(CROSS(xia,yia,x2,y2,xja,yja,xjb,yjb) || CROSS(xib,yib,x2,y2,xja,yja,xjb,yjb))	
					{
						linesCross=true;
						p[pt]=0;
						break;
					}
				}				
				if(linesCross)continue;
				break;
			}
			/* add triangle to triangle list */
			
			ntri=ntri+1;
			if(ntri > triscale*nv)
			{
				printError("Problem generating triangles. There may be a problem with "
				"boundary orientation.");
				goto errRecovery;
			}
			
			g_v1[ntri]=l[i].a;
			g_v2[ntri]=l[i].b;
			g_v3[ntri]=pt;
			k=l[i].flg;
			g_n3[ntri]=k;
			if(k>=0){
				if((g_v1[k]!=l[i].a)&&(g_v1[k]!=l[i].b))g_n1[k]=ntri;
				if((g_v2[k]!=l[i].a)&&(g_v2[k]!=l[i].b))g_n2[k]=ntri;
			}
			
			/* check if lines are on list already */
			
			flag1=-1;
			flag2=-1;
			for(k=0;k<nlines;k++){
				if(((l[k].a==l[i].a)&&(l[k].b==pt))||
					 ((l[k].b==l[i].a)&&(l[k].a==pt)))flag1=k;
				if(((l[k].a==l[i].b)&&(l[k].b==pt))||
					 ((l[k].b==l[i].b)&&(l[k].a==pt)))flag2=k;
			}
			
			/* add first new line segment */
			
			if(flag1==-1){
				l[nlines].a=l[i].a;
				l[nlines].b=pt;
				l[nlines].flg=ntri;	
				nlines=nlines+1;
			}
			else{
			  k=l[flag1].flg;
				g_n2[ntri]=k;
				if(k>=0){
					if((g_v1[k]!=l[i].a)&&(g_v1[k]!=pt))g_n1[k]=ntri;
					if((g_v2[k]!=l[i].a)&&(g_v2[k]!=pt))g_n2[k]=ntri;
					if((g_v3[k]!=l[i].a)&&(g_v3[k]!=pt))g_n3[k]=ntri;
				}
				l[flag1].flg=-2;
			}
			
			/* add second new line segment */
			
			if(flag2==-1){
				l[nlines].a=pt;
				l[nlines].b=l[i].b;
				l[nlines].flg=ntri;
				nlines=nlines+1;
			}
			else{
				k=l[flag2].flg;
				g_n1[ntri]=k;
				if(k>=0){
					if((g_v1[k]!=pt)&&(g_v1[k]!=l[i].b))g_n1[k]=ntri;
					if((g_v2[k]!=pt)&&(g_v2[k]!=l[i].b))g_n2[k]=ntri;
					if((g_v3[k]!=pt)&&(g_v3[k]!=l[i].b))g_n3[k]=ntri;
				}
				l[flag2].flg=-2;
			}
			
			/* remove used line segment*/ 
			
			l[i].flg=-2;
		}
		
		/* compress line list */
		
		j=0;
		for(i=0;i<nlines;i++){
			if(l[i].flg==-2){
				j++;
			}
			else{
				l[i-j]=l[i];
			}
		}	
		nlines -= j;
	}
	
	ntri++;
	
	
	/* calculate goodness of triangles */
	//float gTrival[TRIANGLES];
	//short changeflag,changelist[TRIANGLES];

	//if(!(gTrival = (double *)_NewPtrClear(sizeof(double)*ntri)))goto errRecovery;	
	//if(!(changelist = (short *)_NewPtrClear(sizeof(short) *ntri)))goto errRecovery;
	gTrival = (double *)calloc(ntri,sizeof(double));
	if (gTrival==NULL) {err = memFullErr; goto errRecovery;}
	changelist = (short *)calloc(ntri,sizeof(short));
	if (changelist==NULL) {err = memFullErr; goto errRecovery;}

	//InitCoordinates2(x,y,ptsH,nv); 	
	if ((err = InitCoordinates(x, y, ptsH, nv, 2)) != 0) goto errRecovery;

	for(i=0;i<ntri;i++){
		x1= x[g_v1[i]];
		y1= y[g_v1[i]];
		x2= x[g_v2[i]];
		y2= y[g_v2[i]];
		x3= x[g_v3[i]];
		y3= y[g_v3[i]];
		gTrival[i]=GOOD(x1,y1,x2,y2,x3,y3);
		changelist[i]=1;
	}
	
	changeflag=0;
	for(j=0;j<10;j++){
		for(i=0;i< ntri;i++){
			if(changelist[i]==0)continue;
			
			
			x1= x[g_v1[i]];
			y1= y[g_v1[i]];
			x2= x[g_v2[i]];
			y2= y[g_v2[i]];
			x3= x[g_v3[i]];
			y3= y[g_v3[i]];
			
			s1 = labs(x2-x3) + labs(y2-y3);
			s2 = labs(x3-x1) + labs(y3 - y1);
			s3 = labs(x1-x2) + labs(y1-y2);
			if(s1>s2 && s1>s3)
			{
				side=s1;
			}
			else
			{
				if(s2>s3)
				{
					side=s2;
				}
				else
				{
					side=s3;
				}
			}
			if(side==s1)neig=g_n1[i];
			if(side==s2)neig=g_n2[i];
			if(side==s3)neig=g_n3[i];
			if(!FIX(i,neig,x,y))		
			{
				changelist[i]=0;
				changeflag=changeflag+1;
			}
			else
			{
				if(i!=-1)changelist[i]=1;
				if(g_n1[i]!=-1)changelist[g_n1[i]]=1;
				if(g_n2[i]!=-1)changelist[g_n2[i]]=1;
				if(g_n3[i]!=-1)changelist[g_n3[i]]=1;
				if(neig!=-1)changelist[neig]=1;
				if(g_n1[neig]!=-1)changelist[g_n1[neig]]=1;
				if(g_n2[neig]!=-1)changelist[g_n2[neig]]=1;
				if(g_n3[neig]!=-1)changelist[g_n3[neig]]=1;
			}
		}
		if(changeflag==0)break;
	}
	
	if(!(tempTopoHdl = (TopologyHdl)_NewHandleClear(ntri * sizeof(Topology))))goto errRecovery;	// declared in System.c

	for(i = 0; i < ntri; i ++)
	{
		(*tempTopoHdl)[i].vertex1 = g_v1[i];
		(*tempTopoHdl)[i].vertex2 = g_v2[i];
		(*tempTopoHdl)[i].vertex3 = g_v3[i];
		(*tempTopoHdl)[i].adjTri1 = g_n1[i];
		(*tempTopoHdl)[i].adjTri2 = g_n2[i];
		(*tempTopoHdl)[i].adjTri3 = g_n3[i];
	}
	//InitBooleanHandle(ntri,&gTriSelected);	// defined in CatUtils.c, declared in System.c
	err = false;
	*topoHdl = tempTopoHdl;

errRecovery:
	if(err==memFullErr)
	//if(memerr)
	{
		printError("Not enough memory to generate triangles.");
	}
	/*if(gTrival)_DisposePtr((Ptr)gTrival);
	gTrival = 0;
	if(changelist)_DisposePtr((Ptr)changelist);
	if(p)_DisposePtr((Ptr)p);
	if(l)_DisposePtr((Ptr)l);
	if(g_v1) _DisposePtr((Ptr)g_v1); 
	g_v1 = 0;
	if(g_v2)_DisposePtr((Ptr)g_v2); 
	g_v2 = 0;
	if(g_v3)_DisposePtr((Ptr)g_v3);
	g_v3 = 0;
	if(g_n1)_DisposePtr((Ptr)g_n1);
	g_n1 = 0;
	if(g_n2)_DisposePtr((Ptr)g_n2);
	g_n2 = 0;
	if(g_n3)_DisposePtr((Ptr)g_n3);
	g_n3 = 0;
	if(x)_DisposePtr((Ptr)x);
	if(y)_DisposePtr((Ptr)y);*/

	if(gTrival) {free(gTrival); gTrival = NULL;}
	if(changelist) {free(changelist); changelist = NULL;}
	if(p) {free(p); p = NULL;}
	if(l) {free(l); l = NULL;}
	if(g_v1) {free(g_v1); g_v1 = NULL;}
	if(g_v2) {free(g_v2); g_v2 = NULL;}
	if(g_v3) {free(g_v3); g_v3 = NULL;}
	if(g_n1) {free(g_n1); g_n1 = NULL;}
	if(g_n2) {free(g_n2); g_n2 = NULL;}
	if(g_n3) {free(g_n3); g_n3 = NULL;}
	if(x) {free(x); x = NULL;}
	if(y) {free(y); y = NULL;}


	if (err)
		if(tempTopoHdl) {DisposeHandle((Handle)tempTopoHdl); tempTopoHdl = 0;}
	return  err;
}
