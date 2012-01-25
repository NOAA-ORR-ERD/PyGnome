
#ifndef __TIMEUTILS__
#define __TIMEUTILS__

Seconds RetrieveTime(DialogPtr dialog, short monthItem);
Seconds RetrievePopTime(DialogPtr dialog, short monthItem,OSErr * err);
void DisplayTime(DialogPtr dialog, short monthItem, Seconds seconds);

#endif
