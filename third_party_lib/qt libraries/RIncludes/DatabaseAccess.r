/*
     File:       DatabaseAccess.r
 
     Contains:   Database Access Manager Interfaces.
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1989-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __DATABASEACCESS_R__
#define __DATABASEACCESS_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#if CALL_NOT_IN_CARBON

/* types for the Database Access Manager */

/* 'wstr' - Word Length String Resource */
type 'wstr' {
        wstring;                                                /* string with word length spec. */
};

/* 'qrsc' - Query Resource */
type 'qrsc' {
        integer;                                                /* version */

        integer;                                                /* 'qdef' ID */

        integer;                                                /* STR# ID for ddevName, host,
                                                                     user, password, connstr */

        integer;                                                /* current query */

        /* array of IDs of 'wstr' resources containing queries */
        integer = $$CountOf(QueryArray);                        /* array size */
        wide array QueryArray {
            integer;                                            /* id of 'wstr' resource */
        };

        /* array of types and IDs for other resources for query */
        integer = $$CountOf(ResArray);                          /* array size */
        wide array ResArray {
            literal longint;                                    /* resource type */
            integer;                                            /* resource ID */
        };
};

/* 'dflg' - ddev Flags */
type 'dflg' {
        longint;                                                /* version */

        unsigned bitstring[32]                                  /* ddev flags */
            asyncNotSupp, asyncSupp;
};
#endif  /* CALL_NOT_IN_CARBON */


#endif /* __DATABASEACCESS_R__ */

