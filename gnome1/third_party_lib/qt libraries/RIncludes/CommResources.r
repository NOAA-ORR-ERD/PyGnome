/*
     File:       CommResources.r
 
     Contains:   Communications Toolbox Resource Manager Interfaces.
 
     Version:    Technology: System 7.5
                 Release:    QuickTime 6.0.2
 
     Copyright:  © 1988-2001 by Apple Computer, Inc., all rights reserved
 
     Bugs?:      For bug reports, consult the following page on
                 the World Wide Web:
 
                     http://developer.apple.com/bugreporter/
 
*/

#ifndef __COMMRESOURCES_R__
#define __COMMRESOURCES_R__

#ifndef __CONDITIONALMACROS_R__
#include "ConditionalMacros.r"
#endif

#if CALL_NOT_IN_CARBON

/*----------------------------tbnd ¥ Tool resource bundle list ------------------*/
type 'cbnd' {
        integer = $$CountOf(TypeArray) - 1;
        array TypeArray {
                literal longint;                                /* Type                 */
                integer = $$CountOf(IDArray) - 1;
                wide array IDArray {
                        integer;                                /* Local ID             */
                        integer;                                /* Actual ID            */
                };
        };
};

/*----------------------------fbnd ¥ Tool resource bundle list ------------------*/
type 'fbnd' as 'cbnd';

/*----------------------------cbnd ¥ Tool resource bundle list ------------------*/
type 'tbnd' as 'cbnd';


/*----------------------------flst ¥ Font Family List----------------------------*/
type 'flst' {
        integer = $$CountOf(Fonts);                             /* # of fonts           */
        array Fonts {
            pstring;                                            /*      Font NAME       */
            align word;
            unsigned hex integer    plain;                      /*      Font Style      */
            integer;                                            /*      Font Size       */
            integer;                                            /*      Font Mode       */
        };
};


/*----------------------------caps ¥ Connection tool capabilities list-----------*/

/* Define flags for "Channels" field of 'caps' resource */
#define cmData              (1 << 0)
#define cmCntl              (1 << 1)
#define cmAttn              (1 << 2)

#define cmDataNoTimeout     (1 << 4)
#define cmCntlNoTimeout     (1 << 5)
#define cmAttnNoTimeout     (1 << 6)

#define cmDataClean         (1 << 8)
#define cmCntlClean         (1 << 9)
#define cmAttnClean         (1 << 10)

//  for end of message field of caps resource
#define cmFlagsEOM          (1 << 0);

/* Connection tool capabilities resource */
type 'caps' {
    integer = $$CountOf (PairsArray);
    
    array PairsArray {

        switch {
            case Abort:
                key literal longint     = 'ABRT';
                literal longint
                    supported       = 1,
                    notSupported    = 0;
    
            case AppleTalkBased:
                key literal longint     = 'ATLK';
                longint appletalkBased      =   1,
                        notAppletalkBased   =   0;
                        
            case Break:
                key literal longint     = 'BRK ';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported        = 0;
                    
            case Channels:
                key literal longint     = 'CHAN';
                hex longint;
                    
            case Close:
                key literal longint     = 'CLOS';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported        = 0;
                    
            case EndOfMessage:
                key literal longint     = 'EOM ';
                hex longint;
                
            case Kill:
                key literal longint     = 'KILL';
                literal longint
                    supported       = 1,
                    notSupported    = 0;
                    
            case Listen:
                key literal longint     = 'LSTN';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported        = 0;
                    
            case LowLevelIO:
                key literal longint     = 'LLIO';
                literal longint
                    supported       = 1,
                    notSupported    = 0;
                    
            case MinimumMemory:
                key literal longint     = 'MEMM';
                hex longint;
                    
            case Open:
                key literal longint     = 'OPEN';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported        = 0;
                    
            case Order:
                key literal longint     = 'ORDR';
                longint guaranteed      = 1, 
                        notGuaranteed   = 0;
                
            case Protocol:
                key literal longint     = 'PROT';
                literal longint
                    ISDN    = 'ISDN',
                    TCP     = 'TCP ',
                    ADSP    = 'ADSP',
                    NSP     = 'NSP ',
                    LAT     = 'LAT ',
                    NSPg    = 'NSPg',
                    LATg    = 'LATg',
                    Serial  = 'SERD',
                    Modem   = 'MODM',
                    MacPAD  = 'PAD ';
                    
            case Read:
                key literal longint     = 'READ';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported        = 0;
                    
            case RecommendedMemory:
                key literal longint     = 'MEMR';
                hex longint;
                    
            case Reliability:
                key literal longint     = 'RELY';
                longint guaranteed      = 1, 
                        notGuaranteed   = 0;
                
            case Service:
                key literal longint     = 'SERV';
                literal longint
                    Datagram            = 'DGRM',
                    Stream              = 'STRM';
            
            case Timeout:
                key literal longint     = 'TOUT';
                literal longint
                    supported       = 1,
                    notSupported    = 0;
                    
            case Write:
                key literal longint     = 'WRIT';
                literal longint
                    synchronousOnly     = 'SYNC',
                    asynchronousOnly    = 'ASYN',
                    both                = 'BOTH',
                    notSupported = 0;
                    
            case XTI:                               /* reserved for Apple for nowÉ */
                key literal longint     = 'XTI ';
                literal longint
                    notSupported = 0;
                    
        };
    };
};

/*----------------------------faps ¥ File Transfer tool capabilities list-----------*/
type 'faps' {
    integer = $$CountOf (PairsArray);
    
    array PairsArray {
        switch {
            case BatchTransfers:                    /* i.e. support for FTSend/FTReceive */
                key literal longint     = 'BXFR';
                literal longint
                    supported           =   1,
                    notSupported        =   0;
    
            case FileTypesSupported:                /* types of files that can be transferred */
                key literal longint     = 'TYPE';
                literal longint
                    textOnly            =   'TEXT',
                    allTypes            =   '????';
                
            case TransferModes:                     /* send/receive or both */
                key literal longint     = 'MODE';
                literal longint
                    sendOnly            =   'SEND',
                    receiveOnly         =   'RECV',
                    sendAndReceive      =   'BOTH',
                    sendAndReceiveAsync =   'ASYN',
                    notSupported        =   0;
        };
    };
};

/*----------------------------taps ¥ Terminal tool capabilities list-----------*/
type 'taps' {
    integer = $$CountOf (PairsArray);
    
    array PairsArray {
        switch {
            case TerminalSearching:                 /* i.e. support for TMAddSearch */
                key literal longint     = 'SRCH';
                literal longint
                    supported           =   1,
                    notSupported        =   0;
    
            case TerminalType:              /* types of files that can be transferred */
                key literal longint     = 'TYPE';
                literal longint
                    graphicsTerminal    =   'GRFX',
                    textTerminal        =   'TEXT',
                    both                =   'BOTH';
                
        };
    };
};
#endif  /* CALL_NOT_IN_CARBON */


#endif /* __COMMRESOURCES_R__ */

