

#ifdef MAC

// this file was modified from a sample code file  from Apple
// to support SA_MENU in OS X


#include "CROSS.h"


/*
	This sample code demonstrates how to write an Appearance-savvy, Mac OS X-compatible MDEF.
	It imitates the appearance the standard system MDEF as far as possible.
	
	Features of the standard MDEF that aren't implemented here:
	
		kMenuDrawItemsMsg
		icons
		glyph-based command keys
		custom menu font
		custom menu item text encoding, font, and style
		right-to-left drawing
		pencil glyph substitution
	
	This code has not yet been compiled or run on Mac OS 9. It should be fairly close to
	implementing an Appearance-savvy MDEF for 9, but will require at least CarbonLib 1.3.1
	for DrawThemeTextBox.
	
	There are several tricky platform-specific and OS-version-specific issues to deal with
	in writing an Appearance-savvy MDEF. The major ones:
	
		-	Because menus use a window buffer that contains an alpha channel, drawing into the
			menu must be done with CoreGraphics and not Quickdraw if you would like to have
			a transparent menu. Quickdraw does not understand alpha channels and will simply
			set the alpha channel component of each pixel to 0xFF (making the menu opaque).
			If you are drawing with the Appearance Manager menu-drawing APIs, you will automatically
			use CoreGraphics and your menus will be transparent.
		
		-	Because Quickdraw does not deal with alpha channels, you cannot use Quickdraw to
			scroll a menu's contents without making the menu opaque. Mac OS X 10.1 and later, and
			CarbonLib 1.5 and later, provide a new ScrollMenuImage API that scrolls the menu
			using CoreGraphics. On Mac OS X 10.0.x, there is no good substitute for ScrollMenuImage;
			this sample code just uses ScrollRect.
			
		-	Because menus are partially transparent on Mac OS X, when the Appearance Manager
			draws the menu background inside DrawThemeMenuBackground, it composites the menu
			background together with whatever was previously visible in the menu's window
			buffer. If the menu background is being redrawn because a previously selected menu
			item is being redrawn in an unhilited state, then the previous content of the
			menu's buffer is the blue hilited menu item, and a small fraction of the blue hilite
			will show through the background of the unhilited menu item. The solution to this
			is to erase the area where the menu background will be drawn before drawing the
			background. Mac OS X 10.1 and later, and CarbonLib 1.5 and later, will offer a new
			API to do this, EraseMenuBackground; on Mac OS X 10.0.x, you must use CoreGraphics
			directly.
			
		-	The GetThemeMetric API in Mac OS X 10.1 and later, and in CarbonLib 1.5 and later, has
			several new theme metrics to aid in the layout of a menu. These metrics are not available
			in 10.0.x and must be hard-coded on earlier systems.
			
		-	The check, diamond, bullet, and dash characters, when drawn as mark characters for a
			menu item by the Appearance Manager, are drawn with special customized glyphs that are
			different from the glyphs in the system font. On Mac OS 10.0.x, the public Appearance
			Manager APIs do not provide any way to get these special glyphs. On Mac OS X 10.1 and
			later, the DrawThemeTextBox API has been modified to provide these glyphs automatically,
			so an MDEF should simply always use DrawThemeTextBox to draw the mark characters.
			
		-	When drawing the command, option, control, and shift modifier glyphs in a menu item,
			for compatibility with both Mac OS X 10.0.x and 10.1, you must create the CFString
			containing the glyph characters using CFStringCreateWithBytes, providing a source
			string containing the kMenuCommand/Option/Control/Glyph character codes from Menus.h
			and using the kTextEncodingMacKeyboardGlyphs encoding. If you do not need compatibility
			with 10.0.x, and your MDEF will only run on later releases of Mac OS X, then you can
			create the CFString by specifying the Unicode characters for these glyphs directly
			(these Unicode characters are listed in Events.h). 
			
		-	The following theme fonts are used when drawing a menu:
			
				kThemeMenuItemFont			used for: menu item text, menu command key
				kThemeMenuItemMarkFont		used for: menu mark character
				kThemeMenuItemCmdKeyFont	used for: command key modifier glyphs, menu
											command key if specified by glyph instead of
											character code, menu mark if the mark's character
											code is less than 32 (ASCII space)
											
		-	DrawThemeTextBox uses the ThemeDrawState constant to modify the shadow and boldness
			of the text; it does not, however, modify the text color based on the draw state.
			It assumes that you have already set up the text color appropriately using 
			SetThemeTextColor. However, there is currently no API for setting the text color
			of a CoreGraphics context based on ThemeDrawState. For this reason, DrawThemeTextBox
			makes an exception to its no-text-color-setup rule; if you pass NULL for the context,
			the text color will be set up appropriately for you. This sample code therefore always
			passes NULL as the CGContextRef parameter to DrawThemeTextBox, even though a context
			is passed to the MDEF by the Menu Manager.
*/


//
// determines whether the MDEF is compatible with Mac OS X 10.0.x (set to zero)
// or only with Mac OS X 10.1 and later (set to one)
//
#define AFTER_MACOSX_10_0_x 0


/*==================================================================================================*/
/*	¥ÊConstants																						*/
/*==================================================================================================*/

// for our wrapper around GetThemeMetric
enum
{
	kThemeMenuMetricMarkColumnWidth,
	kThemeMenuMetricExcludedMarkColumnWidth,
	kThemeMenuMetricMarkIndent,
	kThemeMenuMetricTextLeadingEdgeMargin,
	kThemeMenuMetricTextTrailingEdgeMargin,
	kThemeMenuMetricIndentWidth,
	kThemeMenuMetricIconTrailingEdgeMargin
};
typedef int ThemeMenuMetric;


/*==================================================================================================*/
/*	¥ÊTypes																							*/
/*==================================================================================================*/

// for our wrapper glue around EraseMenuBackground, ScrollMenuImage, and CGContextClearRect
//typedef OSStatus 
typedef void 
(*EraseMenuBackgroundProc)(
  MenuRef        inMenu,
  const Rect *   inEraseRect,
  CGContextRef   inContext);

//typedef OSStatus 
typedef void 
(*ScrollMenuImageProc)(
  MenuRef        inMenu,
  const Rect *   inScrollRect,
  int            inHScroll,
  int            inVScroll,
  CGContextRef   inContext);

typedef void
(*CGContextClearRectProc)(
  CGContextRef   context,
  CGRect         rect);


// useful metrics for menu drawing
typedef struct
{
	SInt16		extraWidth;				// extra width for each item
	SInt16		extraHeightPlain;		// extra height for plain text items
	SInt16		extraHeightIcon;		// extra height for items with icons
	SInt16		markWidth;				// width of the column used to draw the mark character
	SInt16		excludedMarkWidth;		// width of the column when the mark character is excluded
	SInt16		markIndent;				// indent into the mark column where drawing starts
	SInt16		textLeadingMargin;		// margin on the leading edge of the text to where text drawing starts
	SInt16		textTrailingMargin;		// margin on the trailing edge of the text to the end of the text box
	SInt16		indentWidth;			// space allocated per indent level as set by SetMenuItemIndent
	SInt16		itemHeight;				// height of a plain text icon
	SInt16		separatorHeight;		// height of a separator
	SInt16		itemBaseline;			// distance from top of item text to baseline
	SInt16		cmdGlyphWidth;			// width of the command key symbol; used as command key width for items without command keys
	SInt16		cmdCharWidth;			// width of 'W' in the menu item font; used as width of all command chars
}
MenuMetrics;

// info needed to calculate the size of an item
typedef struct
{
	MenuMetrics			metrics;
	MenuItemDataRec		itemData;
}
MenuItemCalcInfo;

// our MenuItem drawing callback gets a pointer to this in its userData
typedef struct
{
	MenuRef				menu;
	MenuAttributes		menuAttr;
	MenuItemCalcInfo	calcInfo;
	CGContextRef		context;
	Boolean				itemSelected;
}
MenuItemDrawInfo;

/////////////////////////
// JLM
// we need to know the menu ID and item num of the item under consideration
// the easiest way to do that is to set globals whenever FetchMenuItemData or DrawItem are called  
short 	gNewSAMenu_menuID; // JLM
short 	gNewSAMenu_itemBeingDrawn; // JLM, the item number of the menu item
void NewSAMenu_SetMenuIdAndItemNum(MenuRef menu, short item)
{
	gNewSAMenu_menuID = GetMenuID(menu);
	gNewSAMenu_itemBeingDrawn = item;
}
/////////////////////////


/*==================================================================================================*/
/*	¥ Prototypes																					*/
/*==================================================================================================*/

static void					DrawMenu( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, CGContextRef context );
static void					SetupItemDrawInfo( MenuRef menu, CGContextRef context, MenuItemDrawInfo* outDrawInfo );
static void					SetupItemRect( const Rect* menuBounds, const MenuTrackingData* trackingData, Rect* outItemRect );
static void					DrawItem( MenuRef menu, MenuItemIndex item, const Rect* menuRect, const Rect* itemRect,
									  const MenuTrackingData* trackingData, Boolean eraseFirst,
									  const MenuItemDrawInfo* drawInfo, CGContextRef context );
static void					FetchMenuItemData( MenuRef menu, MenuItemIndex item, MenuItemDataRec* outItemData );
static void					ReleaseMenuItemData( const MenuItemDataRec* itemData );
//static ThemeMenuType		GetThemeMenuType( MenuRef menu );
static ThemeMenuState		GetItemState( MenuItemAttributes attr, Boolean hilite );
static ThemeMenuItemType	GetItemType( MenuRef menu, MenuItemIndex item, const MenuItemDataRec* itemData, Boolean eraseFirst );
static MenuItemDrawingUPP	GetItemDrawingProc();
static pascal_ifMac void			ItemDrawingProc( const Rect *inBounds, SInt16 inDepth, Boolean inIsColorDevice, SInt32 inUserData);
static void					DrawCharTextBox( Byte ch, TextEncoding encoding, ThemeFontID font, ThemeDrawState drawState,
											 const Rect* bounds, int baseline, int just, CGContextRef context );
static int					MeasureUnicode( const UniChar* chars, ByteCount length, ThemeFontID font );
static void					DrawUnicode( const UniChar*, ByteCount length, ThemeFontID font, ThemeDrawState drawState, const Rect* bounds, int baseline, int just, CGContextRef context );
static void					DrawScrollArrow( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, Boolean downArrow, const MenuMetrics* metrics, CGContextRef context );
static void					SizeMenu( MenuRef menu, Point maxSizes );
static void					GetMenuMetrics( MenuMetrics* outMetrics );
static UniChar				GetCommandGlyph( void );
static void					CalcItemSize( MenuRef menu, const MenuItemCalcInfo* calcInfo, int* outWidth, int* outHeight );
static int					GetCommandKeyWidth( const MenuItemCalcInfo* calcInfo );
static const UniChar*		BuildModifierString( UInt32 modifiers, ByteCount* outLength );
static void					CalcMenuPopUpRect( MenuRef menu, Rect* bounds, int mouseH, int mouseV, short* whichItem );
static void					FindMenuItem( MenuRef menu, const Rect* bounds, Point hitPt, MenuTrackingData* trackingData, CGContextRef context );
static void					AutoScroll( MenuRef menu, const Rect* bounds, Point hitPt, MenuTrackingData* trackingData, MenuItemIndex prevItemSelected, const MenuMetrics* metrics, CGContextRef context );
static void					DrawScrolledItem( MenuRef menu, MenuTrackingData* trackingData, const Rect* menuRect, const Rect* itemRect, const MenuMetrics* metrics, CGContextRef context );
static void					HiliteMenuItem( MenuRef menu, const Rect* bounds, HiliteMenuItemData* hiliteData, CGContextRef context );
static void					HiliteItem( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, int i, Boolean hilite, CGContextRef context );
static void					CalcMenuItemBounds( MenuRef menu, Rect* bounds, int i );
static void					GetItemRect( MenuRef menu, const MenuTrackingData* trackingData, const Rect* bounds, int whichItem, int hintItem, int hintBottom, const MenuItemCalcInfo* whichItemInfo, Rect* itemRect );
static int					GetThemeMenuMetric( ThemeMenuMetric metric );
static Boolean				HasNoBackground();
static Boolean				HasAqua();
static void					DoEraseMenuBackground( MenuRef menu, const Rect* rect, CGContextRef context );
static void					DoCGContextClearRect( CGContextRef context, const Rect* rect );
static void					DoScrollMenuImage( MenuRef menu, const Rect* bounds, int dh, int dv, CGContextRef context );


/*==================================================================================================*/
/*	¥ÊFunctions																						*/
/*==================================================================================================*/

/*--------------------------------------------------------------------------------------------------*/
inline Boolean
HasCommandKey( const MenuItemDataRec* itemData )
{
	// the standard MDEF also checks for cmdKeyGlyph and kMenuItemAttrUseVirtualKey here
	return itemData->cmdKey != 0;
}

/*--------------------------------------------------------------------------------------------------*/
//inline CGContextRef GetTextContext( CGContextRef context )
static CGContextRef GetTextContext( CGContextRef context )
{
	#pragma unused (context)
	//
	// There is currently no API for setting up the appropriate text color for a ThemeDrawState
	// in a CGContext. If you pass NULL to DrawThemeTextBox, it will set up the color for you,
	// so we use NULL even though ideally we should be passing the actual context.
	//
	return NULL;
}

/*--------------------------------------------------------------------------------------------------*/
pascal_ifMac void
SaMenu_MDEF( short msg, MenuRef menu, Rect* bounds, Point hitPt, short* whichItem )
{
    switch ( msg )
    {
		case kMenuInitMsg:
			*whichItem = noErr;
			break;
		
        case kMenuDisposeMsg:
            break;

        case kMenuDrawMsg:
            DrawMenu( menu, bounds, (MenuTrackingData*) whichItem, (CGContextRef) ((MDEFDrawData*) whichItem)->context );
            break;

        case kMenuSizeMsg:
            SizeMenu( menu, hitPt );
            break;

		case kMenuPopUpMsg:
			CalcMenuPopUpRect( menu, bounds, hitPt.v, hitPt.h, whichItem );
			break;

        case kMenuFindItemMsg:
            FindMenuItem( menu, bounds, hitPt, (MenuTrackingData*) whichItem, (CGContextRef) ((MDEFFindItemData*) whichItem)->context );
            break;

        case kMenuHiliteItemMsg:
            HiliteMenuItem( menu, bounds, (HiliteMenuItemData*) whichItem, (CGContextRef) ((MDEFHiliteItemData*) whichItem)->context );
            break;

		case kMenuCalcItemMsg:
			CalcMenuItemBounds( menu, bounds, *whichItem );
			break;
		
        case kMenuThemeSavvyMsg:
           	*whichItem = kThemeSavvyMenuResponse;
            break;
            
        default:
            break;
    }
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawMenu( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, CGContextRef context )
{
    int					i;
    int					cItems;
	MenuItemDrawInfo	drawInfo;
	Rect				itemRect;


	trackingData->virtualMenuBottom = 0;
    
	NormalizeThemeDrawingState();
	SetupItemDrawInfo( menu, context, &drawInfo );
	SetupItemRect( bounds, trackingData, &itemRect );
	
	
	
    cItems = CountMenuItems( menu );
    for ( i = 1; i <= cItems; i++ )
    {
		int		height;

		FetchMenuItemData( menu, i, &drawInfo.calcInfo.itemData );
		CalcItemSize( menu, &drawInfo.calcInfo, NULL, &height );
		itemRect.bottom = itemRect.top + height;
        if ( ! ( itemRect.bottom <= bounds->top || itemRect.top >= bounds->bottom ) ) {
	        DrawItem( menu, i, bounds, &itemRect, trackingData, false, &drawInfo, context );
	    }
		ReleaseMenuItemData( &drawInfo.calcInfo.itemData );
		
		trackingData->virtualMenuBottom = itemRect.bottom;
		itemRect.top = itemRect.bottom;
    }

	if ( trackingData->virtualMenuTop < bounds->top )
		DrawScrollArrow( menu, bounds, trackingData, false, &drawInfo.calcInfo.metrics, context );

	if ( trackingData->virtualMenuBottom > bounds->bottom )
		DrawScrollArrow( menu, bounds, trackingData, true, &drawInfo.calcInfo.metrics, context );
}

/*--------------------------------------------------------------------------------------------------*/
static void
SetupItemDrawInfo( MenuRef menu, CGContextRef context, MenuItemDrawInfo* outDrawInfo )
{
	memset(outDrawInfo,0,sizeof(*outDrawInfo));//JLM

	outDrawInfo->menu = menu;
	GetMenuAttributes( menu, &outDrawInfo->menuAttr );
	GetMenuMetrics( &outDrawInfo->calcInfo.metrics );
	outDrawInfo->context = context;
	outDrawInfo->itemSelected = false;
}

/*--------------------------------------------------------------------------------------------------*/
static void
SetupItemRect( const Rect* menuBounds, const MenuTrackingData* trackingData, Rect* outItemRect )
{
	outItemRect->left = menuBounds->left;
	outItemRect->right = menuBounds->right;
	outItemRect->top = trackingData->virtualMenuTop;
	outItemRect->bottom = outItemRect->top;
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawItem( MenuRef menu, MenuItemIndex item, const Rect* menuRect, const Rect* itemRect,
		  const MenuTrackingData* trackingData, Boolean eraseFirst,
		  const MenuItemDrawInfo* drawInfo, CGContextRef context )
{
	NewSAMenu_SetMenuIdAndItemNum(menu,item); // JLM 
	
	if ( eraseFirst )
		DoEraseMenuBackground( menu, itemRect, context );
	
	if ( ( drawInfo->calcInfo.itemData.attr & kMenuItemAttrSeparator ) != 0 )
	{
		DrawThemeMenuSeparator( itemRect );
	}
	else
	{
		
		DrawThemeMenuItem( menuRect, itemRect,
						   trackingData->virtualMenuTop, trackingData->virtualMenuBottom,
						   GetItemState( drawInfo->calcInfo.itemData.attr, drawInfo->itemSelected ),
						   GetItemType( menu, item, &drawInfo->calcInfo.itemData, eraseFirst ),
						   GetItemDrawingProc(), (UInt32) drawInfo );
	}
}

/*--------------------------------------------------------------------------------------------------*/
static void
FetchMenuItemData( MenuRef menu, MenuItemIndex item, MenuItemDataRec* outItemData )
{
	BlockZero( outItemData, sizeof( MenuItemDataRec ) );
	
	outItemData->whichData =  kMenuItemDataMark
							| kMenuItemDataCmdKey
							| kMenuItemDataCmdKeyModifiers
							| kMenuItemDataSubmenuID
							| kMenuItemDataSubmenuHandle
							| kMenuItemDataEnabled
							| kMenuItemDataAttributes
							| kMenuItemDataCFString
							| kMenuItemDataIndent;
					   
	CopyMenuItemData( menu, item, false, outItemData );
	
	NewSAMenu_SetMenuIdAndItemNum(menu,item); // JLM
	
}

/*--------------------------------------------------------------------------------------------------*/
static void
ReleaseMenuItemData( const MenuItemDataRec* itemData )
{
	if ( itemData->cfText != NULL )
		CFRelease( itemData->cfText );
}

#if 0	// not currently used
/*--------------------------------------------------------------------------------------------------*/
static ThemeMenuType
GetThemeMenuType( MenuRef menu )
{
	ThemeMenuType menuType;
	GetMenuType( menu, &menuType );
	
	if ( !IsMenuItemEnabled( menu, 0 ) )
		menuType |= kThemeMenuTypeInactive;
		
	return menuType;
}
#endif

/*--------------------------------------------------------------------------------------------------*/
static ThemeMenuState
GetItemState( MenuItemAttributes attr, Boolean hilite )
{
	if ( hilite )
		return kThemeMenuSelected;
	else if ( ( attr & kMenuItemAttrDisabled ) != 0 )
		return kThemeMenuDisabled;
	else
		return kThemeMenuActive;
}

/*--------------------------------------------------------------------------------------------------*/
static ThemeMenuItemType
GetItemType( MenuRef menu, MenuItemIndex item, const MenuItemDataRec* itemData, Boolean eraseFirst )
{
	ThemeMenuType		menuType;
	ThemeMenuItemType	itemType = kThemeMenuItemPlain;
	
	GetMenuType( menu, &menuType );
	
	if ( itemData->submenuHandle != NULL || itemData->submenuID != 0 )
		itemType |= kThemeMenuItemHierarchical;
		
	if ( item == 1 )
		itemType |= kThemeMenuItemAtTop;
	else if ( item == CountMenuItems( menu ) )
		itemType |= kThemeMenuItemAtBottom;
		
	if ( menuType == kThemeMenuTypeHierarchical )
		itemType |= kThemeMenuItemHierBackground;
	else if ( menuType == kThemeMenuTypePopUp )
		itemType |= kThemeMenuItemPopUpBackground;
		
	if ( !eraseFirst && HasNoBackground() )
		itemType |= kThemeMenuItemNoBackground;
		
	return itemType;
}

/*--------------------------------------------------------------------------------------------------*/
static MenuItemDrawingUPP
GetItemDrawingProc()
{
	static MenuItemDrawingUPP sDrawingProc;
	if ( sDrawingProc == NULL )
		sDrawingProc = NewMenuItemDrawingUPP( ItemDrawingProc );
	return sDrawingProc;
}

/*--------------------------------------------------------------------------------------------------*/
static pascal_ifMac void
ItemDrawingProc(const Rect *inBounds, SInt16 inDepth, Boolean inIsColorDevice, SInt32 inUserData)
{
#pragma unused( inDepth, inIsColorDevice )

	MenuItemDrawInfo*	drawInfo = (MenuItemDrawInfo*) inUserData;
	ThemeDrawState		drawState;
	Rect				bounds = *inBounds;
	Rect				boundsT;
	int					baseline = bounds.top + drawInfo->calcInfo.metrics.itemBaseline;
	
	if ( drawInfo->itemSelected )
		drawState = kThemeStatePressed;
	else if ( ( drawInfo->calcInfo.itemData.attr & kMenuItemAttrDisabled ) != 0 )
		drawState = kThemeStateInactive;
	else
		drawState = kThemeStateActive;
	
	// indent
	if ( drawInfo->calcInfo.itemData.indent > 0 )
		bounds.left += drawInfo->calcInfo.itemData.indent * drawInfo->calcInfo.metrics.indentWidth;
	
	// mark character
	if ( ( drawInfo->menuAttr & kMenuAttrExcludesMarkColumn ) == 0 )
	{
		boundsT = bounds;
		bounds.left += drawInfo->calcInfo.metrics.markWidth;
		
		if ( drawInfo->calcInfo.itemData.mark != 0 )
		{
			ThemeFontID font = kThemeMenuItemMarkFont;
			TextEncoding encoding = GetApplicationTextEncoding();
			
			if ( drawInfo->calcInfo.itemData.mark < kSpaceCharCode )
			{
				font = kThemeMenuItemCmdKeyFont;
				encoding = kTextEncodingMacKeyboardGlyphs;
			}
			
			boundsT.left += drawInfo->calcInfo.metrics.markIndent;
			boundsT.right = bounds.left;
			
			DrawCharTextBox( drawInfo->calcInfo.itemData.mark, encoding, font, drawState,
								&boundsT, baseline, teFlushDefault, drawInfo->context );
		}
	}
	
	
	// JLM
	{	///////////////////////////////////////
		short extraWidth = NewSAMenu_ExtraMenuWidth(gNewSAMenu_menuID);
		if(extraWidth > 0) {
			boundsT = bounds;
			boundsT.right = boundsT.left + extraWidth;
			NewSAMenu_DrawExtra(gNewSAMenu_menuID,gNewSAMenu_itemBeingDrawn, boundsT);
			bounds.left += extraWidth; // move over the extra space
		}
	}	//////////////////////////////////////
	
	
	// text
	if ( drawInfo->calcInfo.itemData.cfText != NULL )
	{
		boundsT = bounds;
		boundsT.left += drawInfo->calcInfo.metrics.textLeadingMargin;
		DrawThemeTextBox( drawInfo->calcInfo.itemData.cfText, kThemeMenuItemFont, drawState,
						  false, &boundsT, teFlushDefault, GetTextContext( drawInfo->context ) );
	}
	
	// command key
	if ( HasCommandKey( &drawInfo->calcInfo.itemData ) )
	{
		ByteCount		cch;
		const UniChar*	modifiers;
		
		// the command key character itself
		boundsT = bounds;
		boundsT.left = boundsT.right - drawInfo->calcInfo.metrics.cmdCharWidth;
		DrawCharTextBox( drawInfo->calcInfo.itemData.cmdKey, GetApplicationTextEncoding(), kThemeMenuItemFont,
						 drawState, &boundsT, baseline, teFlushDefault, drawInfo->context );
		
		// the modifiers
		boundsT.right = boundsT.left;
		boundsT.left = bounds.right - GetCommandKeyWidth( &drawInfo->calcInfo );
		modifiers = BuildModifierString( drawInfo->calcInfo.itemData.cmdKeyModifiers, &cch );
		DrawUnicode( modifiers, cch, kThemeMenuItemCmdKeyFont, drawState, &boundsT,
					 baseline, teFlushDefault, drawInfo->context );
	}
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawCharTextBox( Byte ch, TextEncoding encoding, ThemeFontID font, ThemeDrawState drawState,
				 const Rect* bounds, int baseline, int just, CGContextRef context )
{
	Rect		adjustedBounds = *bounds;
	CFStringRef string = CFStringCreateWithBytes( NULL, &ch, 1, encoding, false );
	
	/*
		Menu item text drawn with the .Keyboard font (used for kThemeMenuItemCmdKeyFont) won't
		always have the same ascent and baseline as text drawn with the regular menu item font,
		since the glyphs in the .Keyboard font may have a different height. Therefore, we first
		determine the baseline of the text and then adjust the bounds rect so the baseline aligns
		with the overall baseline of the menu item.
	*/
	if ( font == kThemeMenuItemCmdKeyFont )
	{
		Point 	size;
		SInt16 	cmdKeyBaseline;
		
		GetThemeTextDimensions( string, kThemeMenuItemCmdKeyFont, drawState, false, &size, &cmdKeyBaseline );
		OffsetRect( &adjustedBounds, 0, baseline - bounds->top - size.v - cmdKeyBaseline );
	}
	
	DrawThemeTextBox( string, font, drawState, false, &adjustedBounds, just, GetTextContext( context ) );
	CFRelease( string );
}

/*--------------------------------------------------------------------------------------------------*/
static int
MeasureUnicode( const UniChar* chars, ByteCount length, ThemeFontID font )
{
	CFStringRef	str = CFStringCreateWithCharacters( NULL, chars, length );
	Point		pt = {0, 0};
	SInt16		baseline;
	
	GetThemeTextDimensions( str, font, kThemeStateActive, false, &pt, &baseline );
	CFRelease( str );
	return pt.h;
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawUnicode( const UniChar* chars, ByteCount length, ThemeFontID font, ThemeDrawState drawState,
			 const Rect* bounds, int baseline, int just, CGContextRef context )
{
	Rect		adjustedBounds = *bounds;
	CFStringRef	string = CFStringCreateWithCharacters( NULL, chars, length );
	
	/*
		Menu item text drawn with the .Keyboard font (used for kThemeMenuItemCmdKeyFont) won't
		always have the same ascent and baseline as text drawn with the regular menu item font,
		since the glyphs in the .Keyboard font may have a different height. Therefore, we first
		determine the baseline of the text and then adjust the bounds rect so the baseline aligns
		with the overall baseline of the menu item.
	*/
	if ( font == kThemeMenuItemCmdKeyFont )
	{
		Point 	size;
		SInt16 	cmdKeyBaseline;
		
		GetThemeTextDimensions( string, kThemeMenuItemCmdKeyFont, drawState, false, &size, &cmdKeyBaseline );
		OffsetRect( &adjustedBounds, 0, baseline - bounds->top - size.v - cmdKeyBaseline );
	}
	
	DrawThemeTextBox( string, font, drawState, false, &adjustedBounds, just, GetTextContext( context ) );
	CFRelease( string );
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawScrollArrow( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, Boolean downArrow,
				 const MenuMetrics* metrics, CGContextRef context )
{
	Rect				itemRect	= *bounds;
	ThemeMenuState		menuState;
	ThemeMenuType		menuType;
	ThemeMenuItemType	itemType;

	if ( downArrow )
		itemRect.top = itemRect.bottom - metrics->itemHeight;
	else
		itemRect.bottom = itemRect.top + metrics->itemHeight;

	//
	// If the entire menu is inactive, disable the item so that it matches the rest of the menu.
	// The menu still scrolls, so technically the item really shouldn't be inactive, but it looks
	// wrong otherwise.
	//
	if ( IsMenuItemEnabled( menu, 0 ) )
		menuState = kThemeMenuActive;
	else
		menuState = kThemeMenuDisabled;

	if ( downArrow )
		itemType = kThemeMenuItemScrollDownArrow;
	else
		itemType = kThemeMenuItemScrollUpArrow;

	GetMenuType( menu, &menuType );

	if ( menuType == kThemeMenuTypeHierarchical )
		itemType |= kThemeMenuItemHierBackground;
	else if ( menuType == kThemeMenuTypePopUp )
		itemType |= kThemeMenuItemPopUpBackground;

	// the arrow's background will be drawn on top of whatever was there before, so erase first
	DoEraseMenuBackground( menu, &itemRect, context );
	
	DrawThemeMenuItem( bounds, &itemRect, trackingData->virtualMenuTop, trackingData->virtualMenuBottom,
					   menuState, itemType, NULL, 0 );
}

/*--------------------------------------------------------------------------------------------------*/
static void
SizeMenu( MenuRef menu, Point maxSizes )
{
    int					i;
    int					cItems;
	int					menuWidth = 0;
	int					menuHeight = 0;
	MenuItemCalcInfo	calcInfo;
	SInt32				result;
	int					maxWidth;
	int					maxHeight;
	Boolean				hasCmdKey = false;

	// determine the maximum allowed width and height of the menu
	if (	( Gestalt( gestaltMenuMgrAttr, &result ) == noErr )
		 && ( result & gestaltMenuMgrSendsMenuBoundsToDefProcMask ) != 0 )
	{
		maxWidth = maxSizes.h;
		maxHeight = maxSizes.v;
	}
	else
	{
		Rect mainDeviceRect = (*GetMainDevice())->gdRect;
		maxWidth = mainDeviceRect.right - mainDeviceRect.left;
		maxHeight = mainDeviceRect.bottom - mainDeviceRect.top - GetMBarHeight();
	}
	
	GetMenuMetrics( &calcInfo.metrics );
	
	//
	// Determine the true width and height of the menu. Note that we must examine every item,
	// even if the height of the menu has already exceeded the maximum height, because we need
	// to calculate the maximum width based on the width of every item.
	//
    cItems = CountMenuItems( menu );
    for ( i = 1; i <= cItems; i++ )
    {
		int		itemWidth;
		int		itemHeight;
		int		newHeight;

		FetchMenuItemData( menu, i, &calcInfo.itemData );
		hasCmdKey |= HasCommandKey( &calcInfo.itemData );
		CalcItemSize( menu, &calcInfo, &itemWidth, &itemHeight );
		ReleaseMenuItemData( &calcInfo.itemData );
		
		if ( itemWidth > menuWidth )
			menuWidth = itemWidth;
			
		newHeight = menuHeight + itemHeight;
		if ( newHeight <= maxHeight )
			menuHeight = newHeight;
    }
	
	//
	// CalcItemSize will add the command key width to every item, even if the item doesn't have a command key.
	// This gives a better appearance when just some items have command keys. However, if no item has a command
	// key, then we remove the command key width entirely.
	//
	if ( !hasCmdKey && menuWidth >= calcInfo.metrics.cmdGlyphWidth )
		menuWidth -= calcInfo.metrics.cmdGlyphWidth;
	
	if ( menuWidth > maxWidth )
		menuWidth = maxWidth;
	if ( menuHeight > maxHeight )
		menuHeight = maxHeight;
	
	SetMenuWidth( menu, menuWidth );
	SetMenuHeight( menu, menuHeight );
}

/*--------------------------------------------------------------------------------------------------*/
static void
CalcItemSize( MenuRef menu, const MenuItemCalcInfo* calcInfo, int* outWidth, int* outHeight )
{
	MenuAttributes		menuAttr;
	
	// initial height
	if ( outHeight != NULL )
	{
		if ( ( calcInfo->itemData.attr & kMenuItemAttrSeparator ) != 0 )
			*outHeight = calcInfo->metrics.separatorHeight;
		else
			*outHeight = calcInfo->metrics.itemHeight;
	}
	
	// initial width
	if ( outWidth == NULL )
		return;
	else
		*outWidth = 0;
	
	// indent
	if ( calcInfo->itemData.indent > 0 )
		*outWidth += calcInfo->itemData.indent * calcInfo->metrics.indentWidth;
	
	// mark character
	GetMenuAttributes( menu, &menuAttr );
	if ( ( menuAttr & kMenuAttrExcludesMarkColumn ) != 0 )
		*outWidth += calcInfo->metrics.excludedMarkWidth;
	else
		*outWidth += calcInfo->metrics.markWidth;
	

	// JLM
	{	///////////////////////////////////////
		*outWidth += NewSAMenu_ExtraMenuWidth(gNewSAMenu_menuID);
	}	//////////////////////////////////////
	

	// text
	if ( calcInfo->itemData.cfText != NULL )
	{
		Point	pt;
		SInt16	baseline;
		
		GetThemeTextDimensions( calcInfo->itemData.cfText, kThemeMenuItemFont, kThemeStateActive, false, &pt, &baseline );
		*outWidth += pt.h + calcInfo->metrics.textLeadingMargin + calcInfo->metrics.textTrailingMargin;
	}
	
	// command key and modifiers
	*outWidth += GetCommandKeyWidth( calcInfo );
	
	// theme-specified extra spacing
	*outWidth += calcInfo->metrics.extraWidth;
}

/*--------------------------------------------------------------------------------------------------*/
static int
GetCommandKeyWidth( const MenuItemCalcInfo* calcInfo )
{
	int width = 0;
	
	if ( HasCommandKey( &calcInfo->itemData ) )
	{
		width = calcInfo->metrics.cmdCharWidth;
		if ( calcInfo->itemData.cmdKeyModifiers == 0 )
		{
			width += calcInfo->metrics.cmdGlyphWidth;
		}
		else
		{
			ByteCount cch;
			const UniChar* modifiers = BuildModifierString( calcInfo->itemData.cmdKeyModifiers, &cch );
			width += MeasureUnicode( modifiers, cch, kThemeMenuItemCmdKeyFont );
		}
	}
	else
	{
		width = calcInfo->metrics.cmdGlyphWidth;
	}
	
	return width;
}

/*--------------------------------------------------------------------------------------------------*/
static const UniChar*
BuildModifierString( UInt32 modifiers, ByteCount* outLength )
{
	static UniChar	ustr[4];
	ByteCount		cch = 0;

/*
	Mac OS X 10.0.x expects the CFStringRef used when drawing with kMenuItemCmdKeyFont to contain
	characters with values in the range given in Menus.h for glyph codes in the .Keyboard font.
	Those glyph codes are actually not valid Unicode values for the associated characters; that
	Mac OS X 10.0.x requires them is a bug in that release.
	
	Mac OS X 10.1 and later expects the CFStringRef to contain characters with the proper Unicode
	values (kCommand/Option/Shift/ControlUnicode).
	
	It so happens that creating a CFStringRef with CFStringCreateWithBytes, using byte values from
	the .Keyboard font encoding, and specifying kTextEncodingMacKeyboardGlyphs, will create a CFString
	with the right Unicode values on both platforms. That is the approach used by this sample code.
	
	However, if your MDEF will only run on Mac OS X 10.1 and later, then it's somewhat more efficient
	to just create the CFString by specifying the correct Unicode values in the first place.
*/	
#if AFTER_MACOSX_10_0_x
	if ( ( modifiers & kMenuControlModifier ) != 0 )
		ustr[cch++] = kControlUnicode;
	if ( ( modifiers & kMenuOptionModifier ) != 0 )
		ustr[cch++] = kOptionUnicode;
	if ( ( modifiers & kMenuShiftModifier ) != 0 )
		ustr[cch++] = kShiftUnicode;
	if ( ( modifiers & kMenuNoCommandModifier ) == 0 )
		ustr[cch++] = kCommandUnicode;
#else
	static Byte		str[4];
	CFStringRef		cfString;
	
	if ( ( modifiers & kMenuControlModifier ) != 0 )
		str[cch++] = kMenuControlGlyph;
	if ( ( modifiers & kMenuOptionModifier ) != 0 )
		str[cch++] = kMenuOptionGlyph;
	if ( ( modifiers & kMenuShiftModifier ) != 0 )
		str[cch++] = kMenuShiftGlyph;
	if ( ( modifiers & kMenuNoCommandModifier ) == 0 )
		str[cch++] = kMenuCommandGlyph;
	
	cfString = CFStringCreateWithBytes( NULL, str, cch, kTextEncodingMacKeyboardGlyphs, false );
	check( cfString != NULL );
	check( CFStringGetLength( cfString ) <= 4 );
	CFStringGetCharacters( cfString, CFRangeMake( 0, cch ), ustr );
	CFRelease( cfString );
#endif
		
	*outLength = cch;
	return ustr;
}

/*--------------------------------------------------------------------------------------------------*/
static void
CalcMenuPopUpRect( MenuRef menu, Rect* bounds, int mouseH, int mouseV, short* whichItem )
{
	MenuTrackingData	trackingData;
	Rect				itemRect;
	MenuItemCalcInfo	calcInfo;
	Rect savedBounds = *bounds; // JLM

	GetMenuMetrics( &calcInfo.metrics );
	FetchMenuItemData( menu, *whichItem, &calcInfo.itemData );
	
	trackingData.virtualMenuTop = 0;	// GetItemRect uses only this field of the tracking data
	GetItemRect( menu, &trackingData, bounds, *whichItem, 0, 0, &calcInfo, &itemRect );	

	ReleaseMenuItemData( &calcInfo.itemData );
	
	// itemRect.left and right will be garbage now, because they're based on the bounds rect,
	// which is uninitialized. But we don't care.

	if ( IsMenuSizeInvalid( menu ) )
		CalcMenuSize( menu );

	SetRect( bounds, mouseH, mouseV - itemRect.top, mouseH + GetMenuWidth( menu ),
		mouseV + GetMenuHeight( menu ) - itemRect.top );
		
	/// JLM
	{ // we need to constrain the bounds (the sample code forgot to do this here) 
		SInt32 result;
		Rect constraintRect;
		short menuWidth;
		short menuHeight;
		short contraintHeight;

		// determine the maximum allowed width and height of the menu
		if (	( Gestalt( gestaltMenuMgrAttr, &result ) == noErr )
			 && ( result & gestaltMenuMgrSendsMenuBoundsToDefProcMask ) != 0 )
		{
			constraintRect = savedBounds;
		}
		else
		{
			Rect mainDeviceRect = (*GetMainDevice())->gdRect;
			// Apple's documentation says to use the device containing the popup point where the user clicked
			// and not the MainDevice
			// but it appears right now that gestaltMenuMgrSendsMenuBoundsToDefProc is true
			// and so this code is not being called -- JLM 5/3/06 
			constraintRect = mainDeviceRect;
			constraintRect.top = mainDeviceRect.top - GetMBarHeight();
		}
		
		menuWidth = GetMenuWidth( menu );
		menuHeight = GetMenuHeight( menu );
		
		contraintHeight = constraintRect.bottom - constraintRect.top;
		if(menuHeight > contraintHeight) menuHeight = contraintHeight;
		
		SetRect( bounds, mouseH, mouseV - itemRect.top, mouseH + menuWidth,
			mouseV + menuHeight - itemRect.top );

		if(bounds->left < constraintRect.left) {
			// the menu is sticking off the constraint to the left
			OffsetRect(bounds,constraintRect.left-bounds->left,0);
		}
			
		if(bounds->right > constraintRect.right) {
			// the menu is sticking off the constraint to the right
			OffsetRect(bounds,constraintRect.right-bounds->right,0);
		}
			
		if(bounds->top < constraintRect.top) {
			// the menu is sticking off the constraint to the top
			OffsetRect(bounds,0,constraintRect.top-bounds->top);
		}
			
		if(bounds->bottom > constraintRect.bottom) {
			// the menu is sticking off the constraint to the bottom
			OffsetRect(bounds,0,constraintRect.bottom-bounds->bottom);
		}
	}
	////////////////

	*whichItem = bounds->top;
}

/*--------------------------------------------------------------------------------------------------*/
static void
FindMenuItem( MenuRef menu, const Rect* bounds, Point hitPt, MenuTrackingData* trackingData, CGContextRef context )
{
	Rect				visibleBounds		= *bounds;
	MenuItemIndex		prevItemSelected	= trackingData->itemSelected;
	MenuItemCalcInfo	calcInfo;
	
	GetMenuMetrics( &calcInfo.metrics );
	
    trackingData->itemSelected = 0;
    trackingData->itemUnderMouse = 0;

	if ( trackingData->virtualMenuTop < bounds->top )
		visibleBounds.top += calcInfo.metrics.itemHeight;

	if ( trackingData->virtualMenuBottom > bounds->bottom )
		visibleBounds.bottom -= calcInfo.metrics.itemHeight;

	if ( PtInRect( hitPt, &visibleBounds ) )
	{
	    int		i;
	    int		cItems;
        Rect	itemRect;
	    
		SetupItemRect( bounds, trackingData, &itemRect );
	    cItems = CountMenuItems( menu );
	    for ( i = 1; i <= cItems; i++ )
	    {
			int		height;

			FetchMenuItemData( menu, i, &calcInfo.itemData );
			CalcItemSize( menu, &calcInfo, NULL, &height );
			ReleaseMenuItemData( &calcInfo.itemData );
			
			itemRect.bottom = itemRect.top + height;
	        if ( PtInRect( hitPt, &itemRect ) )
	        {
	            trackingData->itemUnderMouse = i;
	            trackingData->itemRect = itemRect;
	            if ( IsMenuItemEnabled( menu, i ) )
	                trackingData->itemSelected = i;
	            break;
	        }
			
			// prepare for next item
			itemRect.top = itemRect.bottom;
	    }
    }
    else
    {
    	AutoScroll( menu, bounds, hitPt, trackingData, prevItemSelected, &calcInfo.metrics, context );
    }
}

/*--------------------------------------------------------------------------------------------------*/
static void
AutoScroll( MenuRef menu, const Rect* bounds, Point hitPt, MenuTrackingData* trackingData,
			MenuItemIndex prevItemSelected, const MenuMetrics* metrics, CGContextRef context )
{
	enum
	{
		kSlowScrollSpeed	= 10,
		kFastScrollSpeed	= 1
	};
	
	int					scrollDist	= 0;
	Rect				arrowRect;
	Rect				itemRect;
	Rect				scrollRect	= *bounds;
	UInt32				temp;
	MenuTrackingData	deepestTrackingData;
	UInt32				delayTime	= 0;
	
	// are we in the top scrolling arrow?
	if ( trackingData->virtualMenuTop < bounds->top )
	{
		// there's a top arrow, so remove that item from the scrollRect
		scrollRect.top += metrics->itemHeight;
		
		arrowRect = *bounds;
		arrowRect.bottom = arrowRect.top + metrics->itemHeight;
		if ( PtInRect( hitPt, &arrowRect ) )
		{
			// this will be the bounds of the item that's newly exposed
			itemRect = arrowRect;
			
			// scrolling up means that bits move down on the screen
			scrollDist = metrics->itemHeight;
			
			if ( hitPt.v >= ( ( arrowRect.top + arrowRect.bottom ) / 2 ) )
				delayTime = kSlowScrollSpeed;
			else
				delayTime = kFastScrollSpeed;
		}
	}
	
	// or the bottom scrolling arrow?
	if ( trackingData->virtualMenuBottom > bounds->bottom )
	{
		// there's a bottom arrow, so remove that item from the scrollRect
		scrollRect.bottom -= metrics->itemHeight;
		
		arrowRect = *bounds;
		arrowRect.top = arrowRect.bottom - metrics->itemHeight;
		if ( PtInRect( hitPt, &arrowRect ) )
		{
			// this will be the bounds of the item that's newly exposed
			itemRect = arrowRect;
			
			// scrolling down means that bits move up on the screen
			scrollDist = -metrics->itemHeight;
			
			if ( hitPt.v <= ( ( arrowRect.top + arrowRect.bottom ) / 2 ) )
				delayTime = kSlowScrollSpeed;
			else
				delayTime = kFastScrollSpeed;
		}
	}
	
	if ( scrollDist == 0 )
		return;
	
	// don't scroll if there are other menus open above us
	verify_noerr( GetMenuTrackingData( NULL, &deepestTrackingData ) );
	if ( deepestTrackingData.menu != menu )
		return;
	
	// turn off the hilite on the previous item
	if ( prevItemSelected != 0 )
		HiliteItem( menu, bounds, trackingData, prevItemSelected, false, context );
	
	// scroll me, baby
	DoScrollMenuImage( menu, &scrollRect, 0, scrollDist, context );
	trackingData->virtualMenuTop += scrollDist;
	trackingData->virtualMenuBottom += scrollDist;
	
	// draw the newly exposed item
	OffsetRect( &itemRect, 0, scrollDist );
	DrawScrolledItem( menu, trackingData, bounds, &itemRect, metrics, context );
	
	// draw the arrows, if necessary
	if ( scrollDist < 0 && trackingData->virtualMenuTop < bounds->top )
		DrawScrollArrow( menu, bounds, trackingData, false, metrics, context );

	if ( scrollDist > 0 && trackingData->virtualMenuBottom > bounds->bottom )
		DrawScrollArrow( menu, bounds, trackingData, true, metrics, context );
	
	Delay( delayTime, &temp );
}

/*--------------------------------------------------------------------------------------------------*/
static void
DrawScrolledItem( MenuRef menu, MenuTrackingData* trackingData, const Rect* menuRect, const Rect* itemRect, const MenuMetrics* metrics, CGContextRef context )
{
	int					i;
	MenuItemDrawInfo	drawInfo;
	
	
	i = ( itemRect->top - trackingData->virtualMenuTop ) / metrics->itemHeight;		// zero-based item number
	i++;																			// one-based item number
	
	SetupItemDrawInfo( menu, context, &drawInfo );
	FetchMenuItemData( menu, i, &drawInfo.calcInfo.itemData );
	DrawItem( menu, i, menuRect, itemRect, trackingData, true, &drawInfo, context );
	ReleaseMenuItemData( &drawInfo.calcInfo.itemData );
}

/*--------------------------------------------------------------------------------------------------*/
static void
HiliteMenuItem( MenuRef menu, const Rect* bounds, HiliteMenuItemData* hiliteData, CGContextRef context )
{
    MenuTrackingData	trackingData;
	Boolean				oldFirst = false;
	Boolean				oldLast = false;
	MenuItemDrawInfo	oldDrawInfo;
	MenuItemDrawInfo	newDrawInfo;
	Rect				oldItemRect;
    Rect				newItemRect;
	

    GetMenuTrackingData( menu, &trackingData );
	SetupItemDrawInfo( menu, context, &oldDrawInfo );
	newDrawInfo = oldDrawInfo;
	
	//
	// determine whether previousItem or newItem comes first in the menu so we can get the item rect
	// for the earlier item first, and use its position as a hint to GetItemRect for the later item
	//
	if ( hiliteData->previousItem != 0 )
	{
		oldFirst = hiliteData->previousItem < hiliteData->newItem;
		oldLast = !oldFirst;
	}
	
	if ( oldFirst )
	{
		FetchMenuItemData( menu, hiliteData->previousItem, &oldDrawInfo.calcInfo.itemData );
		GetItemRect( menu, &trackingData, bounds, hiliteData->previousItem, 0, 0,
					 &oldDrawInfo.calcInfo, &oldItemRect );
	}
	
	if ( hiliteData->newItem != 0 )
	{
		FetchMenuItemData( menu, hiliteData->newItem, &newDrawInfo.calcInfo.itemData );
		GetItemRect( menu, &trackingData, bounds, hiliteData->newItem, hiliteData->previousItem,
					 oldItemRect.bottom, &newDrawInfo.calcInfo, &newItemRect );
	}
	
	if ( oldLast )
	{
		FetchMenuItemData( menu, hiliteData->previousItem, &oldDrawInfo.calcInfo.itemData );
		GetItemRect( menu, &trackingData, bounds, hiliteData->previousItem, hiliteData->newItem,
					 newItemRect.bottom, &oldDrawInfo.calcInfo, &oldItemRect );
	}
	
	if ( hiliteData->previousItem != 0 )
	{
		check( oldDrawInfo.itemSelected == false );
		DrawItem( menu, hiliteData->previousItem, bounds, &oldItemRect, &trackingData, true, &oldDrawInfo, context );
		ReleaseMenuItemData( &oldDrawInfo.calcInfo.itemData );
	}
	
	if ( hiliteData->newItem != 0 )
	{
		newDrawInfo.itemSelected = true;
		DrawItem( menu, hiliteData->newItem, bounds, &newItemRect, &trackingData, true, &newDrawInfo, context );
		ReleaseMenuItemData( &newDrawInfo.calcInfo.itemData );
	}
}

/*--------------------------------------------------------------------------------------------------*/
static void
HiliteItem( MenuRef menu, const Rect* bounds, MenuTrackingData* trackingData, int i, Boolean hilite, CGContextRef context )
{
	MenuItemDrawInfo	drawInfo;
	Rect				itemRect;

    if ( i == 0 )
        return;
    
	SetupItemDrawInfo( menu, context, &drawInfo );
	drawInfo.itemSelected = hilite;
	
	FetchMenuItemData( menu, i, &drawInfo.calcInfo.itemData );
    GetItemRect( menu, trackingData, bounds, i, 0, 0, &drawInfo.calcInfo, &itemRect );
	DrawItem( menu, i, bounds, &itemRect, trackingData, true, &drawInfo, context );
	ReleaseMenuItemData( &drawInfo.calcInfo.itemData );
}

/*--------------------------------------------------------------------------------------------------*/
static void
CalcMenuItemBounds( MenuRef menu, Rect* bounds, int i )
{
	MenuTrackingData	trackingData;
	MenuItemCalcInfo	calcInfo;
	Rect				itemRect;
	int					width;
	
	// find the top of the menu, or use zero if the menu isn't open
	if ( GetMenuTrackingData( menu, &trackingData ) != noErr )
		trackingData.virtualMenuTop = 0;
	
	GetMenuMetrics( &calcInfo.metrics );
	FetchMenuItemData( menu, i, &calcInfo.itemData );
	
	// use GetItemRect to determine the item's vertical position in the menu
	// use CalcItemSize to determine the item's actual width (GetItemRect will use the menu's width)
	GetItemRect( menu, &trackingData, bounds, i, 0, 0, &calcInfo, &itemRect );
	CalcItemSize( menu, &calcInfo, &width, NULL );
	
	*bounds = itemRect;
	bounds->right = bounds->left + width;
	
	ReleaseMenuItemData( &calcInfo.itemData );
}

/*--------------------------------------------------------------------------------------------------*/
static void
GetItemRect( MenuRef menu, const MenuTrackingData* trackingData, const Rect* bounds, int whichItem,
			 int hintItem, int hintBottom, const MenuItemCalcInfo* whichItemInfo, Rect* itemRect )
{
	int					i;
	MenuItemCalcInfo	calcInfo;
	
	itemRect->left = bounds->left;
	itemRect->right = bounds->right;
	
	calcInfo.metrics = whichItemInfo->metrics;
	
	//
	// It would be possible to modify this code to use hintBottom even for items that are past whichItem
	// (by working backwards and subtracting off the height instead of adding it), but I didn't feel like
	// adding that complication to the code at this time.
	//
	if ( hintItem >= whichItem )
		hintItem = 0;
	
	if ( hintItem == 0 )
		itemRect->bottom = trackingData->virtualMenuTop;
	else
		itemRect->bottom = hintBottom;
	
	i = hintItem + 1;
	while ( i <= whichItem )
	{
		int		height;
		
		if ( i == whichItem )
			calcInfo.itemData = whichItemInfo->itemData;
		else
			FetchMenuItemData( menu, i, &calcInfo.itemData );
		CalcItemSize( menu, &calcInfo, NULL, &height );
		if ( i != whichItem )
			ReleaseMenuItemData( &calcInfo.itemData );
		
		itemRect->top = itemRect->bottom;
		itemRect->bottom += height;
		i++;
	}
}

/*--------------------------------------------------------------------------------------------------*/
static void
GetMenuMetrics( MenuMetrics* outMetrics )
{
	UniChar		uchCmdGlyph = GetCommandGlyph();
	FontInfo	fi;
	
	GetThemeMenuSeparatorHeight( &outMetrics->separatorHeight );
	GetThemeMenuItemExtra( kThemeMenuItemPlain, &outMetrics->extraHeightPlain, &outMetrics->extraWidth );
	GetThemeMenuItemExtra( kThemeMenuItemHasIcon, &outMetrics->extraHeightIcon, &outMetrics->extraWidth );
	
	outMetrics->markWidth = GetThemeMenuMetric( kThemeMenuMetricMarkColumnWidth );
	outMetrics->excludedMarkWidth = GetThemeMenuMetric( kThemeMenuMetricExcludedMarkColumnWidth );
	outMetrics->markIndent = GetThemeMenuMetric( kThemeMenuMetricMarkIndent );
	outMetrics->textLeadingMargin = GetThemeMenuMetric( kThemeMenuMetricTextLeadingEdgeMargin );
	outMetrics->textTrailingMargin = GetThemeMenuMetric( kThemeMenuMetricTextTrailingEdgeMargin );
	outMetrics->indentWidth = GetThemeMenuMetric( kThemeMenuMetricIndentWidth );
	outMetrics->cmdGlyphWidth = MeasureUnicode( &uchCmdGlyph, 1, kThemeMenuItemCmdKeyFont );
	
	UseThemeFont( kThemeMenuItemFont, smSystemScript );
	GetFontInfo( &fi );
	outMetrics->itemHeight = fi.ascent + fi.descent + fi.leading + outMetrics->extraHeightPlain;
	outMetrics->itemBaseline = fi.ascent;
	outMetrics->cmdCharWidth = fi.widMax;
}

/*--------------------------------------------------------------------------------------------------*/
static UniChar
GetCommandGlyph( void )
{
// see prior discussion in BuildModifierString for why this is here
#if AFTER_MACOSX_10_0_x
	return kCommandUnicode;
#else
	static UniChar	uch;
	Byte			ch;
	CFStringRef		str;
	
	if ( uch == 0 )
	{
		ch = kMenuCommandGlyph;
		str = CFStringCreateWithBytes( NULL, &ch, sizeof( ch ), kTextEncodingMacKeyboardGlyphs, false );
		CFStringGetCharacters( str, CFRangeMake( 0, 1 ), &uch );
		CFRelease( str );
	}
	
	return uch;
#endif
}

/*--------------------------------------------------------------------------------------------------*/
static int
GetThemeMenuMetric( ThemeMenuMetric metric )
{
	SInt32	value;
	
	/*
		We first try to get the metric using GetThemeMetric. If that returns an error,
		we use some hardcoded constants that correspond to the values used by the Mac OS 9
		and Mac OS X system MDEFs.
		
		Unfortunately, we need to check for GetThemeMetric returning noErr but a metric
		value of zero also, because GetThemeMetric will return noErr for some out-of-bounds
		and unsupported metrics on 10.0.x.
	*/
	switch ( metric )
	{
		case kThemeMenuMetricMarkColumnWidth:
			if ( GetThemeMetric( kThemeMetricMenuMarkColumnWidth, &value ) != noErr || value == 0 )
			{
				if ( HasAqua() )
				{
					value = 21;
				}
				else
				{
					FontInfo fi;
					UseThemeFont( kThemeSystemFont, smSystemScript );
					GetFontInfo( &fi );
					value = fi.widMax;
				}
			}
			break;
			
		case kThemeMenuMetricExcludedMarkColumnWidth:
			if ( GetThemeMetric( kThemeMetricMenuExcludedMarkColumnWidth, &value ) != noErr || value == 0 )
				value = 5;
			break;
		
		case kThemeMenuMetricMarkIndent:
			if ( GetThemeMetric( kThemeMetricMenuMarkIndent, &value ) != noErr || value == 0 )
			{
				if ( HasAqua() )
					value = 5;
				else
					value = 0;
			}
			break;
		
		case kThemeMenuMetricTextLeadingEdgeMargin:
			if ( GetThemeMetric( kThemeMetricMenuTextLeadingEdgeMargin, &value ) != noErr || value == 0 )
			{
				if ( HasAqua() )
					value = 0;
				else
					value = 2;
			}
			break;
		
		case kThemeMenuMetricTextTrailingEdgeMargin:
			if ( GetThemeMetric( kThemeMetricMenuTextTrailingEdgeMargin, &value ) != noErr || value == 0 )
			{
				if ( HasAqua() )
					value = 19;
				else
					value = 8;
			}
			break;
			
		case kThemeMenuMetricIndentWidth:
			if ( GetThemeMetric( kThemeMetricMenuIndentWidth, &value ) != noErr || value == 0 )
				value = 12;
			break;
		
		case kThemeMenuMetricIconTrailingEdgeMargin:
			if ( GetThemeMetric( kThemeMetricMenuIconTrailingEdgeMargin, &value ) != noErr || value == 0 )
			{
				if ( HasAqua() )
				{
					value = 4;
				}
				else
				{
					value = CharWidth( kSpaceCharCode );
				}
			}
			break;
		
		default:
			value = 0;
	}
	
	return value;
}

/*--------------------------------------------------------------------------------------------------*/
//	Determines whether kThemeMenuItemNoBackground is available. It is only available on Mac OS X and later.
static Boolean
HasNoBackground()
{
	static Boolean	sHasNoBackground;
	static Boolean	sInited;
	
	if ( !sInited )
	{
		SInt32 result;
		Gestalt( gestaltSystemVersion, &result );
		sHasNoBackground = result >= 0x1000;
		sInited = true;
	}
	
	return sHasNoBackground;
}

/*--------------------------------------------------------------------------------------------------*/
static Boolean
HasAqua()
{
	static Boolean	sHasAqua;
	static Boolean	sInited;
	
	if ( !sInited )
	{
		Collection	c = NewCollection();
		Str255		name;
		Size		size = sizeof( name );
		
		GetTheme( c );
		sHasAqua = GetTheme( c ) == noErr
				&& GetCollectionItem( c, kThemeNameTag, 0, &size, name ) == noErr
				&& EqualString( name, "\pAqua", true, true );
		DisposeCollection( c );
		sInited = true;
	}
	
	return sHasAqua;
}

/*--------------------------------------------------------------------------------------------------*/
static void
DoEraseMenuBackground( MenuRef menu, const Rect* rect, CGContextRef context )
{
#if TARGET_RT_MAC_MACHO
	static EraseMenuBackgroundProc	eraseProc;
	static Boolean					checked;
	
	if ( !checked )
	{
		CFBundleRef bundle = CFBundleGetBundleWithIdentifier( CFSTR("com.apple.Carbon") );
		//eraseProc = CFBundleGetFunctionPointerForName( bundle, CFSTR("EraseMenuBackground") );
		checked = true;
	}
	
	if ( eraseProc != NULL )
		(*eraseProc)( menu, rect, context );
#else
	if ( EraseMenuBackground != NULL )
		EraseMenuBackground( menu, rect, context );
#endif
	else
		DoCGContextClearRect( context, rect );
}

/*--------------------------------------------------------------------------------------------------*/
static void
DoCGContextClearRect( CGContextRef context, const Rect* rect )
{
	Rect	portBounds;
	CGRect	cgRect;
	
#if TARGET_RT_MAC_CFM
	static CGContextClearRectProc	clearProc;
	static Boolean					checked;
	
	if ( !checked )
	{
		CFBundleRef bundle = CFBundleGetBundleWithIdentifier( CFSTR("com.apple.CoreGraphics") );
		clearProc = CFBundleGetFunctionPointerForName( bundle, CFSTR("CGContextClearRect") );
		checked = true;
	}
#endif
		
	// convert from Quickdraw coordinates (zero at top) to CG coordinates (zero at bottom)
	GetPortBounds( GetQDGlobalsThePort(), &portBounds );
	cgRect.origin.x = rect->left;
	cgRect.origin.y = ( portBounds.bottom - portBounds.top ) - rect->bottom;
	cgRect.size.width = rect->right - rect->left;
	cgRect.size.height = rect->bottom - rect->top;
	
#if TARGET_RT_MAC_CFM
	(*clearProc)( context, cgRect );
#else
	CGContextClearRect( context, cgRect );
#endif
}

/*--------------------------------------------------------------------------------------------------*/
static void
DoScrollMenuImage( MenuRef menu, const Rect* bounds, int dh, int dv, CGContextRef context )
{
#if TARGET_RT_MAC_MACHO
	static ScrollMenuImageProc	scrollProc;
	static Boolean				checked;
	
	if ( !checked )
	{
		CFBundleRef bundle = CFBundleGetBundleWithIdentifier( CFSTR("com.apple.Carbon") );
		//scrollProc = CFBundleGetFunctionPointerForName( bundle, CFSTR("ScrollMenuImage") );
		checked = true;
	}
	
	if ( scrollProc != NULL )
		(*scrollProc)( menu, bounds, dh, dv, context );
#else
	if ( ScrollMenuImage != NULL )
		ScrollMenuImage( menu, bounds, dh, dv, context );
#endif
	else
		ScrollRect( bounds, dh, dv, NULL );
}

#endif
