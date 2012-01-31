#!/usr/bin/env python
"""Image utilities.
"""
import logging, os, re, sys, traceback, warnings
import Image     # Python Imaging Library (PIL)

warn = logging.getLogger('image_util').warn


# Suppress FutureWarning from PIL; we can't do anything about it.
warnings.filterwarnings('ignore', '.*return a long.*')

THUMB_PIL_TYPE = "JPEG"   # Thumbnail type; one of PIL's output formats.
THUMB_EXT = ".jpg"        # The filename extension for that type.

# Caches image dimensions for reuse.
_dimensions_cache = {}

RX_DECODER_NOT_AVAILABLE = re.compile( R"decoder .* not available" )
RX_TRUNCATED = re.compile( R"image file is truncated" )

def open_image(image_path):
    """Open an image file in PIL, return the Image object.
       Return None if PIL doesn't recognize the file type.
    """
    try:
        im = Image.open(image_path)
    except IOError, e:
        if str(e) == "cannot identify image file":
            return None
        else:
            raise
    except:
        m = "caught exception identifying '%s', assuming non-image:\n%s"
        e = traceback.format_exc()
        warn(m, image_path, e)
        return None
    return im

def make_thumb(image_path, width):
    """Make a thumbnail and save it in the same directory as the original.

       See get_thumb_path() for the arguments.
       @return The thumbnail filename, or None if PIL
           didn't recognize the image type.

       Does NOT work with PDF originals; use make_thumb_from_pdf for those.
    """
    dst = get_thumb_path(image_path, width)
    im = open_image(image_path)
    if im is None:
        return None
    orig_width, orig_height = im.size
    height = choose_height(width, orig_width, orig_height)
    if im.mode == 'P':
        im = im.convert()   # Convert GIF palette to RGB mode.
    try:
        im.thumbnail((width, height), Image.ANTIALIAS)
    except IOError, e:
        reason = str(e)
        if RX_DECODER_NOT_AVAILABLE.search(reason):
            return None   # PIL error, cannot thumbnail.
        elif RX_TRUNCATED.search(reason):
            return None   # Corrupt image?  Can't thumbnail.
        else:
            raise
    im.save(dst, THUMB_PIL_TYPE)
    return dst

def choose_height(new_width, width, height):
    """Return the height corresponding to 'new_width' that's proportional
       to the original size.
    """
    proportion = float(height) / float(width)
    return int(new_width * proportion)

def get_dimensions(image_path, use_cache=False):
    """Return the width and height of an image.
       Returns (None, None) if PIL doesn't recognize the file type.

       @param use_cache bool If true, use the cached dimensions if
       available.  This cuts down on filesystem accesses, but the cache may
       be wrong if the image has changed.  If false, update the cache anyway
       so it's correct.

       @exc IOError raised by PIL if the image file is missing or you don't
       have read permission for it.
    """
    image_path = str(image_path)   # Don't need a path object.
    if use_cache and image_path in _dimensions_cache:
        return _dimensions_cache[image_path]
    im = open_image(image_path)
    if im is None:
        size = (None, None)
    else:
        size = im.size
    _dimensions_cache[image_path] = size
    return size

def changed(image_path=None):
    """Delete all cached data regarding this path because the file has
       changed.  If arg is unspecified or None, delete all cached data
       for all paths.
    """
    if image_path is None:
        _dimensions_cache.clear()
        return
    if image_path in _dimensions_cache:
        del _dimensions_cache[image_path]

def get_thumb_path(image_path, width):
    """Return the thumbnail path for the given image.
       
       @parm image_path str The original image filename.
       @param width int The thumbnail width in pixels.
       @return path The thumbnail path.
       For "a/foo.jpg", returns path("a/foo_thumbWIDTH.jpg").
       The return value always ends with THUMB_EXT regardless of the original
       extension.
    """
    dir, old_name = os.path.split(image_path)
    base, ext = os.path.splitext(old_name)
    new_name = "%s_thumb%d%s" % (base, width, THUMB_EXT)
    return os.path.join(dir, new_name)

def test():
    print "Height for 600x480 @ width 200 is", choose_height(200, 600, 480)
    print "Path 200 for a/foo.jpg is", get_thumb_path('a/foo.jpg', 200)
    print "Path 200 for a/foo.png is", get_thumb_path('a/foo.png', 200)

if __name__ == "__main__":  test()
