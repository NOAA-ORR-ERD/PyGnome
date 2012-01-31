import glob
import logging
import os
import subprocess

warn = logging.getLogger("multimedia").warn

try:
    from hazpy.image_util import THUMB_EXT
except ImportError: ## this here to support testing without a working image_util
    THUMB_EXT = ".jpg"
    
    
def make_pdf_thumbnail(path, width):
    """Make a thumbnail from a PDF file.

       @parm image_path str The original image filename.
       @param width int The thumbnail width in pixels. (Will be approximate.)
       @return path The thumbnail path.
       For "a/foo.jpg", returns path("a/foo_thumbWIDTH.jpg").
       The return value always ends with THUMB_EXT regardless of the original
       extension.

       Requires the "imagemagick" package to be installed.
    """
    width_str = str(width)
    dir, name = os.path.split(path)
    base, ext = os.path.splitext(name)
    newbase = "%s_thumb%s" % (base, width_str)
    dst = os.path.join(dir, newbase + THUMB_EXT)

    def page(n):
        """Return the filename for page n's thumbnail, n >= 0.
           'n' may also be a string (e.g., "*" for wildcard patterns).
           If 'n' is None, return value has no page suffix.
        """
        if n is not None:
            suffix = "-%s" % n
        else:
            suffix = ""
        return os.path.join(dir, newbase + suffix + THUMB_EXT)

    trashcan = open("/dev/null", "w")
    cmd = ["/usr/bin/convert", "-geometry", width_str, path, dst]
    status = subprocess.call(cmd, shell=False, stderr=trashcan)
    if status:
        warn("make_pdf_thumbnail subcommand exited with status %s: %s", 
            status, cmd)
    trashcan.close()
    found = False
    if os.path.exists(dst):
        found = True
    page0_fn = page(0)
    other_files = glob.glob(page("*"))
    for fn in other_files:
        if fn == page0_fn and not found:
            os.rename(fn, dst)
            found = True
        else:
            os.remove(fn)
    if found:
        return dst
    else:
        return None

def make_pdf_thumbnail2(path, width):
    """Make a thumbnail from a PDF file.

       This version uses just ghostscript, rather than ImageMagik
       -- chb

       @parm image_path str The original image filename.
       @param width int The thumbnail width in pixels. (Will be approximate -- assumes 8.5in wide paper.)
       @return path The thumbnail path.
       For "a/foo.jpg", returns path("a/foo_thumbWIDTH.jpg").
       The return value always ends with THUMB_EXT regardless of the original
       extension.

       Requires ghostscript to be installed.
    """
    width_str = str(width)
    dir, name = os.path.split(path)
    base, ext = os.path.splitext(name)
    newbase = "%s_thumb%s" % (base, width_str)
    dst = os.path.join(dir, newbase + THUMB_EXT)

    def page(n):
        """Return the filename for page n's thumbnail, n >= 0.
           'n' may also be a string (e.g., "*" for wildcard patterns).
           If 'n' is None, return value has no page suffix.
        """
        if n is not None:
            suffix = "-%s" % n
        else:
            suffix = ""
        return os.path.join(dir, newbase + suffix + THUMB_EXT)

    trashcan = open("/dev/null", "w")
    
    ## A few settable options
    if THUMB_EXT == ".jpg":
        filetype = "jpeg" # jpeg
    elif THUM_EXT == ".png":
        filetype = "png16m" # 24 bit png
    else:
        filetype = "jpeg" # should this be default
    
    gs_path = "/usr/local/bin/gs"
    ps_cmd = "save pop currentglobal true setglobal false/product where{pop product(Ghostscript)search{pop pop pop revision 600 ge{pop true}if}{pop}ifelse}if{/pdfdict where{pop pdfdict begin/pdfshowpage_setpage[pdfdict/pdfshowpage_setpage get{dup type/nametype eq{dup/OutputFile eq{pop/AntiRotationHack}{dup/MediaBox eq revision 650 ge and{/THB.CropHack{1 index/CropBox pget{2 index exch/MediaBox exch put}if}def/THB.CropHack cvx}if}ifelse}if}forall]cvx def end}if}if setglobal"
    cmd = [gs_path, "-dSAFER","-dBATCH","-dNOPAUSE","-dLastPage=1","-dTextAlphaBits=4"]
    cmd.append("-sDEVICE=%s"%filetype)
    #dpi  = int(width / 8.5) ## this assumes an 8.5in wide piece of paper.
    dpi = 20
    cmd.append("-r%i"%dpi)
    
    cmd.append("-sOutputFile=%s"% dst)
    cmd.extend(("-c", ps_cmd, "-f"),)
    cmd.append(path)
    
    ## the desired command string
    ## gs -dSAFER -dBATCH -dNOPAUSE -r150 -sDEVICE=jpeg -dTextAlphaBits=4 -sOutputFile=$1-%02d.jpg $1
    status = subprocess.call(cmd, shell=False) #, stdout=trashcan, stderr=trashcan)
    if status:
        warn("make_pdf_thumbnail subcommand exited with status %s: %s", 
            status, cmd)
    trashcan.close()
    found = False
    if os.path.exists(dst):
        return dst
    else:
        return None
    

def get_pdf_text(path):
    raise NotImplementedError

def get_word_text(path):
    raise NotImplementedError


if __name__ == "__main__":
    import optparse
    logging.basicConfig()
    parser = optparse.OptionParser(usage="%prog PDF_FILE")
    opts, args = parser.parse_args()
    if len(args) != 1:
        parser.error("wrong number of command-line arguments")
    source_file = args[0]
    
    width = 200
    dst = make_pdf_thumbnail2(source_file, width)
    print "Thumbnail made:", dst

#ps_cmd = "save pop currentglobal true setglobal false/product where{pop product(Ghostscript)search{pop pop pop revision 600 ge{pop true}if}{pop}ifelse}if{/pdfdict where{pop pdfdict begin/pdfshowpage_setpage[pdfdict/pdfshowpage_setpage get{dup type/nametype eq{dup/OutputFile eq{pop/AntiRotationHack}{dup/MediaBox eq revision 650 ge and{/THB.CropHack{1 index/CropBox pget{2 index exch/MediaBox exch put}if}def/THB.CropHack cvx}if}ifelse}if}forall]cvx def end}if}if setglobal"

#gs -dLastPage=1 -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dNOPAUSE -dBATCH -sDEVICE=jpeg -r20 -sOutputFile=Chem_Sheet_LPG.jpg -c "save pop currentglobal true setglobal false/product where{pop product(Ghostscript)search{pop pop pop revision 600 ge{pop true}if}{pop}ifelse}if{/pdfdict where{pop pdfdict begin/pdfshowpage_setpage[pdfdict/pdfshowpage_setpage get{dup type/nametype eq{dup/OutputFile eq{pop/AntiRotationHack}{dup/MediaBox eq revision 650 ge and{/THB.CropHack{1 index/CropBox pget{2 index exch/MediaBox exch put}if}def/THB.CropHack cvx}if}ifelse}if}forall]cvx def end}if}if setglobal" -f Chem_Sheet_LPG.pdf

#gs -dTextAlphaBits=4 -dGraphicsAlphaBits=4 -dNOPAUSE -dBATCH -sDEVICE=png16m -r9.06531732174037 -sOutputFile=thb%d.png -c "save pop currentglobal true setglobal false/product where{pop product(Ghostscript)search{pop pop pop revision 600 ge{pop true}if}{pop}ifelse}if{/pdfdict where{pop pdfdict begin/pdfshowpage_setpage[pdfdict/pdfshowpage_setpage get{dup type/nametype eq{dup/OutputFile eq{pop/AntiRotationHack}{dup/MediaBox eq revision 650 ge and{/THB.CropHack{1 index/CropBox pget{2 index exch/MediaBox exch put}if}def/THB.CropHack cvx}if}ifelse}if}forall]cvx def end}if}if setglobal" -f Chem_Sheet_LPG.pdf


