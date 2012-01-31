#!/usr/bin/env python2.4

"""
A module to provide a labeled window: one with a regular old wx.Window
and a wx.StaticText next to each other.

What you get is a Sizer with the widgets layed out for you.

"""

import wx

def LabeledWindowSizer(window, label, pos = wx.TOP, space=2, stretch=0, SizerFlags=0):

    """
    LabeledWindow(window,
                  label,
                  pos = wx.TOP,
                  space=2,
                  stretch=0,
                  SizerFlags=0):

    Creates a Sizer that contains the wx.Window passed in, with a
    wx.StaticText as a label to one side of the Window. The resulting
    Sizer can then be Added, as a unit, to another sizer.

    window: Any wx.Window -- created in the usual way.

    label: A string with the text you want in the label.

    pos: position: one of: wx.TOP, wx.LEFT, wx.RIGHT, wx.BOTTOM

    space: number of pixels of space between the label and the Window

    stretch: the Sizer "option" parameter, indicating how much you want
    the Window to stretch

    SizerFlags: any Sizer Flags for the window you want passed into the sizer.
    
    """

    ## use the window's parent for the StaticText
    parent = window.GetParent()
    Text = wx.StaticText(parent, label=label)
    if pos == wx.TOP:
        Sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer.Add(Text, 0, wx.ALIGN_LEFT)
        Sizer.Add((1,space),0)
        Sizer.Add(window, stretch, SizerFlags)

    elif pos == wx.LEFT:
        Sizer = wx.BoxSizer(wx.HORIZONTAL)
        Sizer.Add(Text, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)
        Sizer.Add((space,1), 0)
        Sizer.Add(window, stretch, wx.ALIGN_CENTER_VERTICAL|SizerFlags)

    elif pos == wx.RIGHT:
        Sizer = wx.BoxSizer(wx.HORIZONTAL)
        Sizer.Add(window, stretch, wx.ALIGN_CENTER_VERTICAL|SizerFlags)
        Sizer.Add((space,1), 0)
        Sizer.Add(Text, 0, wx.ALIGN_RIGHT | wx.ALIGN_CENTER_VERTICAL)

    elif pos == wx.BOTTOM:
        Sizer = wx.BoxSizer(wx.VERTICAL)
        Sizer.Add(window, stretch, SizerFlags)
        Sizer.Add((1,space),0)
        Sizer.Add(Text, 0, wx.ALIGN_LEFT)


    else:
        raise ValueError("%s is not a supported Position"%pos)

    return Sizer

if __name__ == "__main__":

    Choices = "these are some choices A_really_Long_One".split()

    class TestPanel(wx.Panel):
        def __init__(self, *args, **kwargs):
            wx.Panel.__init__(self, *args, **kwargs)

            Sizer = wx.FlexGridSizer(cols=2, vgap=20, hgap=20)

            # basic:
            win = wx.Choice(self, choices=Choices)
            win = LabeledWindowSizer(win, label="A top label:")
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL)

            win = wx.Choice(self, choices=Choices)
            win = LabeledWindowSizer(win, label="A left label:", pos=wx.LEFT)
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL)

            # one with a small Window and large label
            win = wx.Choice(self, choices=Choices[:-1])
            win = LabeledWindowSizer(win, label="A long top label:")
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL)

            # A Textcontrol:
            win = wx.TextCtrl(self, value="Some sample text")
            win = LabeledWindowSizer(win,
                                     label="A left Label:",
                                     pos=wx.LEFT,
                                     stretch=1,
                                     )
            Sizer.Add(win, 1, wx.ALIGN_CENTER_VERTICAL|wx.GROW)

            # one with a small Window and small label
            win = wx.Choice(self, choices=Choices[:-1])
            win = LabeledWindowSizer(win, label="A label:")
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL|wx.ALIGN_RIGHT)


            win = wx.TextCtrl(self, value="Some sample text", style=wx.TE_MULTILINE)
            win = LabeledWindowSizer(win,
                                     label="A left Label:",
                                     pos=wx.LEFT,
                                     stretch=1,
                                     )
            Sizer.Add(win, 1, wx.ALIGN_CENTER_VERTICAL|wx.GROW)


            win = wx.Choice(self, choices=Choices)
            win = LabeledWindowSizer(win,
                                     label="A bottom label:",
                                     pos=wx.BOTTOM)
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL)

            win = wx.Choice(self, choices=Choices)
            win = LabeledWindowSizer(win, label=":A right label", pos=wx.RIGHT)
            Sizer.Add(win, 0, wx.ALIGN_CENTER_VERTICAL)

            
            self.SetSizerAndFit(Sizer)
            

    App = wx.App()
    f = wx.Frame(None)
    p = TestPanel(f)
    f.Fit()
    f.Show()
    App.MainLoop()

