These notes are for developers that use the Eclipse software development
platform.  Please feel free to disregard these notes if you are not using Eclipse.


It is sometimes desirable to run a pyramid application in an Eclipse
debug session.

And in order to configure Eclipse to do this, a couple things need to be
configured.

- Create a new PyDev project, or open an existing one.
  - Open the Project Properties
  - Add a new python interpreter or configure the existing one.
    - add the local library pyramid has created (if you used pcreate).
      - Click the New Folder button.
      - Select the folder your pyramid library is in (it is the folder
        that contains the __init__.py for the project's main module)
  - Setup pserve to run on Eclipse
    - Create a "Python Run" configuration
      - In the Main tab, configure the Main Module to point to the
        installed pserve executable.  The location of this executable
        needs to match the python interpreter.
        Example:
          If the full path of my OSX python executable is: /Users/username/envs/gnome/bin/python
          Then the full path of the pserve executable is:  /Users/username/envs/gnome/bin/pserve
      - In the Arguments tab, add the Program argument "development.ini"
      - In the Arguments tab, set the working directory to:
          ${workspace_loc:WebGnome}/../../../web/gnome/webgnome
      - In the Common tab, Uncheck launch in background

