from pyramid.paster import bootstrap
from webgnome import util


def main():
    env = bootstrap('../development.ini')

    util.CleanDirectoryCommand(
        directory=env['registry'].settings.upload_dir,
        description='Remove all uploaded files.')()

    env['closer']()

if __name__ == '__main__':
    main()
