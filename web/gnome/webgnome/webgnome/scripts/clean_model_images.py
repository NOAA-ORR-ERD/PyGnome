from pyramid.paster import bootstrap
from webgnome import util


def main():
    env = bootstrap('../development.ini')

    util.CleanDirectoryCommand(
        directory=env['registry'].settings.model_data_dir,
        description='Remove all model image files.')()

    env['closer']()

if __name__ == '__main__':
    main()
