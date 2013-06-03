import os
from ConfigParser import ConfigParser
from fabric import api
from fabric.api import env
from fabric.contrib import files


env.roledefs = {
    'dev': ['192.168.33.10']
}

env.local_base_dir = os.path.dirname(__file__)
env.home_dir = '/home/vagrant'
env.gnome_base_dir = '%s/src/gnome' % env.home_dir
env.py_gnome_dir = '%s/py_gnome' % env.gnome_base_dir
env.webgnome_dir = '%s/web/gnome/webgnome' % env.gnome_base_dir
env.gnome_venv_dir = '%s/envs/gnome' % env.home_dir


def ensure_gnome_exists():
    """
    Clone the GNOME repo if it does not already exist on the server.
    """
    if not files.exists('~/src'):
        api.run('mkdir ~/src')

    if not files.exists(env.gnome_base_dir):
        print 'Downloading Gnome source...'
        api.run('git clone https://trac.orr.noaa.gov/git/GNOME %s' %
                env.gnome_base_dir)


def setup_py_gnome():
    """
    Setup dependencies for py_gnome.
    """
    with api.cd(env.py_gnome_dir):
        with virtualenv():
            api.run('pip install -r requirements.txt')
            api.run('python setup2.py develop')


def setup_webgnome():
    """
    Setup dependencies for webgnome.
    """
    with api.cd(env.webgnome_dir):
        with virtualenv():
            api.run('pip install -r requirements.txt')
            api.run('python setup.py develop')


def get_current_role():
    """
    Get the name of the Fabric role that matches the current host.
    """
    for role, servers in env.roledefs.items():
        if env.host_string in servers:
            return role


def setup_vagrant_key():
    """
    Set up access to the user's vagrant server.
    """
    if env.key_filename and env.user == 'vagrant':
        return

    if env.host_string in env.roledefs['dev']:
        env.user = 'vagrant'
        with api.lcd(os.path.join(env.local_base_dir, 'conf')):
            result = api.local('vagrant ssh-config | grep IdentityFile',
                               capture=True)
            env.key_filename = result.split()[1].replace('"', '')


def virtualenv():
    return api.prefix('source %s/bin/activate' % env.gnome_venv_dir)


def get_config():
    """
    Parse and load a "vagrant.cfg" file in the current directory if it exists.
    """
    config = ConfigParser()
    config.read(os.path.join(env.local_base_dir, 'conf', 'vagrant.cfg'))
    return config


@api.task
def restart_apache():
    setup_vagrant_key()
    api.sudo('service apache2 restart')


@api.task
def reload_apache():
    setup_vagrant_key()
    api.sudo('service apache2 reload')


@api.task
def test_apache():
    setup_vagrant_key()
    api.sudo('service apache2 configtest')


@api.task
def tail_apache():
    setup_vagrant_key()
    api.sudo('tail -f /var/log/apache2/webgnome-error.log '
             '/var/log/apache2/webgnome-access.log')


@api.task
def setup_apache():
    setup_vagrant_key()
    contexts = {
        'dev': {
            'ip_address': '*',
            'port': 80,
            'hostname': 'localhost',
            'python_path': '/home/vagrant/envs/gnome',
            'deploy_path': '/home/vagrant/src/gnome/web/gnome/webgnome'
        }
    }

    role = get_current_role()

    if not role:
        return

    files.upload_template(
        os.path.join(env.local_base_dir, 'conf', 'apache_site.conf'),
        '/etc/apache2/sites-available/webgnome',
        contexts[role], use_sudo=True)

    api.sudo('a2ensite webgnome')
    api.sudo('a2dissite default')
    api.sudo('apache2ctl configtest')
    api.execute(reload_apache)


@api.task
def setup_host():
    setup_vagrant_key()
    config = get_config()
    git_username = config.get('git', 'username')
    git_password = config.get('git', 'password')
    packages = ['vim', 'apache2', 'libapache2-mod-uwsgi', 'libnetcdf6',
                'libnetcdf-dev', 'python-scipy', 'python-numpy', 'git', 'vim',
                'emacs', 'python-pip', 'python-virtualenv', 'python-lxml',
                'virtualenvwrapper', 'libxslt1-dev', 'python-netcdf',
                'libblas-dev', 'python-dev', 'gfortran', 'g++',
                'subversion', 'libcurl3-dev', 'libhdf5-serial-dev',
                'liblapack-dev', 'libapache2-mod-wsgi']

    print 'Provisioning WebGnome development environment.'
    api.sudo('apt-get --quiet update')
    api.sudo('apt-get --quiet install -y %s' % ' '.join(packages))
    files.append('~/.gitconfig', "[http]\nsslVerify=False\n")
    files.append('~/.gitconfig', "[credential]\nhelper = cache\n")
    files.append('~/.netrc',
                 'machine trac.orr.noaa.gov login %s password %s' % (
                 git_username, git_password))

    # Fix paths to libraries PIL uses.
    # http://jj.isgeek.net/2011/09/install-pil-with-jpeg-support-on-ubuntu-oneiric-64bits/
    api.sudo('ln -s /usr/lib/`uname -i`-linux-gnu/libjpeg.so /usr/lib')
    api.sudo('ln -s /usr/lib/`uname -i`-linux-gnu/libfreetype.so /usr/lib')
    api.sudo('ln -s /usr/lib/`uname -i`-linux-gnu/libz.so /usr/lib')

    ensure_gnome_exists()

    with api.cd(env.gnome_base_dir):
        api.run('git pull origin master')
        api.run('git checkout linux_support', warn_only=True)
        api.sudo('ln -s ~/src/gnome/web/gnome/webgnome/webgnome '
                 '/var/www/', warn_only=True)
        api.sudo('chown -R vagrant:www-data web')

        # Apache needs to be able to write to the models directory and the
        # directory for user-uploaded files.
        api.run('mkdir web/gnome/webgnome/webgnome/static/models',warn_only=True)
        api.run('mkdir web/gnome/webgnome/webgnome/static/uploads', warn_only=True)
        api.run('sudo chmod -R g+w web/gnome/webgnome/webgnome/static/models')
        api.run('sudo chmod -R g+w web/gnome/webgnome/webgnome/static/uploads')

        api.sudo('chown -R vagrant:www-data src')

        print 'Setting up project virtualenv'

        if not files.exists('~/envs'):
            api.run('mkdir ~/envs')
        if not files.exists(env.gnome_venv_dir):
            api.run('virtualenv %s' % env.gnome_venv_dir)

        files.append('~/.bashrc', 'export WORKON_HOME=~/envs')

        # We installed numpy with apt-get to get non-python dependencies, but
        # now we want the latest numpy.
        with api.cd(env.py_gnome_dir):
            with virtualenv():
                api.run('pip install --upgrade numpy')

        setup_py_gnome()
        setup_webgnome()

    api.sudo('chown -R vagrant:vagrant /home/vagrant')
    api.execute(setup_apache)


@api.task
def build_docs():
    base_path = os.path.dirname(os.path.realpath(__file__))
    docs_dir = os.path.join(base_path, 'doc')
    python_docs_dir = os.path.join(docs_dir, 'api', 'python')
    js_docs_dir = os.path.join(docs_dir, 'javascript')
    project_dir = os.path.join(base_path, 'webgnome')
    js_dir = os.path.join(project_dir, 'static', 'js')

    # Auto-generate Python API docs first.
    api.local('sphinx-apidoc -f %s -o %s' % (project_dir, python_docs_dir))

    with api.lcd(docs_dir):
        api.local('make html')

    # Auto-generate JavaScript API docs.
    docco_path = os.path.join(base_path, 'node_modules', 'docco', 'bin', 'docco')
    files_to_include = ['models.js', 'app.js', 'util.js']

    for filename in files_to_include:
        file_path = os.path.join(js_dir, filename)
        api.local('%s %s -o %s' % (docco_path, file_path, js_docs_dir))


@api.task
def pull(branch='master'):
    setup_vagrant_key()
    ensure_gnome_exists()

    with api.cd(env.gnome_base_dir):
        api.run('git pull origin %s' % branch)
        current_branch = api.run('git rev-parse --abbrev-ref HEAD')

        print 'Checking out %s' % branch
        if current_branch != branch:
            api.run('git checkout %s' % branch)


@api.task
def deploy_webgnome(restart=False, branch='master'):
    setup_vagrant_key()
    api.execute(pull, branch)

    with api.cd(env.webgnome_dir):
        api.run('touch app.wsgi')

    if restart:
        api.execute(restart_apache)
