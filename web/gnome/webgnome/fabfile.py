import os
from fabric import api


def build_docs():
    base_path = os.path.dirname(os.path.realpath(__file__))
    docs_dir = os.path.join(base_path, 'doc')
    python_docs_dir = os.path.join(docs_dir, 'api', 'python')
    js_docs_dir = os.path.join(docs_dir, 'javascript')
    project_dir = os.path.join(base_path, 'webgnome')
    js_dir = os.path.join(project_dir, 'static', 'js')

    # Auto-generate Python API docs first.
    api.local('sphinx-apidoc %s -o %s' % (project_dir, python_docs_dir))

    with api.lcd(docs_dir):
        api.local('make html')

    # Auto-generate JavaScript API docs.
    docco_path = os.path.join(base_path, 'node_modules', 'docco', 'bin', 'docco')
    files_to_include = ['gnome.js']

    for filename in files_to_include:
        file_path = os.path.join(js_dir, filename)
        api.local('%s %s -o %s' % (docco_path, file_path, js_docs_dir))