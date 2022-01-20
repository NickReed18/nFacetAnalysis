project = 'nFacet Analysis'


extensions = [
    'nbsphinx',
    'sphinx.ext.duration',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx'#,
#    'sphinx.ext.autoapi'
]

intersphinx_mapping = {
    'python': ('https://docs.python/org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

#autoapi_dirs = ['../source']
