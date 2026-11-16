project = 'ov-radiomics'
author = 'radiomics contributors'
extensions = ['myst_nb', 'sphinxcontrib.mermaid']
html_theme = 'pydata_sphinx_theme'
html_title = 'Radiomics · 教程与文档'
html_baseurl = 'https://hurry060215-tech.github.io/radiomics/'
language = 'zh_CN'
nb_execution_mode = 'off'
exclude_patterns = ['_build']
html_theme_options = {
    'github_url': 'https://github.com/hurry060215-tech/radiomics',
    'navigation_depth': 3,
    'show_toc_level': 2,
    'header_links_before_dropdown': 5,
    'navbar_end': ['theme-switcher', 'navbar-icon-links'],
}
html_context = {'default_mode': 'light'}
myst_heading_anchors = 3
html_static_path = ['_static']
html_css_files = ['custom.css']


