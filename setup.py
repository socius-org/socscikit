from distutils.core import setup

setup(
  name = 'socscikit',         # How you named your package folder (MyLib)
  packages = ['socscikit'],   # Chose the same as "name"
  version = '0.0.3',      # Start with a small number and increase it with every change you make
  license='Apache-2.0',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'TYPE YOUR DESCRIPTION HERE',   # Give a short description about your library
  author = 'Nick S.H Oh',                   # Type in your name
  author_email = 'nick.sh.oh@socialscience.ai',      # Type in your E-Mail
  url = 'https://github.com/nick-sh-oh/socscikit',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/nick-sh-oh/socscikit/archive/refs/tags/0.0.3.tar.gz',    # I explain this later on
  keywords = ['AI', 'SOCIAL SCIENCE'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'tweepy',
          'plotly',
          'conlp'
      ]
)

