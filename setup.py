from distutils.core import setup

setup(
  name = 'socscikit',         # How you named your package folder (MyLib)
  packages = ['socscikit'],   
  version = '0.0.6.1',      
  license='Apache-2.0',        
  description = 'TYPE YOUR DESCRIPTION HERE',   
  author = 'Nick S.H Oh',                   #
  author_email = 'nick.sh.oh@socialscience.ai',      
  url = 'https://github.com/socius-org/socscikit',  
  download_url = 'https://github.com/socius-org/socscikit/archive/refs/tags/0.0.6.1.tar.gz', 
  keywords = ['AI', 'SOCIAL SCIENCE'],   # Keywords that define your package best
  install_requires=[            
          'torch',
          'rich',
          'wandb'
      ],
  include_package_data=True
)

