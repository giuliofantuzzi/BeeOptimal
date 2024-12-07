from setuptools import setup, find_packages

setup(
    name="beeoptimal",              
    version="0.1.0",                  
    description                   = "A python implementation of the Artificial Bee Colony optimization algorithm",
    long_description              = open('README.md').read(),  
    long_description_content_type = "text/markdown", 
    author                        = "Giulio Fantuzzi",
    author_email                  = "giulio.fantuzzi01@gmail.com",
    url                           = "https://github.com/giuliofantuzzi/BeeOptimal",
    packages                      = find_packages(),          
    install_requires              = [],         
    include_package_data          = True,  # Include non-code files
    package_data                  = {"beeoptimal": ["package_assets/*.png"]}
)
