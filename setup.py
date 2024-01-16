

from setuptools import setup, find_packages

setup(
    # Basic package information:
    name    ='simplevec',   
    version ='0.0.3',
    packages=find_packages(),  # Automatically find packages in the directory

    # Dependencies:
    install_requires=[
        'numpy>=1.1.1',  
        'pandas>1.7.7', 
        'gensim>4.1.1',
        'scipy',
        'scikit-learn',
        'sympy',
        'statsmodels', 
        'xgboost',
    ],

    # Metadata for PyPI:
    author          ='James A. Rolfsen',
    author_email    ='james.rolfsen@think.dev', 
	description     ='SimpleVec: Elementary Examples of Vector Modeling',
	url             ='https://github.com/jrolf/simplevec',    # Link to your repo
    license         ='MIT',
    
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',  # If your README is in markdown

    # More classifiers: https://pypi.org/classifiers/
    classifiers=[
        'Programming Language :: Python :: 3.7', 
        'License :: OSI Approved :: MIT License',  # Ensure it matches the LICENSE
        'Operating System :: OS Independent',
        'Natural Language :: English',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    

)





