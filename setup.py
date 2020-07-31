import setuptools

with open('README.md', 'r') as readme:
    long_description = readme.read()

setuptools.setup(
    name='ba-tools',
    version='0.7.12',
    author='Joel McCune',
    author_email='jmccune@esri.com',
    description='Feature engineering using ArcGIS Pro with Business Analyst for using quantitative Geography '
                'with Machine Learning.',
    license='Apache 2.0',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/knu2xs/ba-tools',
    packages=['ba_tools'],
    install_requires=[
        'arcgis>=1.8.2',
        'numpy',
        'pandas',
        'requests',
        'scikit-learn'
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.6',
        'License :: OSI Approved :: MIT License',
        'Operating System :: Microsoft :: Windows',
        'Topic :: Scientific/Engineering :: GIS'
    ]
)
