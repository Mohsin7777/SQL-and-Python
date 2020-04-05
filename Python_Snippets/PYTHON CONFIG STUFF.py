'''

JUPYTER LAB: - Root
From Admin Command prompt (regular or anaconda):
1 > cd /D X:\Python_Scripts\Jupyter_Stuff
launch command:
2 > jupyter lab

JUPYTER LAB - Tensorflow
1) Launch Admin level Commandprompt for the Tensorflow environment
(Either the tensorflow prompt shortcut, or via Navigator)
2) Launch the Command Prompt
3) cd /D X:\Python_Scripts\Jupyter_Stuff
4) jupyter lab

NOTE: Tensorflow environment is needed because it has its own dependencies
>>>>>>> Need to install all needed libraries (pandas etc.) within this environment manually, through the navigator GUI

# Note: Any dataset downloaded in Keras/Tensorflow will save to: C:\Users\Administrator\.keras\datasets (or tensorflow_datasets)

PYTHON install location:
(within Anaconda3's root folder (currently python 3.7))
X:\Anaconda\

** Only use Conda to install packages now wherever possible!!
(pip only for stuff conda doesnt have, and install in seperate environment) **

First update conda to latest:
> conda update -n root conda

Then update all packages
> conda update --all
(NOTE: it will sometimes need to downgrade certain packages to match dependences)


Pycharm doc for Anaconda:
https://docs.anaconda.com/anaconda/user-guide/tasks/pycharm/

Updating Anaconda:
Doc: https://docs.anaconda.com/anaconda/install/update-version/

# PyCharm Import:

# (from main directory:)
# import sys
# sys.path.append("X:\Anaconda\Lib\")
# import urllib as urllib

# (from site packages:)
# For pandas:
# import sys
# sys.path.append("X:\Anaconda\Lib\site-packages")
# import pandas as pd



>> Tutorials: (1)
(done) 0) Data Science specific basics (Codebasics) [jupyter]]
(done) 1A) Data Analysis with Pandas (Codebasics)  [pandas, numpy, matplotlib]
(done) 1B) SQL in pandas (query, import to dataframe, connection strings, etc. ) 
3) Machine Learning (Codebasics)
4A) Deep learning (Codebasics)  -subset of neural networks
4B) Coursera: https://www.coursera.org/specializations/deep-learning
5) Stats/Probability tutorials:
https://www.khanacademy.org/math/statistics-probability


>> Courses (2)

0) Book: Hands-On Machine Learning with Scikit-Learn and TensorFlow: Concepts, Tools, and Techniques to Build Intelligent Systems
>>> datasets + code: https://github.com/ageron/handson-ml2


>> Deep Learning Specialization - Andrew Ng — Coursera



Option 1) -boring... high risk... zero productive value:
>>> Building a Stock Analyzer 


Option 2) Use Free/Open Data Sources:
Create something that you can sell to a company for a billion $$ !
