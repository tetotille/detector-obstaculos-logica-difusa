from os.path import abspath,dirname,join
from sys import argv

########################### COPIAR IMAGEN ####################################

if len(argv)>1:
    filename = join(dirname(dirname(abspath(__file__))),f"img/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))),"img/barco.jpg")

##############################################################################

print(filename)

########################### COPIAR VIDEO ####################################

if len(argv)>1:
    filename = join(dirname(dirname(abspath(__file__))),f"videos/{argv[1]}")
else:
    filename = join(dirname(dirname(abspath(__file__))),"videos/Video2.mp4")

##############################################################################

print(filename)
