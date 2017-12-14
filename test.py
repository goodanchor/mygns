import os

import scipy.misc
import numpy as np
from six.moves import xrange

#from Gan_model import GAN


#from utils import pp,visualize,to_json,show_all_variables
import utils
import tensorflow as tf

from glob import glob


def main():
    a = glob(r"./imout/b/*a.png")
    b = glob(r"./imout/b/*b.png")
    print(a)
    print("ok")
# def main():
#     print("ok")
#     print("wangwang")

if __name__ == "__main__":
    main() 