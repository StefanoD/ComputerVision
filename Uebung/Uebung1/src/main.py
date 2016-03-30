#!/usr/bin/python3

import importlib

# import libcore this way for PyCharm
libcore = importlib.import_module('libcore')
Img = libcore.Img

def main():
    img = Img.load_image('../gletscher.jpg')


    print(img)


if __name__ == "__main__": main()
