import math

def triangle(alas, tinggi):
    return 1/2 * alas * tinggi

def rectangle(panjang, lebar):
    return panjang * lebar

def circle(radius):
    return ["{:.2f}".format(math.pi*(radius**2))]

circle(14)
