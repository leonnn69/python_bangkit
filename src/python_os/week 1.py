import requests as req
response = requests.get("http://google.com")
len(response.text)

import numpy as np

def numpyArray():
    x = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    y = numpy.array([[3, 6, 2], [9, 12, 8]], np.int32)
    return x*y


print(numpyArray())

print("12")

import shutil
du = shutil.disk_usage("/")
print(du)
du.free/du.total * 100

import psutil
psutil.cpu_percent(10)

# buat cek pc
import shutil
import psutil

def check_disk_usage(disk):
    du = shutil.disk_usage(disk)
    free = du.free / du. total * 100
    return free > 20

def checl_cpu_usage():
    usage = psutil.cpu_percent(1)
    return usage < 75

if not check_disk_usage("/") or not checl_cpu_usage():
    print("ERROR!")
else:
    print("everything is OK!")

