file = open("hello_world.txt")
print(file.readline())

print(file.read())

file.close()

# biar otomatik ke close pake with
with open("hello_world.txt") as file:
    print(file.read())

with open("hello_world.txt") as file:
    for line in file:
        print(line.upper())

# buat ilangin spasi diantara line
with open("hello_world.txt") as file:
    for line in file:
        print(line.strip().upper())

#masukin file ke list
file = open("hello_world.txt")
lines = file.readlines()
file.close()
lines.sort()
print(lines)

# w buat bikin file baru tpi bakal ilang klo buat w yang baru
# jdi kyk buat yg baru
with open("novel.txt", "w") as file:
    file.write("pada jaman dahulu")


with open("novel.txt", "w") as file:
    file.write("pada jaman dahulu kala\n")

# klo pake a buat nambahin di akhir
# kyk fungsi append
with open("novel.txt", "a") as file:
    file.write("ada seekor kelinci")

# masukin tamu ke guest
guest = open("guest.txt", "w")
tamu_yang_ada = ["leon", "tyara", "rehan", "tasya"]

for i in tamu_yang_ada:
    guest.write(i + "\n")

guest.close()

#tambahin tamu
tamu_baru = ["gabe", "sadam", "rapi"]

with open("guest.txt", "a") as guest:
    for j in tamu_baru:
        guest.write(j+"\n")

#Cek tamu
with open("guest.txt", "r") as guest:
    for g in guest:
        print(g.strip())

# ada yg check out
tamu_cekout = ["leon", "tyara"]
temp_list = []

with open("guest.txt", "r") as guest:
    for g in guest:
        temp_list.append(g.strip())

print(temp_list)

with open("guest.txt", "w") as guest:
    for name in temp_list:
        if name not in tamu_cekout:
            guest.write(name + "\n")

#Cek tamu
with open("guest.txt", "r") as guest:
    for g in guest:
        print(g.strip())

# cek tamu msh ada ato udh co
tamu_yg_dicek = ["leon", "gabe"]
table_tamu_kosong = []
with open("guest.txt", "r") as guest:
    for g in guest:
        table_tamu_kosong.append(g.strip())
    for tamu in tamu_yg_dicek:
        if tamu in table_tamu_kosong:
            print("tamu atas nama ",tamu," masih ada")
        else:
            print("tamu atas nama", tamu, " sudah tidak ada")

# modify a file
import os
os.remove("novel.txt")
os.rename("novel.txt", "cerpen.txt")
# cek file
os.path.exists("cerpen.txt")
# cek ukuran file
os.path.getsize("guest.txt")
os.path.getmtime("guest.txt")

import datetime
timestamp = os.path.getmtime("guest.txt")
tgl = datetime.datetime.fromtimestamp(timestamp)
print(tgl)

# letak file detail
os.path.abspath("guest.txt")

print(os.getcwd())
os.mkdir("new_dir")

# pindah directory
os.chdir("C:\\Users\\Talitha\\Downloads\\Python\\Python")
os.getcwd()

os.chdir(".vscode")
os.mkdir("newer_dir")
os.rmdir("newer_dir")
os.rmdir("new_dir")

#check isi directory
os.listdir(".vscode")
os.listdir("Using_Python_to_Interact_with_the_Operating_System_Google")

os.listdir(".vscode")
dir = ".vscode"
for name in os.listdir(dir):
    # join buat gabungin nama dir sama name dari file" yg ada di ".vscode"
    fullname = os.path.join(dir, name)
    #isdir buat liat itu directory ato bukan
    if os.path.isdir(fullname):
        print("{} is a directory".format(fullname))
    else:
        print("{} is a file".format(fullname))

# soal no 1
def create_python_script(filename):
    comments = "# Start of a new Python program"
    with open(filename, 'w') as file:
        file.write(comments)
    filesize = os.path.getsize(filename)
    return filesize

with open("program.py", 'w') as file:    
    os.path.getsize(file)

print(create_python_script("program.py"))
# soal no 2
import os

def new_directory(directory, filename):
  current_dir = os.getcwd()
  # Before creating a new directory, check to see if it already exists
  if os.path.isdir(directory) == False:
    os.mkdir(directory)

  # Create the new file inside of the new directory
  os.chdir(directory)
  with open (filename, 'w') as file:
    pass
  
  os.chdir(current_dir)
  # Return the list of files in the new directory
  return os.listdir(directory)

print(new_directory("PythonPrograms", "script.py"))

os.getcwd()
# buat cek isi directory in current directory
os.listdir()

# buat cek isi directory in directory yg di tulus
os.listdir("PythonPrograms")

#soal no 4
import os
import datetime

def file_date(filename):
  # Create the file in the current directory
#   os.mkdir(filename)
  timestamp = os.path.getmtime(filename)
  # Convert the timestamp into a readable format, then into a string
  tgl = datetime.datetime.fromtimestamp(timestamp)
  # strf itu adalah  "string format time", yang merupakan metode dalam modul datetime di Python 
  # yang memungkinkan Anda mengonversi objek datetime menjadi string dalam format yang diinginkan.
  formatted_date = tgl.strftime("%Y-%m-%d")

  # Return just the date portion 
  # Hint: how many characters are in “yyyy-mm-dd”? 
  return (formatted_date)
#   return(tgl)

print(file_date("newfile.txt")) 
# Should be today's date in the format of yyyy-mm-dd

# no 5
import os

def parent_directory():
    # Create a relative path to the parent 
    # of the current working directory 
    # '..' buat mengibaratkan dia adalah directory diatasnya
    relative_parent = os.path.join(os.getcwd(), '..')

    # Return the absolute path of the parent directory
    # os.path.abspath() is used to get the absolute path of the parent directory
    # jdi dia print yg atasnya
    return os.path.abspath(relative_parent)

print(parent_directory())

os.getcwd()
os.chdir("Python")

#---------------------------------
import file_csv.txt

file = open("file_csv.txt")
print(file.read())

import csv
file = open("file_csv.txt")
csv_file = csv.reader(file)
for row in csv_file:
    nama, no_tlp, role = row
    print("nama = {}, no tlp = {}, role = {}".format(nama, no_tlp, role))
file.close()

# masukin data ke csv
host = [["google.com","8.8.8.8"],["cloudfare", "1.1.1.1"],["gatau lagi","6.9.6.9"]]
with open('host.csv', 'w') as hosts:
    writer = csv.writer(hosts)
    writer.writerows(host)

# baca data di csv yg ke bawah arahnya
with open('daftar.csv') as daftar:
    reader = csv.DictReader(daftar)
    jumlah = len(row['nama'])
    print(jumlah - 1)
    for row in reader:
        print(("nama : {}, umur : {}".format(row['nama'],row["umur"])))


# masukin data ke csv pake dictionary
pegawai = [{"nama" : "leon", "umur" : 21, "bagian" : 'machine learning'},
           {"nama" : 'tyara', "umur" : 22, "bagian" : 'back end'},
           {"nama" : 'gabe', "umur" : 23, "bagian" : 'front end'}]
keys = ["nama", "umur", "bagian"]
with open("daftar_karyawan.csv", 'w') as daftar_karyawan:
    writer = csv.DictWriter(daftar_karyawan, fieldnames=keys)
    writer.writeheader()
    writer.writerows(pegawai)

with open('daftar_karyawan.csv') as daftar:
    reader = csv.DictReader(daftar())
    jumlah_pegawai = len(list(reader))
    for row in reader:
        print(("nama : {}, umur : {} bagian : {}".format(row['nama'],row["umur"],row['bagian'])))
    print("jumlah pegawai = {}".format(jumlah_pegawai))

# ------------------------------------

import csv
def read_employees(csv_file_location):
    # ini buat di linux 
    # csv.register_dialect('empDialect', skipinitialspace=True, strict =True )
                                                        # masi gatau buat apa dialect, diapus juga ga ngaruh
  employee_file = csv.DictReader(open(csv_file_location))#, dialect='empDialect')
  employee_list = []
  for data in employee_file:
    employee_list.append(data)
  return employee_list

# employee_list = read_employees('employees.csv')
# print(employee_list)

print(read_employees('employees.csv'))

def process_data(employee_list):
  department_list = []
  for employee_data in employee_list:
    department_list.append(employee_data['Department'])
  department_data = {}
  for department_name in set(department_list):
    department_data[department_name] = department_list.count(department_name)
  return department_data

dictionary = process_data(employee_list)
print(dictionary)

def write_report(dictionary, report_file):
  with open(report_file, "w+") as f:
    for k in sorted(dictionary):
      f.write(str(k) + ':' + str(dictionary[k]) + '\n')
    f.close()

write_report(dictionary,'test_report.txt')
