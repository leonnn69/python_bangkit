log = "July 31 07:52:58 mycomputer bad_process[12345]: ERROR Performing package upgrade"
index = log.index("[")
# print(index)
print(log[index+1:index+6])

# cari yang lebih keren
import re
log = "July 31 07:52:58 mycomputer bad_process[12345]: ERROR Performing package upgrade"
regex = r"\[(\d+)\]"
result = re.search(regex, log)
print(result[1])

# nama = ["rehan", "rapi", "sadam"]
# for n in nama:
#     print("sama sama {}".format(n))

# nyari kata di file
import re
# bisanya di bash
# nyari kata thon dalam suatu file
# grep thon <lokasi.file>
# contoh output : phthon, thona, klithon

# nyari kata walau ada beda di kapital
# grep -i python <lokasi.file>
# contoh output : Python, python

# nyari kata pake .
# grep l.rts <lokasi.file>
# contoh output : flirts, alerts, blurts

# cari kata yg depannya udh disebutin
# grep ^fruit <lokasi.file>
# contoh outpun : fruitloop, fruittea, fruiter

# # cari kata yang belakangnya disebutin
# grep cat$ <lokasi.file>
# contoh output : pussycat, nyancat, lamcat

# contoh nyari di file
import re

# Pola regex yang ingin dicocokkan
pattern = r"thon"

# Nama file yang ingin Anda cari di dalamnya
file_name = "cari.txt"

# Membuka dan membaca file
with open(file_name, "r") as file:
    contents = file.read()

# Mencocokkan pola dengan isi file menggunakan re.search()
result = re.search(pattern, contents)

# Memeriksa apakah ada hasil pencocokan
if result:
    print("Teks ditemukan:", result.group())
else:
    print("Teks tidak ditemukan.")

# nyari dalem suatu string
import re
result = re.search(r"aza", "plaza")
print(result)

# contoh klo gaada
result = re.search(r"azar", "plaza")
print(result)

# nyari dri huruf depan
result = re.search(r"^x", "xoni")
print(result)

# pake dot
result = re.search(r"l.o", "leon")
print(result)

result = re.search(r"p.ng", "pingpong")
print(result)

#biar bisa walau ada huruf kapital
result = re.search(r"p.ng", "Pingpong", re.IGNORECASE)
print(result)

result = re.search(r"[Pp]ython", "python")
print(result)

result = re.search(r"[a-z]way", "the end of the highway")
print(result)

result = re.search(r"cloud[a-zA-Z0-9]", "cloudy")
print(result)

result = re.search(r"cloud[a-zA-Z0-9]", "cloud9")
print(result)

# pake ^ yang artinya kyk bukan
result = re.search(r"[^a-zA-z]", "the end of the highway")
# result bakal space
print(result)

result = re.search(r"[^a-zA-z ]", "the end of the highway.")
# result bakal titik karna space juga udah masuk
print(result)

result = re.search(r"cat|dog", "I like cats.")
print(result)

result = re.search(r"cat|dog", "I like dogs.")
print(result)

# kalo ada 2 di dalam kalimatnya
# kalo gini bakal ke print yg pertama doang
result = re.search(r"cat|dog", "I like dogs and cats.")
print(result)

# harus pake findall
result = re.findall(r"cat|dog", "I like dogs and cats.")
print(result)

# .* buat masukin berapa pun char sampe batas yg disebut
result = re.search(r"py.*n", "pythooooon")
print(result)

result = re.search(r"py.*n", "python programming")
print(result)

# buat ksh tau cuman boleh antara a-z jdi space gabole
result = re.search(r"py[a-z]*n", "python programming")
print(result)

# print sesuai yg huruf ditulis tpi klo ada yg sama ditulis lgi
result = re.search(r"o+l+", "goldfish")
print(result)

result = re.search(r"o+l+", "gooollll")
print(result)

# klo di tengahnya ada yg beda gabole
result = re.search(r"o+l+", "goal")
print(result)

# pake ? buat biar klo ada bole klo engga juga gpp
result = re.search(r"p?each", "i like each of u")
print(result)

result = re.search(r"p?each", "i like peach")
print(result)

# kalo mau cari yg special char di teks
result = re.search(r"\.com", "google.com")
print(result)

result = re.search(r"\.com", "telcom")
print(result)

# \w buat samain semuana
result = re.search(r"\w*", "google is good")
print(result)

result = re.search(r"\w*", "google_is_good")
print(result)

# contoh soal
import re
def check_character_groups(text):
  result = re.search(r"\w[0-9]", text)
  return result != None

print(check_character_groups("One")) # False
print(check_character_groups("123  Ready Set GO")) # True
print(check_character_groups("username user_01")) # True
print(check_character_groups("shopping_list: milk, bread, eggs.")) # False

# nyari yg depannya pake huruf apa dan blkg huruf apa
print(re.search(r"^[Aa].*[Aa]$", "argentina"))

print(re.search(r"^[Aa].*[Aa]$", "Indonesia"))

print(re.search(r"^[Aa].*[Aa]$", "azerbaijan"))

# buat valid
pattern = r"^[a-zA-z_][a-zA-Z0-9_]*$"
print(re.search(pattern,"_yang_benar"))

print(re.search(pattern,"contoh yang salah"))

print(re.search(pattern,"2_contoh_yang_salah"))

print(re.search(pattern,"contoh_yang_benar_12345"))

# cek kalimat
import re

def check_sentence(text):
    result = re.match(r"^[A-Z][a-z ]*[\?\.]$", text)
    return result is not None

print(check_sentence("Is this is a sentence?")) # True
print(check_sentence("is this is a sentence?")) # False
print(check_sentence("Hello")) # False
print(check_sentence("1-2-3-GO!")) # False
print(check_sentence("A star is born.")) # True


import re

def check_zip_code(text):
    # (?<!^) buat mastiin ada string diawal
    # \d untuk cari angkat {5} berarti 5 angka
    # (?:-\d{4})? buat yg optional
    result = re.search(r" (?<!^)\d{5}(?:-\d{4})?", text)
    return result != None

print(check_zip_code("The zip codes for New York are 10001 thru 11104.")) # True
print(check_zip_code("90210 is a TV show")) # False
print(check_zip_code("Their address is: 123 Main Street, Anytown, AZ 85258-0001.")) # True
print(check_zip_code("The Parliament of Canada is at 111 Wellington St, Ottawa, ON K1A0A9.")) # False

result = re.search(r"^(\w*), (\w*)$", "leon, eleazar")
print(result)
print(result.groups())
# klo 0 smuanya
print(result[0])
# klo 1 kata ke 1
print(result[1])
"{} {}".format(result[2],result[1])

def rearrange_name(name):
    result = re.search(r"^(\w*), (\w)$", name)
    if result is None:
        return name
    return("{} {}".format(result[2], result[1]))
rearrange_name("leon, eleazar")
rearrange_name("tyara, regina, nadya")

# biar bisa ada character unik di nama         
def rearrange_name(name):
    result = re.search(r"^([\w \.-]*), ([\w \.-]*)$", name)
    if result is None:
        return name
    return("{} {}".format(result[2], result[1]))
rearrange_name("kennedy , john F.")

# print pake batas
print(re.search(r"[a-zA-z]{5}", "hell o world"))
print(re.search(r"[a-zA-z]{5}", "hello world full of anjing"))
#print smua yg ada 5 huruf
print(re.findall(r"[a-zA-z]{5}", "hello world full of anjing"))
# print yang bener bener 5 huruf
print(re.findall(r"\b[a-zA-z]{5}\b", "hello world full of anjing"))
print(re.findall(r"\w{5,10}", "hello world worldui fullllllllllll of anjingkkkkkkk 12345678911"))
print(re.findall(r"\w{5,}", "hi hello world worldui fullllllllllll of anjingkkkkkkk 12345678911"))
print(re.findall(r"w\w{,10}", "hi hello world worldui fullllllllllll of anjingkkkkkkk 12345678911"))

# print yg jumlah katanya diatas yg disebut
print(re.findall(r"\w{10,}", "hi hello world worldui fullllllllllll of anjingkkkkkkk 12345678911")) 


log = "July 31 07:52:58 mycomputer bad_process[12345]: ERROR Performing package upgrade"
def extract_PID(log_line):
    regex = r"\[(\d+)\]"
    result = re.search(regex, log_line)
    if result is None:
        return " "
    return result

print(extract_PID(log))
print(extract_PID("99 elephant in a [cage]"))

import re

def extract_pid(log_line):
    regex = r"\[(\d+)\]: (\w+)"
    result = re.search(regex, log_line)
    if result is None:
        return None
    return "{} ({})".format(result.group(1), result.group(2))


print(extract_pid("July 31 07:51:48 mycomputer bad_process[12345]: ERROR Performing package upgrade")) # 12345 (ERROR)
print(extract_pid("99 elephants in a [cage]")) # None
print(extract_pid("A string that also has numbers [34567] but no uppercase message")) # None
print(extract_pid("July 31 08:08:08 mycomputer new_process[67890]: RUNNING Performing backup")) # 67890 (RUNNING)

# motong" kalimat
re.split(r"[.,!?]","heloo, apakabar? saya leon.")
# motong kalimat tpi tanda bacanya juga iktu
re.split(r"([.,!?])","heloo, apakabar? saya leon.")

# ganti kata jdi yg dalam r" " buat cari kalimat yg <sebuah kata>@<sebuah kata lgi>
re.sub(r"[\w,%+-]+@[\w.-]+", "DISENSOR", "received an email from anonymous@gmail.com")

re.sub(r"^([\w .-]*), ([\w .-]*)$",r"\2,+1-,\1", "leon, eleazar")

# soal no 1
import re
def transform_record(record):
  new_record = re.sub(r"^([\w ]*),([\d ]*),([\w]*)$",r"\1,+1-\2,\3",record)
  return new_record

print(transform_record("Sabrina Green,802-867-5309,System Administrator")) 
# Sabrina Green,+1-802-867-5309,System Administrator

print(transform_record("Eli Jones,684-3481127,IT specialist")) 
# Eli Jones,+1-684-3481127,IT specialist

print(transform_record("Melody Daniels,846-687-7436,Programmer")) 
# Melody Daniels,+1-846-687-7436,Programmer

print(transform_record("Charlie Rivera,698-746-3357,Web Developer")) 
# Charlie Rivera,+1-698-746-3357,Web Developer

# soal no 2
import re
def multi_vowel_words(text):
  pattern = r"\w+[aiueo][aiueo][aiueo]\w*"
  result = re.findall(pattern, text)
  return result

print(multi_vowel_words("Life is beautiful")) 
# ['beautiful']

print(multi_vowel_words("Obviously, the queen is courageous and gracious.")) 
# ['Obviously', 'queen', 'courageous', 'gracious']

print(multi_vowel_words("The rambunctious children had to sit quietly and await their delicious dinner.")) 
# ['rambunctious', 'quietly', 'delicious']

print(multi_vowel_words("The order of a data queue is First In First Out (FIFO)")) 
# ['queue']

print(multi_vowel_words("Hello world!")) 
# []

# soal no 4 
import re
def transform_comments(line_of_code):
  result = re.sub(r"#+\s","// ",line_of_code)
  return result

print(transform_comments("### Start of program")) 
# Should be "// Start of program"
print(transform_comments("  number = 0   ## Initialize the variable")) 
# Should be "  number = 0   // Initialize the variable"
print(transform_comments("  number += 1   # Increment the variable")) 
# Should be "  number += 1   // Increment the variable"
print(transform_comments("  return(number)")) 
# Should be "  return(number)"

# soal no 5
import re
def convert_phone_number(phone):
  result = re.sub(r"(\b\d{3})-(\d{3})-(\d{4}\b)", r"(\1) \2-\3",phone)
  return result

print(convert_phone_number("My number is 212-345-9999.")) # My number is (212) 345-9999.
print(convert_phone_number("Please call 888-555-1234")) # Please call (888) 555-1234
print(convert_phone_number("123-123-12345")) # 123-123-12345
print(convert_phone_number("Phone number of Buckingham Palace is +44 303 123 7300")) # Phone number of Buckingham Palace is +44 303 123 7300

import os
os.getcwd()
#--------------------------------------------
import re
import csv
def contains_domain(address, domain):
  """Returns True if the email address contains the given,domain,in the domain position, false if not."""
  domain = r'[\w\.-]+@'+domain+'$'
  if re.match(domain,address):
    return True
  return False
def replace_domain(address, old_domain, new_domain):
  """Replaces the old domain with the new domain in the received address."""
  old_domain_pattern = os.getcwd() + old_domain + '$'
  address = re.sub(old_domain_pattern, new_domain, address)
  return address
def main():
  """Processes the list of emails, replacing any instances of the old domain with the new domain."""
  old_domain, new_domain = 'abc.edu', 'xyz.edu'
  csv_file_location = 'user_emails.csv'
  report_file = '' + '/updated_user_emails.csv'
  user_email_list = []
  old_domain_email_list = []
  new_domain_email_list = []
  with open(csv_file_location, 'r') as f:
    user_data_list = list(csv.reader(f))
    user_email_list = [data[1].strip() for data in user_data_list[1:]]
    for email_address in user_email_list:
      if contains_domain(email_address, old_domain):
        old_domain_email_list.append(email_address)
        replaced_email = replace_domain(email_address,old_domain,new_domain)
        new_domain_email_list.append(replaced_email)
    email_key = ' ' + 'Email Address'
    email_index = user_data_list[0].index(email_key)
    for user in user_data_list[1:]:
      for old_domain, new_domain in zip(old_domain_email_list, new_domain_email_list):
        if user[email_index] == ' ' + old_domain:
          user[email_index] = ' ' + new_domain
  f.close()
  with open(report_file, 'w+') as output_file:
    writer = csv.writer(output_file)
    writer.writerows(user_data_list)
    output_file.close()
main()














