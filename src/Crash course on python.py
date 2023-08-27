# print(15 + 5 (3 * 2) / 4 ** 2 + (3 - 7) * 7)
# print(15 + 5 + (3 * 2) / 4 ** 2 + (3 - 7) * 7)


def convert_jam(detik):
    jam = detik // 3600
    menit = (detik - jam * 3600) // 60
    detik = detik - jam * 3600 - menit * 60
    return jam, menit, detik

jam, menit, detik = convert_jam(5000)
print(jam, menit, detik)

def lucky_number(name):
    number = len(name) * 9
    print("Hello " + name, ",Your lucky number is = " + str(number))

lucky_number("leon")

def is_even(number):
    if number % 2 == 1:
        return True
    return False

is_even(3)

n = 4
if n*6 > n ** 2 or n % 2 ==0:
    print("Check")

print((24 == 5*2) and (24 > 3*5) and (2*6 == 12))

def sum_positive_numbers(n):
    if n < 1:
        return 0
    return n + sum_positive_numbers(n - 1)

print(sum_positive_numbers(3))
print(sum_positive_numbers(5))

# Fill in the blanks so that the while loop continues to run while the
# "divisor" variable is less than the "number" parameter.

def sum_divisors(number):

# Initialize the appropriate variables
  divisor = 1
  total = 0

  # Avoid dividing by 0 and negative numbers 
  # in the while loop by exiting the function
  # if "number" is less than one
  if number < 1:
    return 0 

  # Complete the while loop
  while divisor < number:
    if number % divisor == 0:
      total += divisor
    # Increment the correct variable
    divisor += 1

  # Return the correct variable 
  return total


print(sum_divisors(0)) # Should print 0
print(sum_divisors(3)) # Should print 1
print(sum_divisors(36)) #55
print(sum_divisors(102)) #114

product = 1
for n in range(1,10):
   product = product * n

print(product)

def fah_to_celc(x):
   return (x - 32) * 5 / 9

for x in range (0,101,10):
   print(x, fah_to_celc(x))

for left in range(7):
  for right in range(left,7):
    print(f"[{left}|{right}]", end=" ")
  print()

teams = ['Dragon','Wolves','Panda','Unicorns']
for home in teams:
  for away in teams:
    if home != away:
      print(home + " vs " + away)

for n in range(10):
    print(n+n)

for n in range(0,18+1,2):
    print(n*2)

def factorial(n):
  print 

x = 1
sum = 0
while x <= 10:
    sum += x
    x += 1
print(sum)

num1 = 0
num2 = 0

for x in range(5):
    num1 = x
    for y in range(14):
        num2 = y + 3

print(num1 + num2)

stringtest = "test "
print(f"{stringtest.strip()}HALO")

for x in range(1, 10, 3):
    print(x)

def count_to_ten():
  # Loop through the numbers from first to last 
  x = 1
  while x <= 10:
    print(x)
    x = 1


count_to_ten()

def even_numbers(maximum):
    return_string = "" # Initializes variable as a string

    # Complete the for loop with a range that includes all even numbers
    # up to and including the "maximum" value, but excluding 0.
    for x in range(1,maximum+1):
        if x % 2 == 0:
            return_string += str(x) + " "

        # Complete the body of the loop by appending the even number
        # followed by a space to the "return_string" variable.

    # This .strip command will remove the final " " space at the end of
    # the "return_string".
    return return_string.strip()

for outer_loop in range(2, 6+1):
    for inner_loop in range(outer_loop):
        print(f"outer loop: {outer_loop}")
        if inner_loop % 2 == 0:
            print(inner_loop)

#string
name = "leon Eleazar"
name.rfind("eon")
print(name[1])
print(name[-1])
print(name[1:5])
print(len(name))
print(name[:4])
print(name[5:])
print(name[:4]+name[5:])

new_name = name[:5] + "a" + name[6:]
print(new_name)
new_name.index("a")
"leon" in new_name

new_name.upper()
new_name.lower()
"   hi  ".strip().upper()
"   hi  ".lstrip()
"   hi  ".rstrip()
"akakakakakak".count("a")
new_name.endswith("zar")
"aku122".isnumeric()
"12345678".isnumeric()

a = (["1","2","3"])
print(a)
" + ".join(a)

"aku aku aku aku".split()

nama = "leon"
number = len(nama) * 3
print("Hello {}, ur lucky number is{}".format(nama,number))

print("your lucky number is {number}, {nama}".format(nama = nama, number = len(nama) * 3))

# batasin angka abis koma
price = 7.5
with_tax = price * 1.11
print(price,with_tax)
print("price =  ${:.2f}\nwith tax = ${:.2f}".format(price,with_tax))

def fah_to_celc(x):
   return (x - 32) * 5 / 9
for x in range(0,101,10):
   print("Fahrenheit = {:>3} | Celcius = {:.2f}\n".format(x,fah_to_celc(x)))

# string.lower() - Returns a copy of the string with all lowercase characters
# string.upper() - Returns a copy of the string with all uppercase characters
# string.lstrip() - Returns a copy of the string with the left-side whitespace removed
# string.rstrip() - Returns a copy of the string with the right-side whitespace removed
# string.strip() - Returns a copy of the string with both the left and right-side whitespace removed
# string.count(substring) - Returns the number of times substring is present in the string
# string.isnumeric() - Returns True if there are only numeric characters in the string. If not, returns False.
# string.isalpha() - Returns True if there are only alphabetic characters in the string. If not, returns False.
# string.split() - Returns a list of substrings that were separated by whitespace (whitespace can be a space, tab, or new line)
# string.split(delimiter) - Returns a list of substrings that were separated by whitespace or a delimiter
# string.replace(old, new) - Returns a new string where all occurrences of old have been replaced by new.
# delimiter.join(list of strings) - Returns a new string with all the strings joined by the delimiter 

input_string = "hello"
output_string = ""

for i in range(len(input_string) - 1, -1, -1):
    output_string += input_string[i]

print(output_string)

def is_palindrome(input_string):
    new_string = ""
    reverse_string = ""

    for letter in input_string:
        if letter != " ":
            new_string = new_string + letter
            reverse_string = letter + reverse_string

    if new_string.lower() == reverse_string.lower():
        return True
    return False

print(is_palindrome("Never Odd or Even"))  # Should be True
print(is_palindrome("abc"))  # Should be False
print(is_palindrome("kayak"))  # Should be True

def replace_ending(sentence, old, new):
    # Check if the old substring is at the end of the sentence 
    if sentence.endswith(old):
        # Using i as the slicing index, combine the part
        # of the sentence up to the matched string at the 
        # end with the new string
        #rfind buat cari huruf pertama di angkat old ada di index ke berapa
        i = sentence.rfind(old)
        # yg [:1] buat ksh batasan sentence sampe ke mana
        new_sentence = sentence[:i] + new
        return new_sentence


    # Return the original sentence if there is no match 
    return sentence
    
print(replace_ending("It's raining cats and cats", "cats", "dogs")) 
# Should display "It's raining cats and dogs"
print(replace_ending("She sells seashells by the seashore", "seashells", "donuts")) 
# Should display "She sells seashells by the seashore"
print(replace_ending("The weather is nice in May", "may", "april")) 
# Should display "The weather is nice in May"
print(replace_ending("The weather is nice in May", "May", "April")) 
# Should display "The weather is nice in April"

# list
x = ["now", "we", "are", "cooking!"]
print(x)
print(len(x))
"now" in x
"n"in x
print(x[0])
print(x[1:3])

# Using the "split" string method from the preceding lesson, complete the get_word function to return the {n}th word from a passed sentence. 
# For example, get_word("This is a lesson about lists", 4) should return "lesson", which is the 4th word in this sentence. 
# Hint: remember that list indexes start at 0, not 1. 
n="this is a lesson about list"
# split buat bikin string jadi list
n.split()
print(n[2])

def get_word(sentence, n):
	# Only proceed if n is positive 
	if n > 0:
		words = sentence.split()
		# Only proceed if n is not more than the number of words 
		if n <= len(words):
            # -1 karena index dri 0 jdi klo n = 4 biar sesuai ama ketentuan hrs di kurang 1
			return words[(n - 1)]
	return("")

print(get_word("This is a lesson about lists", 4)) # Should print: lesson
print(get_word("This is a lesson about lists", -4)) # Nothing
print(get_word("Now we are cooking!", 1)) # Should print: Now
print(get_word("Now we are cooking!", 5)) # Nothing

# add object in list
fruits = ["apple", "cherry", "melon", "strawberry"]
# append add di blkg
fruits.append("kiwi")
print(fruits)
#insert bisa taro dimana aja pake parameternya
fruits.insert(0, "orange")
print(fruits)
fruits.remove("cherry")
print(fruits)
# apus berdasakan indeks pake pop
fruits.pop(1)
print(fruits)
#ganti item
fruits[3]= "banana"
print(fruits)

animals = ["anjing", "kucing", "babi", "monyet"]
chars = 0
for animal in animals:
    chars += len(animal)
print("Jumlah huruf pada hewan = {}\njumlah rata rata huruf hewan = {}".format(chars, chars / len(animals)))

winners = ["leon", "eleazar", "ganteng"]
for index, person in enumerate(winners):
    print("{}. {}".format(index+1,person))

def full_names(person):
    result = []
    for email, name, umur in person:
        print("{} <{}> {}".format(name,email,umur))
    return result

print(full_names([("leon@gmail.com", "leon", 20), ("leon2@gmail.com", "leon2", 21)]))

fruits = ["apple", "banana", "cherry"]

for fruit in enumerate(fruits):
    print(f"Index : {fruit}")


def skip_elements(elements):
	# code goes here
	result = []
	for index, element in enumerate(elements):
		if index % 2 == 0:
			result.append(element)
	return result

print(skip_elements(["a", "b", "c", "d", "e", "f", "g"])) # Should be ['a', 'c', 'e', 'g']
print(skip_elements(['Orange', 'Pineapple', 'Strawberry', 'Kiwi', 'Peach'])) # Should be ['Orange', 'Strawberry', 'Peach']

multiple = []
for x in range(1,11):
    multiple.append(x *7)
print(multiple)
# list comperhension
multiple = [ x * 7 for x in range(1,11)]
print(multiple)

language = ["python", "ruby", "c++", "java"]
lenghts = [len(lang) for lang in language]
print(lenghts)

kelipatan_3 = [x for x in range(101) if x % 3 == 0]
print(kelipatan_3)

namee = "leon ele azar"
# namee[0] == 
word = namee.split()
print(word)
print(namee[1])

def pig_latin(text):
  say = ""
  # Separate the text into words
  words = text.split()
  for word in words:
    # Create the pig latin word and add it to the list
    word[0] = word(-1)
    # Turn the list back into a phrase
  return word
print(pig_latin("hello how are you"))

def pig_latin(text):
    say = ""
    # Separate the text into words
    words = text.split()
    pig_latin_words = []  # Create an empty list to store Pig Latin words

    for word in words:
        # Create the pig latin word and add it to the list
        # jadi buat var baru yang isinya word dri angke ke 2 sampai akhir trs
        # tambahain huruf pertama di akhir dan tambah ay
        pig_latin_word = word[1:] + word[0] + "ay"
        # terus masukin ke dalam list yg udh di buat diatas pake append
        pig_latin_words.append(pig_latin_word)

    # Turn the list back into a phrase
    # pake join biar list yg tdi join ke string yg kosong itu dan berubah jadi string biasa
    return " ".join(pig_latin_words)

print(pig_latin("hello how are you"))  # Should be "ellohay owhay reaay ouyay"
print(pig_latin("programming in python is fun"))  # Should be "rogrammingpay niay ythonpay siay unfay"

filenames = ['program.c', 'stdio.hpp', 'sample.hpp', 'a.out', 'math.hpp', 'hpp.out']
newfilenames = []

for filename in filenames:
    if filename.endswith('.c'):
        newfilenames.append(filename[:-2] + 'h')
    elif filename.endswith('.hpp'):
        newfilenames.append(filename[:-3] + 'h')
    else:
        newfilenames.append(filename)

print(newfilenames)


" ".isalpha()
"leon".replace("leon","hi")

# This function accepts a string variable "data_field".  
def count_words(data_field):

    # Splits the string into individual words. 
    split_data = data_field.split()
  
    # Then returns the number of words in the string using the len()
    # function. 
    return len(split_data)
    
    # Note that it is possible to combine the len() function and the 
    # .split() method into the same line of code by inserting the 
    # data_field.split() command into the the len() function parameters.

# Call to the function
count_words("Catalog item 3523: Organic raw pumpkin seeds in shell")
# Should print 9

genre = "transcendental"
genre[:-8]

def count_letters(text):
    # Initialize a new dictionary.
    dictionary = {}
    
    # Complete the for loop to iterate through each "text" character and 
    # use a string method to ensure all letters are lowercase.
    for x in text.lower():
        # Complete the if-statement using a string method to check if the
        # character is a letter.
        if x.isalpha(): 
            # Complete the if-statement using a logical operator to check if 
            # the letter is not already in the dictionary.
            if x not in dictionary:
                # Use a dictionary operation to add the letter as a key
                # and set the initial count value to zero.
                dictionary[x] = 0   
            # Use a dictionary operation to increment the letter count value 
            # for the existing key.
            dictionary[x] += 1 
            
    # Return the dictionary with letter counts.
    return dictionary

print(count_letters("AaBbCc"))
# Should be {'a': 2, 'b': 2, 'c': 2}

print(count_letters("Math is fun! 2+2=4"))
# Should be {'m': 1, 'a': 1, 't': 1, 'h': 1, 'i': 1, 's': 1, 'f': 1, 'u': 1, 'n': 1}

print(count_letters("This is a sentence."))
# Should be {'t': 2, 'h': 1, 'i': 2, 's': 3, 'a': 1, 'e': 3, 'n': 2, 'c': 1}
