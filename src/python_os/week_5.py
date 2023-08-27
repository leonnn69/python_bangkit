import re
def rearrange_name(name):
    result = re.search(r"^([\w.]*), ([\w.]*)$", name)
    if result is None:
        return name
    return "{} {}".format(result[2], result[1])

rearrange_name("leon")

rearrange_name("")

def validate_name(username, minlen):
    assert type(username) == str, "username must be a string"
    if minlen < 1:
        raise ValueError("minlen must be at least 1")
    if len(username) < minlen:
        return False
    if not username.isalnum():
        return False
    return True

validate_name("leon", 1)
validate_name(123, 1)

import os
os.getcwd()


#email.py
#!/usr/bin/env python3
import csv
import sys
def populate_dictionary(filename):
  """Populate a dictionary with name/email pairs for easy lookup."""
  email_dict = {}
  with open(filename) as csvfile:
    lines = csv.reader(csvfile, delimiter = ',')
    for row in lines:
      name = str(row[0].lower())
      email_dict[name] = row[1]
  return email_dict
def find_email(argv):
  """ Return an email address based on the username given."""
  # Create the username based on the command line input.
  try:
    fullname = str(argv[1] + " " + argv[2])
    # Preprocess the data
    email_dict = populate_dictionary('/home/student-04-b4ba5097cd6e/data/user_em                                                                                        ails.csv')
     # If email exists, print it
    if email_dict.get(fullname.lower()):
      return email_dict.get(fullname.lower())
    else:
      return "No email address found"
  except IndexError:
    return "Missing parameters"
def main():
  print(find_email(sys.argv))
if __name__ == "__main__":
  main()

# emails test.py
#!/usr/bin/env python3
import unittest

from emails import find_email
class EmailsTest(unittest.TestCase):
  def test_basic(self):
    testcase = [None, "Bree", "Campbell"]
    expected = "breee@abc.edu"
    self.assertEqual(find_email(testcase), expected)
  def test_one_name(self):
    testcase = [None, "John"]
    expected = "Missing parameters"
    self.assertEqual(find_email(testcase), expected)
  def test_two_name(self):
    testcase = [None, "Roy","Cooper"]
    expected = "No email address found"
    self.assertEqual(find_email(testcase), expected)
if __name__ == '__main__':
  unittest.main()
