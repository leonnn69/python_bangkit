def to_second(hour, minutes, second):
    return hour * 3600 + minutes * 60 + second

print("Welcome to second converter")

cont = 'y'
while cont.lower() == 'y':
    hour = int(input("Enter the number of hours: "))
    minutes = int(input("Enter the number of minutes: "))
    second = int(input("Enter the number of seconds: "))
    
    print("That's {} seconds".format(to_second(hour, minutes, second)))
    print()
    cont = input("Want to do more conversion? [y to continue] ")

print("Goodbye!")

import subprocess
subprocess.run(["date"])
subprocess.run(["sleep","2"])

result = subprocess.run(["host", "8.8.8.8"], capture_output=True)
print(result.returncode)

import re
pattern = r" USER \((\w+)\)$"
line = " jul 6 14:4:01 computer.name CRON[29440] : USER (naughty_user)"
result = re.search(pattern, line)
print(result[1])

#-------------------------------------------------------------------------
import re

def show_time_of_pid(line):
    pattern = r"(\w{3} \d{1} \d{2}:\d{2}:\d{2}).+\[(\d+)\]"
    result = re.search(pattern, line)
    return "{} pid:{}".format(result.group(1), result.group(2))


print(show_time_of_pid("Jul 6 14:01:23 computer.name CRON[29440]: USER (good_user)")) # Jul 6 14:01:23 pid:29440
print(show_time_of_pid("Jul 6 14:02:08 computer.name jam_tag=psim[29187]: (UUID:006)")) # Jul 6 14:02:08 pid:29187
print(show_time_of_pid("Jul 6 14:02:09 computer.name jam_tag=psim[29187]: (UUID:007)")) # Jul 6 14:02:09 pid:29187
print(show_time_of_pid("Jul 6 14:03:01 computer.name CRON[29440]: USER (naughty_user)")) # Jul 6 14:03:01 pid:29440
print(show_time_of_pid("Jul 6 14:03:40 computer.name cacheclient[29807]: start syncing from \"0xDEADBEEF\"")) # Jul 6 14:03:40 pid:29807
print(show_time_of_pid("Jul 6 14:04:01 computer.name CRON[29440]: USER (naughty_user)")) # Jul 6 14:04:01 pid:29440
print(show_time_of_pid("Jul 6 14:05:01 computer.name CRON[29440]: USER (naughty_user)"))


username = {}
name ="good_user"
username[name]= username.get(name,0) +1
print(username)

#-----------------------------
#!/usr/bin/env python3
import sys
import os
import re
log_file = "fishy.log"
def error_search(log_file):
  error = input("What is the error? ")
  returned_errors = []
  with open(log_file, mode='r',encoding='UTF-8') as file:
    for log in  file.readlines():
      error_patterns = ["error"]
      for i in range(len(error.split(' '))):
        error_patterns.append(r"{}".format(error.split(' ')[i].lower()))
      if all(re.search(error_pattern, log.lower()) for error_pattern in error_patterns):
        returned_errors.append(log)
    file.close()
  return returned_errors
  
def file_output(returned_errors):
  with open(os.path.expanduser('~') + 'errors_found.log', 'w') as file:
    for error in returned_errors:
      file.write(error)
    file.close()
if __name__ == "__main__":
  log_file = sys.argv[1]
  returned_errors = error_search(log_file)
  file_output(returned_errors)
  sys.exit(0)

