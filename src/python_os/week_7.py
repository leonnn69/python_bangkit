#csv_to_html.py
#!/usr/bin/env python3
import sys
import csv
import os

def process_csv(csv_file):
    """Turn the contents of the CSV file into a list of lists"""
    print("Processing {}".format(csv_file))
    with open(csv_file,"r") as datafile:
        data = list(csv.reader(datafile))
    return data

def data_to_html(title, data):
    """Turns a list of lists into an HTML table"""

    # HTML Headers
    html_content = """
<html>
<head>
<style>
table {
  width: 25%;
  font-family: arial, sans-serif;
  border-collapse: collapse;
}

tr:nth-child(odd) {
  background-color: #dddddd;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}
</style>
</head>
<body>
"""


    # Add the header part with the given title
    html_content += "<h2>{}</h2><table>".format(title)

    # Add each row in data as a row in the table
    # The first line is special and gets treated separately
    for i, row in enumerate(data):
        html_content += "<tr>"
        for column in row:
            if i == 0:
                html_content += "<th>{}</th>".format(column)
            else:
                html_content += "<td>{}</td>".format(column)
        html_content += "</tr>"

    html_content += """</tr></table></body></html>"""
    return html_content


def write_html_file(html_string, html_file):

    # Making a note of whether the html file we're writing exists or not
    if os.path.exists(html_file):
        print("{} already exists. Overwriting...".format(html_file))

    with open(html_file,'w') as htmlfile:
        htmlfile.write(html_string)
    print("Table succesfully written to {}".format(html_file))

def main():
    """Verifies the arguments and then calls the processing function"""
    # Check that command-line arguments are included
    if len(sys.argv) < 3:
        print("ERROR: Missing command-line argument!")
        print("Exiting program...")
        sys.exit(1)

    # Open the files
    csv_file = sys.argv[1]
    html_file = sys.argv[2]

    # Check that file extensions are included
    if ".csv" not in csv_file:
        print('Missing ".csv" file extension from first command-line argument!')
        print("Exiting program...")
        sys.exit(1)

    if ".html" not in html_file:
        print('Missing ".html" file extension from second command-line argument!                                                                                        ')
        print("Exiting program...")
        sys.exit(1)

    # Check that the csv file exists
    if not os.path.exists(csv_file):
        print("{} does not exist".format(csv_file))
        print("Exiting program...")
        sys.exit(1)

    # Process the data and turn it into an HTML
    data = process_csv(csv_file)
    title = os.path.splitext(os.path.basename(csv_file))[0].replace("_", " ").ti                                                                                        tle()
    html_string = data_to_html(title, data)
    write_html_file(html_string, html_file)

if __name__ == "__main__":
    main()

# tricky_check.py
#!/usr/bin/env python3

import re
import csv

# Inisialisasi kamus (dictionary)
error_dict = {}
user_dict = {}

# Pola ekspresi reguler untuk mencocokkan entri log
error_pattern = r"ERROR ([\w\s']+)"
user_pattern = r"\((\w+)\)"

# Membuka dan membaca syslog.log
with open("syslog.log", "r") as log_file:
    for line in log_file:
        error_match = re.search(error_pattern, line)
        user_match = re.search(user_pattern, line)

        # Jika ada kesesuaian untuk pesan error
        if error_match:
            error = error_match.group(1)
            if error in error_dict:
                error_dict[error] += 1
            else:
                error_dict[error] = 1

        # Jika ada kesesuaian untuk pengguna
        if user_match:
            user = user_match.group(1)
            if user not in user_dict:
                user_dict[user] = {"INFO": 0, "ERROR": 0}

            # Menghitung pesan INFO dan ERROR untuk pengguna tertentu
            if "INFO" in line:
                user_dict[user]["INFO"] += 1
            elif "ERROR" in line:
                user_dict[user]["ERROR"] += 1

# Mengurutkan kamus
sorted_errors = sorted(error_dict.items(), key=lambda x: x[1], reverse=True)
sorted_users = sorted(user_dict.items())

# Menulis laporan error ke dalam error_message.csv
with open("error_message.csv", "w", newline="") as error_csv:
    csv_writer = csv.writer(error_csv)
    csv_writer.writerow(["Error", "Count"])
    csv_writer.writerows(sorted_errors)

# Menulis laporan pengguna ke dalam user_statistics.csv
with open("user_statistics.csv", "w", newline="") as user_csv:
    csv_writer = csv.writer(user_csv)
    csv_writer.writerow(["Username", "INFO", "ERROR"])
    for user, counts in sorted_users:
        csv_writer.writerow([user, counts["INFO"], counts["ERROR"]])

# Menampilkan pesan sukses
print("Laporan berhasil dihasilkan.")
