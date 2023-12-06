
import os
import csv

def writeCSV(file_name, line):
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)
    else:
        with open(file_name, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(line)

def clearFile(file_path):
    with open(file_path, 'w') as file:
        file.truncate(0)