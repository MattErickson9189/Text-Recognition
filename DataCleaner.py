import os
path = "./Data/Resized/"
wordsList = "./Data/words.txt"

list = open(wordsList)

total = 0;
count = 0

errorCount = 0

for line in list:
    #Skips all the comment and blank lines
    if not line or line[0] == '#':
        continue
    #Keeps track of the total number of lines
    total += 1

    split = line.strip().split(' ')
    assert len(split) >= 9

    if(split[1] == 'err'):
        #Counts how many files are going to be deleted
        count += 1

        #Gets the files name
        fileName = split[0] + '-Resized.png'
        fileToDelete = path + fileName
        try:
            os.remove(fileToDelete)
        except FileNotFoundError:
            errorCount += 1
            continue

print("Number of files: ", total)
print("Number of files removed: ", count)
print("Files previously deleted: ", errorCount)
print()
#Checks how many files are in the directory

check = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for files in filenames:
        check += 1
print("Images in Resized directory:", check)