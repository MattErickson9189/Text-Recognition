
path = "./Data/Resized/"
wordsList = "./Data/words.txt"

list = open(wordsList)

count = 0

for line in list:
    if not line or line[0] == '#':
        continue

    split = line.strip().split(' ')
    assert len(split) >= 9

    if(split[1] == 'err'):
        count += 1

print("Number of err files: ", count)