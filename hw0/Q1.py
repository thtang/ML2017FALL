from collections import defaultdict
import sys


path = sys.argv[1]
print("load "+path)
text_file = open(path, "r")
words = text_file.read().split()


d = defaultdict(int)
for word in words:
    d[word]+=1

word_set = []
for word in words:
    if word not in word_set:
        word_set.append(word)

i = 0
with open('Q1.txt', 'w') as the_file:
    for word in word_set[:-1]:
        j = str(i)
        the_file.write(word +" "+j+" "+str(d[word])+"\n")
        i+=1
    the_file.write(word_set[-1] +" "+str(i)+" "+str(d[word_set[-1]]))