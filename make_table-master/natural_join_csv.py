import csv
from collections import defaultdict

sub = list(csv.reader(open('../Investigation/bin/table_last/sub.csv'), delimiter=','))
new_color = list(csv.reader(open('../Investigation/bin/table_last/new_color_histgram.csv'), delimiter=','))
size = list(csv.reader(open('../Investigation/bin/table_last/new_size.csv'), delimiter=','))
new_shape = list(csv.reader(open('../Investigation/bin/table_last/new_shape.csv'), delimiter=','))
taste = list(csv.reader(open('../Investigation/bin/table_last/new_taste.csv'), delimiter=','))
hardness = list(csv.reader(open('../Investigation/bin/table_last/new_hardness.csv'), delimiter=','))
move = list(csv.reader(open('../Investigation/bin/table_last/new_move_k_medoids_100.csv'), delimiter=','))

hash = {}
main_1 = []
main_2 = []

for idx, row1 in enumerate(sub):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in new_color:
    if row2[0] in hash:
         newRow = sub[hash[row2[0]]] + row2[1:]
         main_1.append(newRow)

#print(len(main_1))

hash.clear()

for idx, row1 in enumerate(main_1):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in size:
    if row2[0] in hash:
         newRow = main_1[hash[row2[0]]] + row2[1:]
         main_2.append(newRow)

#print(len(main_2))

hash.clear()
main_1.clear()

for idx, row1 in enumerate(main_2):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in new_shape:
    if row2[0] in hash:
         newRow = main_2[hash[row2[0]]] + row2[1:]
         main_1.append(newRow)

hash.clear()
main_2.clear()

for idx, row1 in enumerate(main_1):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in taste:
    if row2[0] in hash:
         newRow = main_1[hash[row2[0]]] + row2[1:]
         main_2.append(newRow)

hash.clear()
main_1.clear()

for idx, row1 in enumerate(main_2):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in hardness:
    if row2[0] in hash:
         newRow = main_2[hash[row2[0]]] + row2[1:]
         main_1.append(newRow)

#print(len(main_1))

hash.clear()
main_2.clear()

for idx, row1 in enumerate(main_1):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in move:
    if row2[0] in hash:
         newRow = main_1[hash[row2[0]]] + row2[1:]
         main_2.append(newRow)

#print(len(main_2))

    
with open('../Investigation/bin/table_last/main_new_-1.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(main_2)
