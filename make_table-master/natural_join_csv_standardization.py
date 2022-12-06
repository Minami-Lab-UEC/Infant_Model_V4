import csv
from collections import defaultdict

sub = list(csv.reader(open('../Infant_Model_V2/table/sub.csv'), delimiter=','))
new_color = list(csv.reader(open('../Infant_Model_V2/table/new_color_histgram_standardization.csv'), delimiter=','))
new_shape = list(csv.reader(open('../Infant_Model_V2/table/new_shape_standardization.csv'), delimiter=','))
taste = list(csv.reader(open('../Infant_Model_V2/table/new_taste_standardization.csv'), delimiter=','))
move = list(csv.reader(open('../Infant_Model_V2/table/new_move_k_medoids_100_standardization.csv'), delimiter=','))

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

for row2 in new_shape:
    if row2[0] in hash:
         newRow = main_1[hash[row2[0]]] + row2[1:]
         main_2.append(newRow)

hash.clear()
main_1.clear()

for idx, row1 in enumerate(main_2):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in taste:
    if row2[0] in hash:
         newRow = main_2[hash[row2[0]]] + row2[1:]
         main_1.append(newRow)

hash.clear()
main_2.clear()

for idx, row1 in enumerate(main_1):
    hash[row1[0]] = idx #save the index of row by the key value

for row2 in move:
    if row2[0] in hash:
         newRow = main_1[hash[row2[0]]] + row2[1:]
         main_2.append(newRow)

#print(len(main_2))

    
with open('../Infant_Model_V2/table/main_new_standardization.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(main_2)
