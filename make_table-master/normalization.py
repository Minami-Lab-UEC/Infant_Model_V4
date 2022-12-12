import pandas as pd

d = 100 
trj_size = 180

new_color = pd.read_csv('../Investigation/bin/table_last/new_color_histgram.csv', index_col=0)
size = pd.read_csv('../Investigation/bin/table_last/size.csv', index_col=0)
new_shape = pd.read_csv('../Investigation/bin/table_last/new_shape.csv', index_col=0)
new_taste = pd.read_csv('../Investigation/bin/table_last/new_taste.csv', index_col=0)
hardness = pd.read_csv('../Investigation/bin/table_last/hardness.csv', index_col=0)
move = pd.read_csv('../Investigation/bin/table_last/new_move_k_medoids_100.csv', index_col=0)

#print(move)
move_before = move.iloc[:,:d*trj_size]
#print(move_before)
move_after = move.iloc[:,d*trj_size:]
#print(move_after)

color = (new_color - new_color.min()) / (new_color.max() - new_color.min())
size = (size - size.min()) / (size.max() - size.min())
shape = (new_shape - new_shape.min()) / (new_shape.max() - new_shape.min())
taste = (new_taste - new_taste.min()) / (new_taste.max() - new_taste.min())
hardness = (hardness - hardness.min()) / (hardness.max() - hardness.min())
move_after = (move_after - move_after.min()) / (move_after.max() - move_after.min())

move = pd.concat([move_before, move_after], axis=1)
#print(move)


color.to_csv('../Investigation/bin/table_last/new_color_histgram_normalization.csv')
size.to_csv('../Investigation/bin/table_last/size_normalization.csv')
shape.to_csv('../Investigation/bin/table_last/new_shape_normalization.csv')
taste.to_csv('../Investigation/bin/table_last/new_taste_normalization.csv')
hardness.to_csv('../Investigation/bin/table_last/hardness_normalization.csv')
move.to_csv('../Investigation/bin/table_last/new_move_k_medoids_100_normalization.csv')
