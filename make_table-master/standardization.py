import pandas as pd
from sklearn.preprocessing import StandardScaler

d = 100 
trj_size = 180

new_color = pd.read_csv('../Infant_Model_V2/table/new_color_histgram.csv', index_col=0)
new_shape = pd.read_csv('../Infant_Model_V2/table/new_shape.csv', index_col=0)
new_taste = pd.read_csv('../Infant_Model_V2/table/new_taste.csv', index_col=0)
move = pd.read_csv('../Infant_Model_V2/table/new_move_k_medoids_100.csv', index_col=0)

#print(move)
move_before = move.iloc[:,:d*trj_size]
#print(move_before)
move_after = move.iloc[:,d*trj_size:]
#print(move_after)

sc = StandardScaler()
#color = (new_color - new_color.mean()) / (new_color.std())
new_color.loc[:,:] = sc.fit_transform(new_color)
#shape = (new_shape - new_shape.mean()) / (new_shape.std())
new_shape.loc[:,:] = sc.fit_transform(new_shape)
#taste = (new_taste - new_taste.mean()) / (new_taste.std())
new_taste.loc[:,:] = sc.fit_transform(new_taste)
#move_after = (move_after - move_after.mean()) / (move_after.std())
move_after.loc[:,:] = sc.fit_transform(move_after)

move = pd.concat([move_before, move_after], axis=1)
#print(move)


new_color.to_csv('../Infant_Model_V2/table/new_color_histgram_standardization.csv')
new_shape.to_csv('../Infant_Model_V2/table/new_shape_standardization.csv')
new_taste.to_csv('../Infant_Model_V2/table/new_taste_standardization.csv')
move.to_csv('../Infant_Model_V2/table/new_move_k_medoids_100_standardization.csv')
