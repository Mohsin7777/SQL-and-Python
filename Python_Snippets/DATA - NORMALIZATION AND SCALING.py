


###############################################
# Scaling and Normalization
###############################################

# Detected Issue: scaling is messed up
# The y axis goes from 40K to 160K (very wide), while X axis is only 27-42 (too narrow)

# Use MinMax scaler to scale the income feature
# It will rescale income to a range between 0 and 1
scaler = MinMaxScaler()
scaler.fit(df[['Income']])
df.Income = scaler.transform(df[['Income']])

# Do the same scalling for Age (between 0 and 1)
scaler.fit(df[['Age']])
df.Age = scaler.transform(df[['Age']])

df.head()