


################################################################
# Connect to SQL Server
# NOTE: different DBs have different connection strings
################################################################

import pandas as pd
from sqlalchemy import create_engine

# connection string (this connects with windows authentication, launch SQL Server with admin mode):
sqlcon = create_engine('mssql+pyodbc://@' + 'GHOST-117\SQLEXPRESS' + '/' + 'MOHSIN' + '?trusted_connection=yes&driver=ODBC+Driver+13+for+SQL+Server')

# Now use SQL Query to bring in whatever (yay!) - USE THIS QUERY TO BRING IN SPECIFIC DATA:
df = pd.read_sql_query("SELECT TOP (1000) * FROM [MOHSIN].[dbo].[customers]", sqlcon)

# to bring in an entire table:
df = pd.read_sql_table("table_name", sqlcon)
# specific columns:
df = pd.read_sql_table("table_name", sqlcon, columns['c2','c3'])

# Write table 'df2' into SQL Server
df2.to_sql(
    name='test', #the name of the created table
    con=sqlcon,
    index=False,
    schema='MOHSIN.dbo',
    if_exists='fail'  #fail, replace, append
)
