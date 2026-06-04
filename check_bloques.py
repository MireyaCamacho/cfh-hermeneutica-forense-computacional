import sqlite3 
import pandas as pd 
conn = sqlite3.connect('cfh.db') 
df = pd.read_sql("SELECT corpus, COUNT(*) as n FROM bloques GROUP BY corpus", conn) 
print(df) 
print("TOTAL:", pd.read_sql("SELECT COUNT(*) as n FROM bloques", conn).iloc[0,0]) 
conn.close() 
