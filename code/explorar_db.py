import sqlite3
import pandas as pd

conn = sqlite3.connect('cfh.db')

print('--- bloques ---')
print(pd.read_sql('PRAGMA table_info(bloques)', conn)[['name','type']].to_string())

print('\n--- anotaciones ---')
print(pd.read_sql('PRAGMA table_info(anotaciones)', conn)[['name','type']].to_string())

print('\n--- anotaciones muestra ---')
print(pd.read_sql('SELECT * FROM anotaciones LIMIT 3', conn).to_string())

conn.close()
