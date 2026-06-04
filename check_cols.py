import sqlite3 
conn = sqlite3.connect('cfh.db') 
cursor = conn.cursor() 
cursor.execute("PRAGMA table_info(bloques)") 
print([r[1] for r in cursor.fetchall()]) 
conn.close() 
