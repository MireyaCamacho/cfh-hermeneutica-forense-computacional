import sqlite3
import pandas as pd

conn = sqlite3.connect('cfh.db')

# Verificar si inner_id corresponde al id de bloques
print('--- bloques primeras filas (id, texto truncado) ---')
df_b = pd.read_sql('SELECT id, documento_id, seccion, texto FROM bloques LIMIT 5', conn)
df_b['texto'] = df_b['texto'].str[:80]
print(df_b.to_string())

print('\n--- anotaciones inner_ids únicos (primeros 10) ---')
df_a = pd.read_sql('SELECT DISTINCT inner_id, etiquetas_combinadas FROM anotaciones LIMIT 10', conn)
print(df_a.to_string())

print('\n--- JOIN anotaciones + bloques por inner_id = bloques.id ---')
df_join = pd.read_sql('''
    SELECT DISTINCT a.inner_id, a.etiquetas_combinadas,
           b.id as bloque_id, b.seccion,
           substr(b.texto, 1, 100) as texto_muestra
    FROM anotaciones a
    LEFT JOIN bloques b ON a.inner_id = b.id
    LIMIT 5
''', conn)
print(df_join.to_string())

print('\n--- ¿cuántos inner_id tienen match en bloques? ---')
df_match = pd.read_sql('''
    SELECT COUNT(DISTINCT a.inner_id) as con_match
    FROM anotaciones a
    INNER JOIN bloques b ON a.inner_id = b.id
''', conn)
print(df_match.to_string())

conn.close()
