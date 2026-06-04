import pandas as pd

df = pd.read_excel('data/CFH_IAA_Ciego_84.xlsx', sheet_name='02_Anotacion', skiprows=[1])
col_texto = df.columns[3]
vacios = df[col_texto].isna() | (df[col_texto].astype(str).str.strip() == '')
print(f'Total fragmentos : {len(df)}')
print(f'Con texto        : {(~vacios).sum()}')
print(f'Sin texto        : {vacios.sum()}')
print(f'\nFragmentos sin texto:')
print(df[vacios][['ID', df.columns[1], df.columns[2]]].to_string())
