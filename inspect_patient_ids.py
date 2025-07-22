import pandas as pd

df = pd.read_csv('data/BangladeshcombineddatasetJan252022.csv')

def extract_id_type(study_id):
    parts = str(study_id).split('-')
    if len(parts) == 4:
        patient_id = '-'.join(parts[:3])
        sample_type = parts[3][0].upper()
        return patient_id, sample_type
    return None, None

df['patient_id'], df['sample_type'] = zip(*df['Study_ID'].map(extract_id_type))

# Drop rows where patient_id is None
valid = df['patient_id'].notnull()
df = df[valid]

# Group by patient_id and collect sample types
sample_types = df.groupby('patient_id')['sample_type'].apply(set)
both_types = sample_types.apply(lambda s: 'C' in s and 'H' in s)

print(f'Total unique patient IDs: {sample_types.shape[0]}')
print(f'Patients with both heel and cord: {both_types.sum()}')
print(f'Patients missing a sample type: {(~both_types).sum()}')

# Filter to only patients with both sample types
both_patients = set(sample_types[both_types].index)
df_both = df[df['patient_id'].isin(both_patients)]
print(f'Filtered dataframe shape: {df_both.shape}')
df_both.to_csv('data/Bangladeshcombineddataset_both_samples.csv', index=False)
print('Saved filtered dataframe to data/Bangladeshcombineddataset_both_samples.csv') 