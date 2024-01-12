import pandas as pd
import numpy as np
import gzip
# import argparse

def rename_duplicates(df):
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        count = sum(cols == dup)
        mask = cols == dup
        cols.loc[mask] = [f'{dup}_{i}' if i != 0 else dup for i in range(count)]
    df.columns = cols
    return df

def rename_duplicate_index(df):
    index = df.index.tolist()
    seen = {}
    for i, idx in enumerate(index):
        if idx not in seen:
            seen[idx] = 0
        else:
            seen[idx] += 1
            index[i] = (idx[0], f'{idx[1]}_{seen[idx]}')
    df.index = index
    return df


def zero(df,locus):
    data = df[locus].iloc[4:]
    index = np.arange(data.shape[0]//2)
    data1 = df.iloc[2*index].values
    data2 = df.iloc[2*index+1].values
    return (data1==data2).sum()


def convert_hap_samples_to_dataframe(hap_file, samples_file):
    # Read the .hap.gz file
    with gzip.open(hap_file, 'rt') as f:
        hap_data = f.readlines()

    # Split each line into a list of strings
    hap_data = [line.strip().split() for line in hap_data]

    # Read the .samples file
    with open(samples_file, 'r') as f:
        samples = f.readlines()
    # print(samples)

    # Ignore the header and get sample names from the first column
    sample_names = [line.strip().split()[0] for idx, line in enumerate(samples) if idx not in (0,1)]
    # print(sample_names)

    # Create the DataFrame
    df = pd.DataFrame(hap_data)

    # Set column and row names
    df.columns = ['POS', 'ID', 'POS', 'REF', 'ALT'] + [f'{name}_{i}' for name in sample_names for i in range(2)]
    df.index = df['POS']
    # Drop the extra POS column
    df = df.drop(columns=['POS'])
    df = rename_duplicate_index(df)

    # print(df)

    df = df.T
    # print(df)

    return df

    


def count_combinations(df, col1, col2):
    # 获取指定的两列
    data = df[[col1, col2]]
    # print(data)

    # # 计算四种可能的组合的出现次数
    # freq = np.zeros((2,2))
    # freq[0,0] = ((data[col1] == '0') & (data[col2] == '0')).sum()/6404
    # freq[0,1] = fi.loc[col1,'freq0'] - freq[0,0]
    # freq[1,0] = fi.loc[col2,'freq0'] - freq[0,0]
    # freq[1,1] = 1-freq.sum()

    counts = {
        ('0', '0'): ((data[col1] == '0') & (data[col2] == '0')).sum(),
        ('1', '1'): ((data[col1] == '1') & (data[col2] == '1')).sum(),
        ('0', '1'): ((data[col1] == '0') & (data[col2] == '1')).sum(),
        ('1', '0'): ((data[col1] == '1') & (data[col2] == '0')).sum()
    }

    # 计算总的组合数量/ all haps count
    total = 3202*2

    # 计算每种组合的频率
    frequencies = {k: v / total for k, v in counts.items()}

    return frequencies

# def main():
#     parser = argparse.ArgumentParser(description='Convert .hap.gz and .samples files to a DataFrame.')
#     parser.add_argument('hap_file', type=str, help='The path to the .hap.gz file')
#     parser.add_argument('samples_file', type=str, help='The path to the .samples file')
#     args = parser.parse_args()

#     df = convert_hap_samples_to_dataframe(args.hap_file, args.samples_file)
#     # print(df.columns)
#     counts, frequencies = count_combinations(df, ('chr6','28500797'), ('chr6','28500830'))
#     print("Counts: ", counts)
#     print("Frequencies: ", frequencies)
