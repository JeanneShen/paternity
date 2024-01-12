import random
import pysam
import numpy as np
import pandas as pd
import os
import argparse
import json
from copy import deepcopy

from convert import convert_hap_samples_to_dataframe

def get_file(data_path):
    '''从文件夹中，得到各文件的路径'''
    for filename in os.listdir(data_path):
        if filename.endswith('.gz'):
            hap_file = os.path.join(data_path,filename)
        if filename.endswith('.sample'):
            samples_file = os.path.join(data_path,filename)
        if filename.endswith('.vcf'):
            vcf_file = os.path.join(data_path,filename)
        if filename.endswith('.ped'):
            ped_file = os.path.join(data_path,filename)
    assert (hap_file and samples_file and vcf_file and ped_file),'missing data file'
    return  hap_file,samples_file,vcf_file,ped_file

def parse_args():
    parser = argparse.ArgumentParser(description="Simulate generations of families and populations")
    parser.add_argument("dataset", help="Input data file with initial individuals")
    parser.add_argument("output_file_path", help="path for saving simulation files ")
    parser.add_argument("-g", "--num_generations", type=int, default=3, help="Number of generations to simulate (default: 3)")
    parser.add_argument("-c", "--num_couples", type=int, default=10, help="Number of initial couples (default: 20)")
    parser.add_argument("-l", "--num_linkages", type=int, default=1000, help="Number of consecutive non-recombining positions (default: 1000)")
    parser.add_argument("-k", "--num_kids", type=int, default=2, help="Number of child for every couple (default: 2)")
    parser.add_argument("-r", "--num_recombination", type=int, default=5, help="Number of intervals for recombination (default: 5)")

    return parser.parse_args()

def choice_person(df_data,num_couples):
    res0 = pd.DataFrame()
    select_people = {}
    num_people = df_data.shape[0]//2
    person_list = df_data.index.tolist()
    df_data.columns = np.arange(df_data.shape[1])
    for i in range(2*num_couples):
        index = random.randint(0,num_people)
        select_people[i] = person_list[2*index]
        select_data = deepcopy(df_data.iloc[index*2:index*2+2])
        select_data['personID'] = [i,i]
        res0 = pd.concat([res0,select_data])
    res0.set_index(np.arange(4*num_couples),drop=True,inplace=True)
    return res0,select_people

def generate_linkage_intervals(num_linkage,num_locus):
    '''生成给定数量的重组间隔，即随机选择一定数量的变异位点作为重组点，形成相邻两个变异位点之间的重组区间。'''
    linkage_init = random.randint(0,num_locus-num_linkage)
    linkage_interval = np.arange(num_linkage)+linkage_init
    return linkage_interval

def generate_recombination_intervals(num_recombinations,num_locus,linkage_interval):
    num_linkage = len(linkage_interval)
    interval1 = range(0,linkage_interval[0])
    interval2 = range(linkage_interval[-1]+1,num_locus+1)
    points1 = linkage_interval[0]*num_recombinations//(num_locus-num_linkage)
    points2 = num_recombinations - points1
    recombination_points1 = sorted(random.sample(interval1, 2 * points1)) 
    recombination_points2 = sorted(random.sample(interval2, 2 * points2))
    recombination_points = recombination_points1 + recombination_points2
    recombination_intervals = [(recombination_points[i], recombination_points[i + 1]) for i in range(0, len(recombination_points), 2)]
    return recombination_intervals

def is_variant_in_recombination_intervals(variant_pos, recombination_intervals):
    '''检查一个变异位点是否在给定的重组区间内'''
    for start, end in recombination_intervals:
        if start <= variant_pos <= end:
            return True
    return False

def sim_meosis(res,num_recombinations,linkage_interval):
    num_locus = res.shape[1]-1
    temp = deepcopy(res[res.index%2==0])
    for i in range(temp.shape[0]):
        recombination_intervals = generate_recombination_intervals(num_recombinations,num_locus,linkage_interval)
        for j in range(num_locus):
            if is_variant_in_recombination_intervals(j,recombination_intervals):
                temp.iloc[i,j] = res.iloc[i+1,j]
    return temp

def couple_legal(person1,person2,family=None):
    if not family:
        return True
    else:
        if person1 in family.keys() and person2 in family.keys():
            if family[person1] == family[person2]:
                return False
            else:
                return True
        else:
            return True

def select_parent(person_rest,family):
    persons = deepcopy(person_rest)
    parent1,parent2 = random.sample(persons,2)
    while not couple_legal(parent1,parent2,family):
        parent1,parent2 = random.sample(persons,2)
    persons.remove(parent1)
    persons.remove(parent2)
    return persons,parent1,parent2

def main():
    args = parse_args()

    num_generation = args.num_generations
    num_couple = args.num_couples
    num_child = args.num_kids
    num_linkage = args.num_linkages
    num_recombination = args.num_recombination

    result_df = pd.DataFrame()

    hap_file,samples_file,_,_ = get_file(args.dataset)
    df_data = convert_hap_samples_to_dataframe(hap_file,samples_file)
    df_data.drop(['ID','REF','ALT'],axis=0,inplace=True)
    num_locus = df_data.shape[1]
    
    linkage_interval = generate_linkage_intervals(num_linkage,num_locus)
    # 初代的人数据
    res0,family = choice_person(df_data,num_couple)
    print(family)
    result_df = pd.concat([result_df,res0])
    personID = 2*num_couple
    # print(res0.shape,result_df.shape,personID)

    for g in range(num_generation):
        print('#####################',g)
        couple = []
        for k in range(num_child):
            print('#########',k)
            temp = sim_meosis(res0,num_recombination,linkage_interval)
            persons_rest = list(map(int,list(temp['personID'].values)))
            # print(temp.shape,persons_rest)
            if k==0:
                while persons_rest!=[]:
                    persons_rest,parent1,parent2 = select_parent(persons_rest,family)
                    print('-----------------',parent1,parent2,persons_rest)
                    couple.append((parent1,parent2))
                    family[personID] = [parent1,parent2]
                    res_temp = deepcopy(temp[temp['personID'].isin([parent1,parent2])])
                    res_temp['personID'] = [personID,personID]
                    result_df = pd.concat([result_df,res_temp])
                    personID += 1
            else:
                for (parent1,parent2) in couple:
                    family[personID] = [parent1,parent2]
                    res_temp = deepcopy(temp[temp['personID'].isin([parent1,parent2])])
                    res_temp['personID'] = [personID,personID]
                    result_df = pd.concat([result_df,res_temp])
                    personID += 1
        #     display(result_df)
        res0 = result_df.iloc[2*personID-4*num_couple:2*personID]
        res0.set_index(np.arange(res0.shape[0]),drop=True,inplace=True)
    family['start'] = int(linkage_interval[0])
    family['end'] = int(linkage_interval[-1])
    result_df.to_csv(args.output_file_path + 'data3.csv')
    with open(args.output_file_path+"info3.json","w",encoding='utf-8') as f:
        json.dump(family,f)

if __name__ == '__main__':
    main()