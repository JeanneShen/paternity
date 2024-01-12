import random
from re import match
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
    parser.add_argument("-c", "--num_couples", type=int, default=10, help="Number of initial couples (default: 10)")
    parser.add_argument("-l", "--num_linkages", type=int, default=1000, help="Number of consecutive non-recombining positions (default: 1000)")
    parser.add_argument("-p", "--num_keys", type=int, default=0, help="key points for linkage interval (default: 0)")
    parser.add_argument("-k", "--num_kids", type=int, default=2, help="Number of child for every couple (default: 2)")
    parser.add_argument("-r", "--num_recombinations", type=int, default=5, help="Number of intervals for recombination (default: 5)")

    return parser.parse_args()

def choice_person(df_data,num_couples,linkage_keys):
    start = linkage_keys[0]
    end = linkage_keys[-1]
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
    for i in range(res0.shape[0]):
        res0.iloc[i,start:end] = res0.iloc[i,start]
    return res0,select_people

def generate_linkage_intervals(num_linkage,num_locus,num_keys=0):
    linkage_init = random.randint(0,num_locus-num_linkage)
    keys = np.random.choice(np.arange(1,num_linkage-1), num_keys,replace=False) + linkage_init
    linkage_keys = np.sort(np.append(keys,[linkage_init,linkage_init+num_linkage+1]))
    return linkage_keys

def generate_recombination_intervals(num_recombinations,num_locus,linkage_keys):
    start = linkage_keys[0]
    end = linkage_keys[-1]
    num_linkage = end-start
    interval1 = range(0,start)
    interval2 = range(end+1,num_locus+1)
    points1 = start*num_recombinations//(num_locus-num_linkage)
    points2 = num_recombinations - points1
    recombination_points1 = sorted(random.sample(interval1, 2 * points1)) 
    recombination_points2 = sorted(random.sample(interval2, 2 * points2))
    recombination_points = recombination_points1 + recombination_points2
    recombination_intervals = [(recombination_points[i], recombination_points[i + 1]) for i in range(0, len(recombination_points), 2)]
    return recombination_intervals

def is_variant_in_recombination_intervals(variant_pos, recombination_intervals):
    '''检查一个变异位点是否在给定的重组区间内'''
    if recombination_intervals!=[]:
        for start, end in recombination_intervals:
            if start <= variant_pos <= end:
                return True
    return False

def sim_meosis(res,num_recombinations,linkage_keys):
    num_locus = res.shape[1]-1
    start = linkage_keys[0]
    end = linkage_keys[-1]
    temp = deepcopy(res[res.index%2==0])
    for i in range(temp.shape[0]):
        if num_recombinations>0:
            recombination_intervals = generate_recombination_intervals(num_recombinations,num_locus,linkage_keys)
        else:
            recombination_intervals = []
        for j in range(num_locus):
            if j<start or j>=end:
                if is_variant_in_recombination_intervals(j,recombination_intervals):
                    temp.iloc[i,j] = res.iloc[i+1,j]
            # else:
            #     # index = np.searchsorted(linkage_keys,j,side='right') - 1   # type: ignore
            #     temp.iloc[i,j] = res.iloc[i,start]

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

def valid_linkage(df,linkage_keys):
    for i in range(len(linkage_keys)-1):
        start = linkage_keys[i]
        end = linkage_keys[i+1]
        a = (df.iloc[:,start]=='0').sum()
        for j in range(start,end):
            if (df.iloc[:,j]=='0').sum() != a:
                print('不满足强连锁，列数：',i)
                return False
    return True
                
def genetype(df_family):
    family = df_family.drop(['personID'], axis=1)
    person_data = pd.DataFrame(columns = np.arange(family.shape[1]))
    print(family.shape,person_data.shape)
    for j in range(family.shape[0]//2):
        print(j)
        person_data.loc[j] = list(zip(family.iloc[2*j].values.astype(int),family.iloc[2*j+1].values.astype(int)))

    # person_data.to_csv('/home/anran/paternity/family-data/sim_data3/person.csv',index=False)
    # person = []
    # for key in dict_family.keys():
    #     if key != 'linkage_keys':
    #         person.append([int(key)]+list(map(int,family[key])))
    # persons = pd.DataFrame(person,columns = ['child','father','mother'])
    # persons.to_csv('/home/anran/paternity/family-data/sim_data/ped.csv',index=False)
    return person_data

def dataset(df_data,linkage_keys,args):
    num_generation = args.num_generations
    num_couple = args.num_couples
    num_child = args.num_kids
    num_recombination = args.num_recombinations
    result_df = pd.DataFrame()

    res0,family = choice_person(df_data,num_couple,linkage_keys)
    assert valid_linkage(res0,linkage_keys),'dataset select data non linkage'
    result_df = pd.concat([result_df,res0])
    personID = 2*num_couple
    for g in range(num_generation):
        print('#####################',g)
        couple = []
        for k in range(num_child):
            print('#########',k)
            temp = sim_meosis(res0,num_recombination,linkage_keys)
            persons_rest = list(map(int,list(temp['personID'].values)))
            # print(temp.shape,persons_rest)
            if k==0:
                while persons_rest!=[]:
                    persons_rest,parent1,parent2 = select_parent(persons_rest,family)
                    # print('-----------------',parent1,parent2,persons_rest)
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
        res0 = result_df.iloc[2*personID-4*num_couple:2*personID]
        res0.set_index(np.arange(res0.shape[0]),drop=True,inplace=True)

    assert valid_linkage(result_df,linkage_keys),'dataset final data non linkage'
    family['linkage_keys'] = list(map(int,linkage_keys))
    result_df.to_csv(args.output_file_path + 'df_data.csv',index=False)
    with open(args.output_file_path+"df_data.json","w",encoding='utf-8') as f:
        json.dump(family,f)
    return 0

def test_dataset(df_data,linkage_keys,args):
    num_couple = 5
    num_recombination = args.num_recombinations
    result_df = pd.DataFrame()
    family = {}

    res0,_ = choice_person(df_data,num_couple,linkage_keys)
    assert valid_linkage(res0,linkage_keys),'family select data non linkage'
    result_df = pd.concat([result_df,res0])
    personID = 2*num_couple
   
    temp = sim_meosis(res0,num_recombination,linkage_keys)
    persons_rest = list(map(int,list(temp['personID'].values)))
    # print(temp.shape,persons_rest)

    while persons_rest!=[]:
        persons_rest,parent1,parent2 = select_parent(persons_rest,family)
        print('-----------------',parent1,parent2,persons_rest)
        family[personID] = [parent1,parent2]
        res_temp = deepcopy(temp[temp['personID'].isin([parent1,parent2])])
        res_temp['personID'] = [personID,personID]
        result_df = pd.concat([result_df,res_temp])
        personID += 1

    assert valid_linkage(result_df,linkage_keys),'family final data non linkage'
    # result_df.to_csv(args.output_file_path + 'family.csv',index=False)
    with open(args.output_file_path+"family.json","w",encoding='utf-8') as f:
        json.dump(family,f)
    person_data = genetype(result_df)
    person_data.to_csv(args.output_file_path + 'person.csv',index=False)
    return 0



def main():
    args = parse_args()
    num_linkage = args.num_linkages
    num_keys = args.num_keys

    hap_file,samples_file,_,_ = get_file(args.dataset)
    df_data = convert_hap_samples_to_dataframe(hap_file,samples_file)
    df_data.drop(['ID','REF','ALT'],axis=0,inplace=True)
    num_locus = df_data.shape[1]
    
    linkage_keys = generate_linkage_intervals(num_linkage,num_locus,num_keys)
    dataset(df_data,linkage_keys,args)
    test_dataset(df_data,linkage_keys,args)

    return 0

if __name__ == '__main__':
    main()