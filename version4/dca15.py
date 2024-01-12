import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysam
import random

from convert import convert_hap_samples_to_dataframe
import os
import math
from copy import deepcopy
import numba
import os
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects import numpy2ri
import subprocess
import json

import warnings
warnings.filterwarnings('ignore')

K=10

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

'''
# def select_locus(vcf, n, father, mother, child):
#     # 从vcf中随机抽出父母和孩子n行的数据
#     df = pd.DataFrame(columns=['locus','moGT','faGT', 'chGT','f0'])
#     for record in vcf:   # 先将所有数据读进df
#         locus = (record.chrom,str(record.pos))
#         f0 = 1-record.info['AF'][0]
#         if f0<1:  # 除去f0=1的情况，不然会导致之后 arctanh(1)=inf
#             mo = record.samples[mother]['GT']
#             ch = record.samples[child]['GT']
#             if (mo!=(0,1) and mo!=(1,0)) or (ch!=(0,1) and ch!=(1,0)):
#                 fa = record.samples[father]['GT']
#                 df.loc[len(df.index)] = [locus,mo,fa,ch,f0] # type: ignore
#     num_row = df.shape[0]
#     assert num_row>=n,'not enough locus to select'
#     df_select = df.sample(n,replace=False,axis=0)
#     df_select.set_index('locus',drop=True,inplace=True)
#     return df_select

# def select_locus(vcf, n, father, mother, child):
#     # 从vcf中随机抽出<<连续的父母>>和孩子n行的数据
#     df = pd.DataFrame(columns=['locus','moGT','faGT', 'chGT','f0'])
#     for record in vcf:   # 先将所有数据读进df
#         locus = (record.chrom,str(record.pos))
#         f0 = 1-record.info['AF'][0]
#         if f0<1:  # 除去f0=1的情况，不然会导致之后 arctanh(1)=inf
#             mo = record.samples[mother]['GT']
#             ch = record.samples[child]['GT']
#             if (mo!=(0,1) and mo!=(1,0)) or (ch!=(0,1) and ch!=(1,0)):
#                 fa = record.samples[father]['GT']
#                 df.loc[len(df.index)] = [locus,mo,fa,ch,f0] # type: ignore
#     num_row = df.shape[0]
#     assert num_row>=n,'not enough locus to select'
#     i = random.randint(n, num_row)
#     df_select = df.iloc[i-n:i,:]
#     df_select.set_index('locus',drop=True,inplace=True)
#     return df_select
'''

def calcul_XY(df_select):
    '''遍历选择的位点计算X,Y和c, 返回logX,logY and c'''
    log_mu = -8 # 默认突变概率为1e-8
    log2 = np.log10(2)

    n = df_select.shape[0]
    log_x = 0
    log_y = 0
    c = []
    
    for i in range(n):
        father = df_select['faGT'][i]
        mother = df_select['moGT'][i]
        child = df_select['chGT'][i]
        if child == (0,0):
            if mother == (0,1) or mother == (1,0):
                log_y = log_y - log2
                c.append(0)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - 2*log2
                elif father == (0,0):
                    log_x = log_x - log2
                else:
                    log_x = log_x + log_mu - log2

            elif mother == (0,0):
                c.append(0)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    pass
                else:
                    log_x = log_x + log_mu

            else:
                log_y = log_y + log_mu
                c.append(0)
                if father == (0,1) or father == (1,0):
                    log_x = log_x + log_mu -log2
                elif father == (0,0):
                    log_x = log_x + log_mu
                else:
                    log_x = log_x + 2*log_mu

        elif child == (0,1) or child == (1,0):
            if mother == (0,1) or mother == (1,0):
                c.append(0)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    log_x = log_x - log2
                else:
                    log_x = log_x - log2
            #    assert 1==0,'mother and child are both heteGT'

            elif mother == (0,0):
                c.append(1)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    log_x = log_x + log_mu
                else:
                    pass

            else:
                c.append(0)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    pass
                else:
                    log_x = log_x + log_mu

        else:
            if mother == (0,1) or mother == (1,0):
                log_y = log_y - log2
                c.append(1)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - 2*log2
                elif father == (0,0):
                    log_x = log_x + log_mu - log2
                else:
                    log_x = log_x -log2

            elif mother == (0,0):
                log_y = log_y + log_mu
                c.append(1)
                if father == (0,1) or father == (1,0):
                    log_x = log_x + log_mu - log2
                elif father == (0,0):
                    log_x = log_x + 2* log_mu
                else:
                    log_x = log_x + log_mu

            else:
                c.append(1)
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    log_x = log_x + log_mu
                else:
                    pass
    c = np.array(c)
    # print('X,Y,len_c:',log_x,log_y,c.shape)
    return log_x,log_y,c


def calcul_pi_ind(df_select,log_x,log_y,c):
    '''计算独立情况的亲权系数'''
    f0 = df_select['f0'].values
    pr_ind = (np.log10(np.where(c==0,f0,1-f0))).sum() # type: ignore
    return log_x - log_y - pr_ind

@numba.jit(nopython=True)
def freq2(x,y):
    '''返回bool列表,True表示x,y对应位置同时为0'''
    return (x==0)&(y==0)

def calcul_fij(df_data,fi,locus_list,n):
    num_row = df_data.shape[0]
    fij = np.zeros((n,n))
    for i in range(n):  # 因为fij对称，只遍历上三角矩阵
        col1 = df_data[locus_list[i]].to_numpy().astype(int)   # 这里类型的变换，是为了使用numba加速
        for j in range(i+1,n):
            col2 = df_data[locus_list[j]].to_numpy().astype(int)
            fij[i,j] = freq2(col1,col2).sum()/num_row  # 统计两列数据同时为０的概率
            fij[j,i] = fij[i,j]
    for i in range(n):
        fij[i,i] = fi[i]
    return fij

def freq_pc(df_data,locus_list,fi,n,pc):
    fij = calcul_fij(df_data,fi,locus_list,n)
    return (1-pc)*fi + pc/2,(1-pc)*fij + pc/4

def cMat(fi,fij,n,alpha):
    cmat = fij - fi*(fi.reshape(n,1))
    print('\t c的最小特征值',np.min(np.linalg.eigvals(cmat)))
    cmat = np.where(cmat>alpha,cmat,0)
    return cmat

def inv_eig(cmat):
    D,V = np.linalg.eig(cmat)
    lambda_inv= np.diag(list(map(lambda x: x.real/(0.001+x.real**2), D)))
    inv_C2 = np.dot(np.dot(V, lambda_inv), V.T).real
    return inv_C2

def invc_glassoR(cmat,rho1,rho2):
    c_path = '/home/anran/paternity/version4/cmat.txt'
    invc_path = '/home/anran/paternity/version4/invc.txt'
    invg_path = "/home/anran/paternity/version4/invc_glasso.csv"
    np.savetxt(c_path,cmat)
    np.savetxt(invc_path,inv_eig(cmat))
    r_path = f"/home/anran/paternity/version4/invc.R"
    r_command = ['Rscript',r_path]
    r_command.extend(map(str,[c_path, invc_path, invg_path,rho1, rho2]))
    subprocess.run(r_command,check=True)
    invc = pd.read_csv(invg_path,sep=' ').values
    if(os.path.isfile(c_path)):  # 删除保存的数据文件
        os.remove(c_path)
        os.remove(invc_path)
        os.remove(invg_path)
    return invc

def calcul_e(invc,fi,n):
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a = 2*fi[i]*fi[j]
            delta = 1-4*a*invc[i, j]
            if delta<0 or fi[i]==0 or fi[j]==0:
                res[i,j] = -invc[i,j]
            elif delta==0:
                res[i,j] =  -1/(2*a)
            else:
                s1 = (-1+math.sqrt(delta))/(2*a)
                s2 = (-1+math.sqrt(delta))/(2*a)
                if abs(s1-invc[i,j])>abs(s2-invc[i,j]):  # 当有2实根，选择距-invc较近的根
                    res[i,j] = s2
                else:
                    res[i,j] = s1
    print('\t emat:',res.min(),res.max())
    return res

def margin(fi,emat,n):
    e = deepcopy(emat)
    for i in range(n):  # h计算时，对j!=i的eij项操作
        e[i,i] = 0
    h = np.arctanh(fi) + (e**2 * fi.reshape(n,1) * (1-fi**2)).sum(axis=1) - (e*fi).sum(axis=1)
    print('\t h:',h.min(),h.max())
    return h

def get_b(k):
    '''根据k生成b, 维度为(2^k,k), 一行是一个样本'''
    num_rows = 2**k  # 矩阵b的行数
    b = np.zeros((num_rows, k), dtype=int)
    for i in range(num_rows):
        binary_representation = format(i, '0' + str(k) + 'b')  # 将整数i转换为k位的二进制字符串
        for j in range(k):
            b[i, j] = int(binary_representation[j])
    return b

def select_emat(seq,j,emat,k):
    '''对于第j个位点, 只考虑它及其前面的k-1个位点
    取出emat中的子方阵[j-k+1.j-k+1]->[j,j]
    根据seq计算对应的系数(1 or -1)'''
    s = np.tile(seq,(k,1))
    mask = np.logical_not(s ^ s.T)  # 00/11->1,01/10->0
    mask = np.where(mask==0,-1,mask)  
    return np.triu(emat[j-k+1:j+1,j-k+1:j+1]*mask,1)

def get_addition(j,k,b,e,h):
    '''返回E向量, 维度(2**k,)'''
    coef = np.zeros(2**k)
    for i in range(2**k):
        seq = b[i,:]
        mask = np.where(np.logical_not(seq[-1]^seq[:-1]),1,-1)
        coef[i] = (e[j,j-k+1:j] * mask).sum() + h[j]*(-1)**seq[-1]
    return coef

def calcul_Z(e,h,k,n):
    z = np.zeros((2**k,n-k+1))
    b = get_b(k)

    for i in range(2**k):  # 计算ｚ的第一列数据z[0]
        seq = b[i,:]
        z[i,0] = (h[:k]*np.where(seq==0,1,-1)).sum()+select_emat(seq,k-1,e,k).sum()
    
    for l in range(1,n-k+1): # 循环计算z[1]到z[n-k], l指向z的index, 对应e/h的index为 l+(k-1)
        coef = get_addition(l+k-1,k,b,e,h)
        for i in range(2**k):
            seq = b[i,:]
            l0 = (i-seq[-1]) // 2  # 二进制数右移移位，最高位为0
            l1 = (l0 + 2**(k-1)) # 二进制数右移移位，最高位为１
            z0 = min(z[l0,l-1],z[l1,l-1])
            z1 = max(z[l0,l-1],z[l1,l-1])
            if z1-z0>200:
                z[i,l] = coef[i] + z1
            else: 
                z[i,l] = coef[i] + z0 + np.log(np.exp(1)+np.exp(z1-z0))
    z_min = z[:,-1].min()
    log10_z = np.log10(np.exp(z[:,-1]-z_min).sum()) + np.log10(np.exp(1))*z_min
    return log10_z

def calcul_num(c, e, h, n, k):
    '''输入e维度N*N
    返回10的指数部分'''
    nums = np.zeros(n-k+1)
    sum_h = (h*np.where(c==0,1,-1)).sum()

    nums[0] = select_emat(c[:k],k-1,e,k).sum()

    for j in range(k,n):
        l = j-k+1  # nums中的index
        mask = np.where(np.logical_not(c[j]^c[l:j]),1,-1)  # 使用c_j与其前k-1个位点计算mask
        coef = (e[j,l:j] * mask).sum()
        nums[l] = coef + nums[l-1]
    return (nums[-1]+sum_h)*np.log10(np.exp(1))


# def get_fi(vcf_file,n):
#     '''select N locus from vcf_file'''
#     vcf = pysam.VariantFile(vcf_file)
#     f0=[]
#     for record in vcf:
#         f1 = record.info['AF'][0]
#         locus = (record.chrom,str(record.pos))
#         if f1<0.8 and f1>0.2:
#             f0.append((locus,1-f1))
#     l = random.randint(0,len(f0)-n)
#     freq0 = dict(f0[l:l+n])
#     fi = pd.Series(freq0)
#     return fi

def get_fi(df_data,start,end):
    f0 = {}
    totle = df_data.shape[0]
    columns = df_data.columns.to_list()
    print('linkage aera: ',columns[start],' to ', columns[end])
    for i in range(start,end-1):
        locus = columns[i]
        freq = (df_data[locus]==0).sum()/totle
        f0[locus] = freq
    fi = pd.Series(f0)
    return fi

def get_fi_ind(df_data,start,end,n):
    f0={}
    totle = df_data.shape[0]
    num_col = df_data.shape[1]
    # index_available = np.append(np.arange(start),np.arange(end+1,num_col))
    # print(index_available.shape)
    index = np.random.choice(np.append(np.arange(start),np.arange(end+1,num_col)),size=n,replace=False)
    columns = df_data.columns.to_list()
    for i in index:
        locus = columns[i]
        freq = (df_data[locus]==0).sum()/totle
        f0[locus] = freq
    fi = pd.Series(f0)
    return fi

'''
def get_data(locus_list,vcf, father, mother, child):
    # 根据locus_list抽取人的数据
    df = pd.DataFrame(columns=['locus','moGT','faGT', 'chGT','f0'])
    for record in vcf:   # 先将所有数据读进df
        locus = (record.chrom,str(record.pos))
        f0 = 1-record.info['AF'][0]
        if locus in locus_list:  # 除去f0=1的情况，不然会导致之后 arctanh(1)=inf
            mo = record.samples[mother]['GT']
            ch = record.samples[child]['GT']
            fa = record.samples[father]['GT']
            df.loc[len(df.index)] = [locus,mo,fa,ch,f0] # type: ignore
    df.set_index('locus',drop=True,inplace=True)
    return df
'''

def get_data(start,fi_serie,df_person, father, mother, child):
    '''根据locus_list抽取人的数据'''
    df = pd.DataFrame(columns=['locus','moGT','faGT', 'chGT','f0'])
    for locus in range(start,start+fi_serie.shape[0]):
        mo = df_person.iloc[mother,locus]
        fa = df_person.iloc[father,locus]
        ch = df_person.iloc[child,locus]
        f0 = fi_serie[locus-start]
        df.loc[len(df.index)] = [locus,mo,fa,ch,f0] # type: ignore
    df.set_index('locus',drop=True,inplace=True)
    return df


if __name__=="__main__":
    pc = 0.001
    alpha = 0.001
    rho1 = 0.5
    rho2 = 0.1
    data_path = "/home/anran/paternity/family-data/sim_seg1" 
    print(data_path)
    # 数据集信息
    hap_file,samples_file,_,_ = get_file(data_path)
    df_data_init = convert_hap_samples_to_dataframe(hap_file,samples_file)
    # df_data.drop(['ID','REF','ALT'],axis=0,inplace=True)
    df_data = pd.read_csv('/home/anran/paternity/family-data/sim_data/df_data.csv')
    df_data.drop(['personID'], axis=1,inplace=True)
    df_data.columns = df_data_init.columns
    
    # df_res_linkage = pd.DataFrame(columns=['N','trio','F','C','X','Y','log_PI_ind','log_PI_nind','seq_c'])
    # 家庭信息
    print('################################################# 强连锁结果')
    # 选择N个连续位点，并计算相关的相关系数，边际和配分函数
    with open('/home/anran/paternity/family-data/sim_data/df_data.json') as f:
        data = json.load(f)
    start = data['linkage_keys'][0]
    end = data['linkage_keys'][-1]

    with open('/home/anran/paternity/family-data/sim_data/family.json') as f:
        family = json.load(f)
    family2 = deepcopy(family)
    person = []
    for key in family2.keys():
        if key != 'linkage_keys':
            person.append([int(key)]+family[key])
    persons = pd.DataFrame(person,columns = ['child','father','mother'])
    print(persons)

    # person_data = pd.read_csv('/home/anran/paternity/family-data/sim_data2/person.csv')
       
    fi_serie = get_fi(df_data,start,end)
    N = fi_serie.shape[0]

    # df_c_linkage = pd.DataFrame(columns = np.arange(N))

    print(f"使用强连锁,计算25组数据,k=10,N={N}")
    fi_serie.to_csv(f'/home/anran/paternity/version4/result_debug/locus_linkage0.csv',index=False)
    locus_list = fi_serie.index.to_list()
    fi =  fi_serie.values
    # fij = calcul_fij(df_data,fi,locus_list,N)
    fi,fij = freq_pc(df_data,locus_list,fi,N,pc)
    print('fi and fij:',fi[0],fij[0,0])
    cmat = cMat(fi,fij,N,alpha)  
    invc = invc_glassoR(cmat,rho1,rho2)
    diag = ((invc-np.diag(np.diagonal(invc)))==0).all()  # 判断逆矩阵是否为对角矩阵
    emat = calcul_e(invc,fi,N) 
    h = margin(fi,emat,N)
    log_z = calcul_Z(emat,h,K,N)  

    plt.figure(figsize=(15,15))
    plt.subplot(2, 2, 1) 
    plt.hist(fi) 
    plt.title('fi')

    plt.subplot(2, 2, 2)  
    plt.hist(fij,bins=10) 
    plt.title('fij')

    plt.subplot(2, 2, 3)  
    plt.hist(emat,bins=10) 
    plt.title('emat')

    plt.subplot(2, 2, 4)  
    plt.hist(h) 
    plt.title('h')
    plt.savefig('/home/anran/paternity/version4/result_debug/figure.png')

    res = pd.DataFrame(columns=['father','mother','child','c','log_x','log_y','log_fi','num','z','pi_ind','pi_nind'])
    for fa in [(0,0),(0,1),(1,1)]:
        for ch in [(0,0),(0,1),(1,1)]:
            for mo in [(0,0),(0,1),(1,1)]:
                no_father1 = pd.DataFrame(columns=['moGT','faGT','chGT','f0'])
                no_father1.faGT = [fa]*N
                no_father1.moGT = [mo]*N
                no_father1.chGT = [ch]*N
                no_father1.f0 = fi
                # display(no_father1.head())
                log_x,log_y,c = calcul_XY(no_father1)
                # print('log_x,log_y',log_x,log_y)
                sum_c = (c == 1).sum()
                prod_fi = (np.log10(np.where(c==0,fi,1-fi))).sum()
                # print('fi连乘:',(np.log10(np.where(c==0,f0,1-f0))).sum())
                log_pi_ind = calcul_pi_ind(no_father1,log_x,log_y,c)
                log_num = calcul_num(c,emat,h,N,K)
                # print('分子:',log_num)
                log_pi_nind = log_x-log_y-log_num+log_z
                # print('pi_ind:',log_pi_ind)
                # print('pi_nind:',log_pi_nind)
                res = res._append({'father':fa,
                                'c':sum_c,
                                'mother':mo,
                                'child':ch,
                                'log_x':log_x,
                                'log_y':log_y,
                                'log_fi':prod_fi,
                                'num':log_num,
                                'z':log_z,
                                'pi_ind':log_pi_ind,
                                'pi_nind':log_pi_nind}, ignore_index=True) # type: ignore
    res.to_csv('/home/anran/paternity/version4/result_debug/final_result.csv')

    # for i in range(5):
    #     for j in range(5):
    #         print("-----------------------------",N,i,j)
    #         df_select = get_data(start,fi_serie,person_data,persons.father[i],persons.mother[j],persons.child[j])
    #         df_select.set_index(np.arange(df_select.shape[0]),drop=True,inplace=True)
    #         log_x,log_y,c = calcul_XY(df_select)
    #         df_c_linkage.loc[len(df_c_linkage.index)] = c
    #         print('\tlog_x,log_y:',log_x,log_y)
    #         log_pi_ind = calcul_pi_ind(df_select,log_x,log_y,c)
    #         log_num = calcul_num(c,emat,h,N,K)
    #         log_pi_nind = log_x-log_y-log_num+log_z
    #         print("pi_ind,pi_nind:",log_pi_ind,log_pi_nind)
    #         df_res_linkage = df_res_linkage._append({'F':i,
    #                                 'N':N,
    #                                 'trio':(df_select.faGT[0],df_select.moGT[0],df_select.chGT[0]),
    #                                 'C':j,
    #                                 'X':log_x,
    #                                 'Y': log_y,
    #                                 'seq_c':(c==0).sum(),
    #                                 'log_PI_ind': log_pi_ind,
    #                                 'log_PI_nind': log_pi_nind}, ignore_index=True) # type: ignore
    # df_res_linkage.to_csv("/home/anran/paternity/version4/result_debug3/linkage0.csv",index=False)
    # (df_c_linkage.T).to_csv("/home/anran/paternity/version4/result_debug3/c_linkage0.csv",index=False)

# 1205 1526

    # print('################################################# 位点独立结果')
    # df_res_indepen = pd.DataFrame(columns=['N','K','F','C','X','Y','log_PI_ind','log_PI_nind'])
    # fi_serie = get_fi_ind(df_data,start,end,N)
    # N = fi_serie.shape[0]
    # df_c_indepen = pd.DataFrame(columns = np.arange(N))
    # print(f"使用独立的位点,计算25组数据,k=10,N={N}")
    # fi_serie.to_csv(f'/home/anran/paternity/version4/result_debug/locus_ind0.csv')
    # locus_list = fi_serie.index.to_list()
    # fi =  fi_serie.values
    # # fij = calcul_fij(df_data,fi,locus_list,N)
    # fi,fij = freq_pc(df_data,locus_list,fi,N,pc)
    # cmat = cMat(fi,fij,N,alpha)  
    # invc = invc_glassoR(cmat,rho1,rho2)
    # diag = ((invc-np.diag(np.diagonal(invc)))==0).all()  # 判断逆矩阵是否为对角矩阵
    # if (np.abs(invc)).max()>20:
    #     file_name = f'/home/anran/paternity/version4/locusForInvcError.csv'
    #     fi_serie.to_csv(file_name)
    #     assert 1==0,'invc max value too large'
    # emat = calcul_e(invc,fi,N) 
    # h = margin(fi,emat,N)
    # log_z = calcul_Z(emat,h,K,N)  

    # for i in range(5):
    #     for j in range(5):
    #         print("-----------------------------",N,i,j)
    #         vcf = pysam.VariantFile(vcf_file)
    #         df_select = get_data(locus_list,vcf,persons.father[i],persons.mother[j],persons.child[j])
    #         log_x,log_y,c = calcul_XY(df_select)
    #         df_c_indepen.loc[len(df_c_indepen)] = c
    #         print('\tlog_x,log_y:',log_x,log_y)
    #         log_pi_ind = calcul_pi_ind(df_select,log_x,log_y,c)
    #         log_num = calcul_num(c,emat,h,N,K)
    #         log_pi_nind = log_x-log_y-log_num+log_z
    #         print("pi_ind,pi_nind:",log_pi_ind,log_pi_nind)
    #         df_res_indepen = df_res_indepen._append({'F':i,
    #                                 'N':N,
    #                                 'K':K,
    #                                 'C':j,
    #                                 'X':log_x,
    #                                 'Y': log_y,
    #                                 'log_PI_ind': log_pi_ind,
    #                                 'log_PI_nind': log_pi_nind}, ignore_index=True) # type: ignore
    # df_res_indepen.to_csv("/home/anran/paternity/version4/result_debug/indepen0.csv")
    # (df_c_indepen.T).to_csv("/home/anran/paternity/version4/result_debug/c_indep0.csv")
