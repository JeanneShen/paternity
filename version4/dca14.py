'''更新数据, 使用采样法计算Z'''
import numpy as np
import pandas as pd
import pysam
import random
from convert import convert_hap_samples_to_dataframe
import os
import math
from copy import deepcopy
import numba


def get_fi(vcf,n):
    '''select N locus from vcf_file'''
    f0=[]
    for record in vcf:
        f1 = record.info['AF'][0]
        locus = (record.chrom,str(record.pos))
        f0.append((locus,1-f1))
    print('len_set f',len(set(f0)))
    f = random.sample(set(f0), n)
    freq0 = dict(f)
    fi = pd.Series(freq0)
    return fi

def select_locus(vcf,locus_list,father,mother,child):
    '''根据选择的位点, 从vcf中抽出父母和孩子的数据'''
    df = pd.DataFrame(columns=['locus','mother','father', 'child'])
    list = deepcopy(locus_list)
    for record in vcf:
        locus = (record.chrom,str(record.pos))
        if locus in list:
            # if locus not in df["locus"]:
            df = df._append({'locus':locus,
                            'child': record.samples[child]['GT'],
                            'father': record.samples[father]['GT'],
                            'mother':record.samples[mother]['GT']}, ignore_index=True)
            list.remove(locus)
    df.set_index('locus', inplace=True)
    return df


def calcul_XY(df_family):
    '''遍历选择的位点计算X,Y和c, 返回logX,logY and c'''
    n = df_family.shape[0]
    log_x = 0
    log_y = 0
    c = []
    log2 = - np.log(2)
    for locus in range(n):
        log_mu = -8
        father = df_family['father'][locus]
        mother = df_family['mother'][locus]
        child = df_family['child'][locus]
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
                log_y = log_y - log2
                c.append(0)   # 这里不确定是０/１
                if father == (0,1) or father == (1,0):
                    log_x = log_x - log2
                elif father == (0,0):
                    log_x = log_x - log2
                else:
                    log_x = log_x - log2

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

# ---------------------------------------------独立情况
def calcul_pi_ind(df,log_x,log_y,c):
    '''计算独立情况的亲权系数'''
    f0 = df['freq0'].values
    pr_ind = (np.log10(np.where(c==0,f0,1-f0))).sum()
    return log_x - log_y - pr_ind


# ----------------------------------------------非独立情况
@numba.jit(nopython=True)
def freq2(x,y):
    return (x==0)&(y==0)

@numba.jit(nopython=True)
def freq2_diag(fij,fi):
    for i in range(len(fi)):
        fij[i,i] = fi[i]
    return fij

def calcul_fij(df_data,locus_list,fi,N):
    fij = np.zeros((N,N))
    for i in range(N):
        col1 = df_data[locus_list[i]].to_numpy().astype(int)
        for j in range(i+1,N):
            col2 = df_data[locus_list[j]].to_numpy().astype(int)
            fij[i,j] = freq2(col1,col2).sum()/6404
            fij[j,i] = fij[i,j]
    fij = freq2_diag(fij,fi)
    return fij

def freq_pc(df_data,locus_list,fi,n,pc):
    fij = calcul_fij(df_data,locus_list,fi,n)
    return (1-pc)*fi + pc/2,(1-pc)*fij + pc/4


def invC(fi_pc,fij_pc,N):
    '''计算线性的耦合矩阵, N*K '''
    # cmat = cMat(fi_pc,fij_pc)
    cmat = fij_pc - fi_pc*(fi_pc.reshape(N,1))
    D,V = np.linalg.eig(cmat)
    lambda_inv= np.diag(list(map(lambda x: x.real/(PC+x.real**2) if x>=1e-3 else 0, D))) #####
    inv_C = np.dot(np.dot(V, lambda_inv), np.linalg.inv(V)).real
    return inv_C


def calcul_e(fi,fij):
    n = fi.shape[0]
    invc = invC(fi,fij,n)
    res = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            a = 2*fi[i]*fi[j]
            delta = 1-4*a*invc[i, j]
            if delta==0:
                res[i,j] =  -1/(2*a)
            elif delta > 0:
                s1 = (-1+math.sqrt(delta))/(2*a)
                s2 = (-1+math.sqrt(delta))/(2*a)
                if abs(s1-invc[i,j])>abs(s2-invc[i,j]):  # 当有2实根，选择距invc较近的根
                    res[i,j] = s2
                else:
                    res[i,j] = s1
            else:
                # res.append([complex(-1,math.sqrt(-delta))/(2*a),complex(-1,-math.sqrt(-delta))/(2*a)])
                res[i,j] = invc[i,j]
    return res


def margin(fi,emat):
    '''计算非线性边际概率'''
    h = np.arctanh(fi) - np.dot(emat,fi) + fi*np.dot(emat**2,1-fi**2)
    return h


def select_emat(seq,emat,n):
    ''' 根据序列seq,抽取eij(si,sj),满足i<j '''
    s = np.tile(seq,(n,1))
    mask = np.logical_not(s ^ s.T)
    mask = np.where(mask==0,-1,mask)  # 00/11-> 1, 01/10->0
    return np.triu(emat*mask,k=1)


def calcul_Z(emat,h):
    l = np.zeros(10000)
    for i in range(10000):
        # seq_random = np.random.rand(N)
        # seq = (seq_random > fi.mean()).astype(int)
        seq = np.random.randint(2,size=N)
        l[i] = (h*np.where(seq==0,1,-1)).sum() + (select_emat(seq,emat,N)).sum()
    log_z = np.log10(np.exp(l).mean()) + np.log10(2.0)*N
    # log_z = l.mean() + N*np.log10(2.0)
    return log_z


def calcul_pi_nind(df_family,df_data,n,pc,x,y,c):
    fi =  df_family['freq0'].values
    locus_list = df_family.index
    fi,fij = freq_pc(df_data,locus_list,fi,n,pc)
    emat = calcul_e(fi,fij)
    h = margin(fi,emat)
    num = (h*np.where(c==0,1,-1)).sum() + select_emat(c,emat,n).sum()
    log_num = np.log10(np.exp(1))*num
    log_z = calcul_Z(emat,h)
    return x-y-log_num+log_z


def get_file(data_path):
    for filename in os.listdir(data_path):
        if filename.endswith('.gz'):
            hap_file = os.path.join(data_path,filename)
        if filename.endswith('.samples'):
            samples_file = os.path.join(data_path,filename)
        if filename.endswith('.vcf'):
            vcf_file = os.path.join(data_path,filename)
        if filename.endswith('.ped'):
            ped_file = os.path.join(data_path,filename)
    return  hap_file,samples_file,vcf_file,ped_file



if __name__=="__main__":
    N=5000
    PC=0.01
    data_path = "/home/anran/paternity/family-data/seg1" 
    hap_file,samples_file,vcf_file,ped_file=get_file(data_path)
    persons = pd.read_table(ped_file)
    df_data = convert_hap_samples_to_dataframe(hap_file,samples_file)
    df_data.drop(['ID','REF','ALT'],axis=0,inplace=True)

    vcf = pysam.VariantFile(vcf_file)
    fi = get_fi(vcf,N)
    print("fi",fi.shape)
    locus_list = fi.index.to_list()

    df_res = pd.DataFrame(columns=['F','C','X','Y','log_PI_ind','log_PI_nind'])
    for i in range(5):
        for j in range(5):
            print("-----------------------------",i,j)
            vcf = pysam.VariantFile(vcf_file)
            df_family = select_locus(vcf,locus_list,persons.father[i],persons.mother[j],persons.child[j])
            df_family['freq0'] = fi
            # print(persons.father[i],persons.mother[j],persons.child[j],df_family.shape)
            x,y,c = calcul_XY(df_family)
            log_pi_ind = calcul_pi_ind(df_family,x,y,c)
            log_pi_nind = calcul_pi_nind(df_family,df_data,N,PC,x,y,c)
            print("result",log_pi_ind,log_pi_nind)
            df_res = df_res._append({'F':i,
                            'C':j,
                            'X':x,
                            'Y': y,
                            'log_PI_ind': log_pi_ind,
                            'log_PI_nind': log_pi_nind}, ignore_index=True)
    df_res.to_csv("/home/anran/paternity/version4/results/pi_seg_5000.csv")