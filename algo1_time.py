import time
import random
import numpy as np
import pandas as pd

def creating_distribution(filename):
    start_time = time.time()
    d = {}
    count = 0
    with open(filename, 'r') as file:
        curr_date = None 
        for line in file:
            if line[0] == 'Q':
                count += 1
                phrase = line.split('Q')[1].lstrip()
                if phrase in d:
                    d[phrase] += 1
                else:
                    d[phrase] = 1
    print("--- %s seconds ---" % (time.time() - start_time), count)
    return d, count


# d_2009_04, num_phrases_2009_04 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2009-04.txt')
d_2009_03, num_phrases_2009_03 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2009-03.txt')
d_2009_02, num_phrases_2009_02 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2009-02.txt')
# d_2009_01, num_phrases_2009_01 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2009-01.txt')
d_2008_12, num_phrases_2008_12 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-12.txt')
d_2008_11, num_phrases_2008_11 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-11.txt')
# d_2008_10, num_phrases_2008_10 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-10.txt')
# d_2008_09, num_phrases_2008_09 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-09.txt')
# d_2008_08, num_phrases_2008_08 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-08.txt')

def lp_norm(d, p):
    s = 0
    for i in d:
        s += abs(d[i])**p
    return s**(1/p)

def lp_distance(d1, d2, p):
    s = 0
    for i in d1:
        if i in d2:
            s += abs(d1[i]-d2[i])**p
        else:
            s += abs(d1[i])**p
    for i in d2:
        if i not in d1:
            s += abs(d2[i])**p
    return s**(1/p)

def list_to_distribution(arr):
    d = {} # maps from word to count
    for word in arr:
        if word in d:
            d[word]+=1
        else:
            d[word]=1
    return d

def namestr(obj, namespace=globals()):
    return [name for name in namespace if namespace[name] is obj][0]


def l2_norm_estimate(d_p_full, d_q_full, samples, fraction = 2500):
    ### processing distributions
    d_p_multiple = {key:val for key, val in d_p_full.items() if val > 1}
    d_q_multiple = {key:val for key, val in d_q_full.items() if val > 1}
    p_total = sum(d_p_multiple.values())
    q_total = sum(d_q_multiple.values())

    d_p = {key:val/p_total for key, val in d_p_multiple.items()}
    d_q = {key:val/q_total for key, val in d_q_multiple.items()}
    
    
    ### setting error and sample parameters
    d_p_l2norm = lp_norm(d_p, 2)
    d_q_l2norm = lp_norm(d_q, 2)
    b = max(d_p_l2norm**2, d_q_l2norm**2)

    pq_l4norm = lp_distance(d_p, d_q, 4)
    pq_l1norm = lp_distance(d_p, d_q, 1)
    e = pq_l1norm  # set e to be a factor of l1 distance
    #print(pq_l1norm)
    #print('yoyoyo', e)
    #m = (b**0.5*pq_l4norm**2/e**4 + b**0.5/e**2)  # set m to be a constant factor
    #print("old m is: ", m)
    
    avg_size = (p_total + q_total)//2
    #c = avg_size/(m*fraction)
    #m *= c
    m = samples
    print("new m is: ", m)
  
    
    ### sampling from distributions
    s = np.random.poisson(lam=m)
    S_p = random.choices(list(d_p.keys()), weights=d_p.values(), k=s)
    S_q = random.choices(list(d_q.keys()), weights=d_q.values(), k=s)

    S_p_d = list_to_distribution(S_p)
    S_q_d = list_to_distribution(S_q)

    
    ### calculating Z stat
    Z = 0
    for i in S_p_d:
        if i in S_q_d:
            Z += (S_p_d[i]-S_q_d[i])**2-S_p_d[i]-S_q_d[i]
        else:
            Z += S_p_d[i]**2-S_p_d[i]

    for i in S_q_d:
        if i not in S_p_d:
            Z += S_q_d[i]**2-S_q_d[i]
            
    
    ### output compare values
    d_p_l2norm = lp_distance(d_p, d_q, 2)
    return (Z/(m**2), d_p_l2norm**2, e**2, m)


d_p_full = d_2009_02
d_q_full = d_2009_03
results = []
# frac = [100, 250, 750, 1000, 2500, 10000, 25000, 75000, 100000]
# frac.reverse()
frac = [10**-10, 10**-9, 10**-8, 10**-7, 10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1]
results = []
# for i in frac:
#     count = 0
#     for i in range(50):
#         a,b,c,d = l2_norm_estimate(d_p_full, d_q_full, c=i)
#         if abs(a-b)<c:
#             count+=1
#     results.append(i, count/50, a, b, c, d)
#     print(i, count/50)
#     print(a,b,c,d)
#     results.append([a,b,c,d])
results = {}
samples = [120, 160, 480, 1200, 4800, 12000, 16000, 48000, 120000]
for i in samples:
    #total_time = 0
    #for i in range(1):
    start_time = time.time()
    a,b,c,d = l2_norm_estimate(d_p_full, d_q_full, i)
    time_spent = (time.time()-start_time)
    results[d] = time_spent
    print(d, time_spent)
    #print(d, total_time/10)

print(results)
