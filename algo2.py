import math
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

d_2008_12, num_phrases_2008_12 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-12.txt')
d_2008_11, num_phrases_2008_11 = creating_distribution('/storage/linx3/projects/meme_data/quotes_2008-11.txt')



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




def self_pairwise_collisions(arr):
    collisions = 0
    for i in range(len(arr)-1):
        for j in range(i+1, len(arr)):
            if arr[i] == arr[j]:
                collisions += 1
    return collisions


def across_list_collisions(arr1, arr2):
    collisions = 0
    for i in arr1:
        for j in arr2:
            if i == j:
                collisions += 1
    return collisions


def multidimensional_shifting(num_samples, sample_size, elements, probabilities):
    # replicate probabilities as many times as `num_samples`
    replicated_probabilities = np.tile(probabilities, (num_samples, 1))
    # get random shifting numbers & scale them correctly
    random_shifts = np.random.random(replicated_probabilities.shape)
    random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
    # shift by numbers & find largest (by finding the smallest of the negative)
    shifted_probabilities = random_shifts - replicated_probabilities
    return np.argpartition(shifted_probabilities, sample_size, axis=1)[:, :sample_size]

def random_sampling(elements, prob, m):
    return multidimensional_shifting(m, 1, elements, prob).T[0]

def l2_distance_tester(p, q, m, e, delta):
    num_reject = 0 
    m = round(m)
    rounds = round(math.log(1/delta)) # rounds is O(log(1/delta))
    p_elements = (list(p.keys()))
    p_prob = (list(p.values()))
    q_elements = (list(q.keys()))
    q_prob = (list(q.values()))
    for _ in range(rounds):
#         F_p = random.choices(list(p.keys()), weights=p.values(), k=m)
#         F_q = random.choices(list(q.keys()), weights=q.values(), k=m)
        F_p = random_sampling(p_elements, p_prob, m)
        F_q = random_sampling(q_elements, q_prob, m)
    
        r_p = self_pairwise_collisions(F_p)
        r_q = self_pairwise_collisions(F_q)
        
#         Q_p = random.choices(list(p.keys()), weights=p.values(), k=m)
#         Q_q = random.choices(list(q.keys()), weights=q.values(), k=m)
        Q_p = random_sampling(p_elements, p_prob, m)
        Q_q = random_sampling(q_elements, q_prob, m)

        s_pq = across_list_collisions(Q_p, Q_q)
        
        r = 2*m/(m-1)*(r_p+r_q)
        s = 2*s_pq
        if r-s > 3*m**2*e**2/4:
            num_reject += 1
    
    if num_reject >= rounds/2:
        return 0
    return 1

def sampling_distribution_prime(p, domain, sampled_set, m):
    result = []
    p_elements = (list(p.keys()))
    prob = np.ones(len(p_elements))/len(p_elements)
    domain_prob = np.ones(len(domain))/len(domain)
    for i in range(m):
#         sample = np.random.choice(p_elements,1)[0]
        sample = random_sampling(p_elements,prob,1)[0]
#         sample = random.choices(list(p.keys()), weights=p.values(), k=1)[0]
        if sample not in sampled_set:
            result.append(sample)
        else:
            result.append(np.random.choice(domain, domain_prob, 1))[0]
#             result.append(random.sample(domain, 1)[0])
    return list_to_distribution(result)


def l1_distance_tester(p, q, e, delta, samples = 100):
    domain = set(p.keys()).intersection(q.keys())
    n = len(domain)
    b = (e/n)**(2/3)
#     print(n,e,delta)
#     m = e**(-8/3)*n**(2/3)*math.log(n/delta) #m is O of this
#     print("yoyoyo m", m)
#     m = int(m*1000000)
#     m = k
    m = samples
    print("m is:", m)
    
    print(len(list(p.keys())))
    p_elements = (list(p.keys()))
    p_prob = (list(p.values()))
    q_elements = (list(q.keys()))
    q_prob = (list(q.values()))
     
#         F_p = random_sampling(p_elements, p_prob, m)
#         F_q = random_sampling(q_elements, q_prob, m)
    S_p = random_sampling(p_elements, p_prob, m)
    S_q = random_sampling(q_elements, q_prob, m)
#     S_p = random.choices(list(p.keys()), weights=p.values(), k=m)
#     S_q = random.choices(list(q.keys()), weights=q.values(), k=m)
    
    S_p_d = list_to_distribution(S_p)
    S_q_d = list_to_distribution(S_q)
    threshold = (1-e/26)*m*b
    for word in list(S_p_d.keys()):
        if S_p_d[word] < threshold:
            del S_p_d[word]
    for word in list(S_q_d.keys()):
        if S_q_d[word] < threshold:
            del S_q_d[word]
    
    print("hehehehe")
            
    m_ = n**(2/3)/e**(8/3) # O of this
    e_ = e/(2*n**0.5)
    delta_ = delta/2
    if len(S_p_d) == 0 and len(S_q_d) == 0:
        return l2_distance_tester(p, q, m_, e_, delta_)
    
    if lp_distance(S_p_d, S_q_d, 1) > e*m/8:
        return 0

    sampled_set = set(S_p).intersection(set(S_q))
    p_prime = sampling_distribution_prime(p, domain, sampled_set, m)
    q_prime = sampling_distribution_prime(q, domain, sampled_set, m)
    
    return l2_distance_tester(p_prime, q_prime, m_, e_, delta_)




d_p_full = d_2008_11
d_q_full = d_2008_12

d_p_multiple = {key:val for key, val in d_p_full.items() if val > 1}
d_q_multiple = {key:val for key, val in d_q_full.items() if val > 1}
p_total = sum(d_p_multiple.values())
q_total = sum(d_q_multiple.values())

d_p = {key:val/p_total for key, val in d_p_multiple.items()}
d_q = {key:val/q_total for key, val in d_q_multiple.items()}

l1_distance = lp_distance(d_p, d_q, 1)
print(l1_distance)


delta = 0.1
e = l1_distance*250
domain = set(d_p.keys()).intersection(d_q.keys())
n = len(domain)

a=(e**(4/3)/(32*n**(1/3)))
b=(e/(4*n**0.5))
print(n)
print(a,b,max(a,b), e)



delta = 0.75
# e = l1_norm/2

# print(l1_distance_tester(d_p, d_q, e, delta))
# count = 0
# for i in range(100):
#     if i%10==0:
#         print(i, count)
#     count+=l1_distance_tester(d_p, d_q, e, delta)
# print("result:", count)


result = {}
# m_test = [10, 25, 50, 75, 100, 150, 200, 400, 800, 1600]
m_test = [3200, 6400]
for m in m_test:
#     print(m)
    total_time = 0
    for i in range(1):
        start_time = time.time()
        l1_distance_tester(d_p, d_q, e, delta, samples=m)
        time_used = time.time() - start_time
        total_time += time_used
    avg_time = total_time/3
    result[m] = avg_time
    print(m, avg_time)
