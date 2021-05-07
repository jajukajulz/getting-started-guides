from collections import Counter

x = [0,1,0,0,0,1]
c = Counter(x)

def probability(a):
    # returns the probability of a given number a
    return float(c[a]) / len(x)

prob = probability(1)
print(prob)