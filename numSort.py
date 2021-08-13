#
# ordering a sum
#
import numpy as np




def pq(p,q):
    return 2**p*3**q



def mxpolino(n):

    S= []


    for q in range(n//2+1):
        for p in range(n//2+1):
            if q >n//2+1:
                    return S[n]

            res = pq(p,q)
            sort = True
            if len(S)==0:
                S.append(res)
                continue
            it = len(S) - 1

            while sort:
                    if res>=S[it]:
                        S.insert(it+1, res)
                        sort = False
                    else:
                        it -=1






if __name__ == "__main__":

    n= 3
    res = mxpolino( n )

    print(res)



