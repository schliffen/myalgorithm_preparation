#
#
#
import numpy as np
import math
import os
import random
import re
import sys

#
# Complete the 'nonDivisibleSubset' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. INTEGER k
#  2. INTEGER_ARRAY s
#


def nonDivisibleSubset( k, s ):
    # Write your code here

    max = 0
    # s1 = []
    rarr = [0]*k
    for i in range(len(s)):
        rem = s[i] % k
        # s1.append(rem)
        rarr[rem] += 1


    for i in range(1,(k+1)//2):
        if rarr[i] >= rarr[k-i]:
           max+=rarr[i]
        else:
            max+=rarr[k-i]

    if k%2==0 and rarr[k//2]>0: max+=1
    if rarr[0]>0: max+=1

    return max






if __name__ == '__main__':


    k=1
    s= [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # s = [770528134, 663501748, 384261537 ,800309024, 103668401, 538539662, 385488901 ,101262949 ,557792122 ,46058493]
    result = nonDivisibleSubset(k, s)

    print( result )





    # int max=1, add;
    # std::vector<int> s1;
    #
    # for (int i=0; i<s.size(); i++){
    # s1.clear(); s1.push_back( s.at(i) );
    #
    # for (int j=0; j<s.size(); j++){
    # add =1;
    # for (int l=0; l<s1.size(); l++)
    # if ( (s1.at(l) + s.at(j))%k ==0 || s1.at(l) == s.at(j)) {
    # add =0;
    # break;
    # }
    #
    # if (add==1) s1.push_back(s.at(j));
    #
    # }
    #
    # if (s1.size()> max) max = s1.size();
    #
    # }