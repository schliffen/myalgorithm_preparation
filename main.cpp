#include <iostream>
#include <vector>





/*
int nonDivisible(int k, std::vector<int> s){
    int rem, lmax=0;
    std::vector<int> rarr(k);

    for (int i=0; i<s.size(); i++) {
        rem = s[i] % k;
        rarr.at(rem) +=1;
    }


    for (int i=0; i<(k+1)/2;i++) {
        if (rarr[i] >= rarr[k - i]) lmax += rarr[i];
        else  lmax += rarr[k - i];
    }
    if (k%2==0 && rarr[k/2]>0) lmax+=1;
    if (rarr[0]>0) lmax+=1;


    return lmax;
}

*/

int main() {

    int k=3;
    std::vector<int> s{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    //int res = nonDivisible(k, s);


    //std::cout << "Result: " << res << std::endl;
    return 0;
}
