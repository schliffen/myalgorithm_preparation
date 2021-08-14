#
# Leet
#
import numpy as np
import math



def getOptions(allprices, dollars, tsum, ite, track):
    if tsum>dollars: return 0;

    if(tsum<=dollars  ):
        if not track in tracker:
            tracker.append(track.copy())

    if (ite == len(allprices)-1): return 0

    for it1, item0 in enumerate(allprices):
        for it2, item1 in enumerate(allprices[it1]):
            tsum +=item1
            track[str(it1)] += item1
            ite = it1 + it2
            getOptions( allprices, dollars, tsum, ite, track.copy())
            track[str(it1)] -= item1
            tsum -=item1




if __name__ == '__main__':
       poShoes = [2,3]
       potops=[4]
       poskirt = [2,3]
       pojeans = [1,2]

       dollars = 10

       allprices = []

       allprices.append(poShoes)
       allprices.append(potops)
       allprices.append(poskirt)
       allprices.append(pojeans)

       tracker = []
       track = {'0':0, '1':0, '2':0, '3':0}

       for i1 in poShoes:
           for i2 in potops:
              for i3 in poskirt:
                  for i4 in pojeans:

                       totsum = i1 + i2 + i3 + i4
                       if totsum>dollars:
                           continue

                       # coding the combination
                       track['1'] = i1; track['2'] = i2;track['3'] = i3;track['4'] = i4;
                       getOptions(allprices, dollars, totsum, 0, track.copy() )


       print( "count" ,len(tracker)  )


