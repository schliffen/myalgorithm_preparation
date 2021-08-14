#
#
#
import numpy as np
import pandas as pd
import operator
import pickle

# PARAMETER SETTINGS
data_dir = '/home/ali/Projlab/Algorithms/paper/Salahi-Simulation/'
num_simul = 10000 # number of simulations
xdata1 = []
ydata1 = []
num_input = 2
num_output = 2
num_lambda_rows = 821
num_lambdas = 20

xls = pd.ExcelFile(data_dir + 'Russell_Sim_RERM-20DMUs.xlsx') # ERM/Russell_Sim_Trial1-ERM

pre_xdata = xls.parse(0)
data_real = xls.parse(1)
lambdas = xls.parse(2)
#
# --------
# preparing xdata
dmu_size = len(data_real.keys()[1:])
for item in range(1, num_input+1):
    if xdata1 ==[]:
        xdata1 = np.array(list(pre_xdata.iloc[1:,item]) )
        xdata1 = np.vstack((xdata1, list(pre_xdata.iloc[1:,item  + num_input + 1])) )
    else:
        xdata1 = np.vstack((xdata1,  np.array(list(pre_xdata.iloc[1:,item]) ) ) )
        xdata1 = np.vstack((xdata1,  list(pre_xdata.iloc[1:, item  +  num_input + 1] ) ) )

for item in range(1, num_output + 1):
    yitem = item + 2*num_input + 1
    if ydata1 ==[]:
        ydata1 = np.array(list(pre_xdata.iloc[1:,yitem  + 1]) )
        ydata1 = np.vstack((ydata1, np.array(list(pre_xdata.iloc[1:,yitem +  num_output + 2]))) )
    else:
        ydata1 = np.vstack((ydata1,  np.array(list(pre_xdata.iloc[1:, yitem  + 1])) ) )
        ydata1 = np.vstack((ydata1,  np.array(list(pre_xdata.iloc[1:, yitem + num_output + 2]) ) ) )

xdata = np.hstack( (xdata1.T, ydata1.T ) )
# ydata1.append(
#         [list(pre_xdata.iloc[1:,yitem+1]), list(pre_xdata.iloc[1:,yitem + 1 + 6])]
#     )
#
gama = np.array( list( data_real['Gamma'] ) )
# xdata1 = np.array(xdata1).transpose(1,0)

columns = list(lambdas.columns[2:2 + num_lambdas])
lambdas1 = []
for item in range(len(columns)):
    if lambdas1 == []:
        lambdas1 = np.array(lambdas.iloc[:num_lambda_rows-1, item + 2]) # , dtype = np.float64
    else:
        lambdas1 = np.vstack( (lambdas1, np.array( lambdas.iloc[:num_lambda_rows-1, item + 2]) )) # , dtype = np.float64
# lambdas1.append(list(lambdas[item]))
# lambdas1 = np.array(lambdas1, dtype = np.float64)
#
# initial parameters

#
# shape: dmu x gamma x dimension

def sort_dics(v1, v2):

    dv1 = {}
    dv2 = {}
    for i in range(v1.shape[0]):
        dv1.update({i+1:v1[i]})
        dv2.update({i+1:v2[i]})


    sorted_dv1 = sorted(dv1.items(), key=operator.itemgetter(1), reverse=True)
    sorted_dv2 = sorted(dv2.items(), key=operator.itemgetter(1), reverse=True)

    return sorted_dv1, sorted_dv2

def check_reanking(produced, truth):

    produced = np.array(produced)
    truth = np.array(truth)[:,1:]

    gama_comp = []
    for j1 in range(produced.shape[0]): # this means for each gamma

        tmp_tr = truth[j1]
        tmp_pr = produced[j1]

        tmp_tr, tmp_pr = sort_dics(tmp_tr, tmp_pr)

        pr_dic={}
        tr_dic = {}
        pr_rank_list = []
        tr_rank_list = []
        pr_dmu_list = []
        tr_dmu_list = []

        for i1 in range(len(tmp_pr)): #
            if i1 == 0:
                pr_dic.update({tmp_pr[i1][0]: (i1+1, tmp_pr[i1][1]) })
                pr_rank_list.append(i1+1)
                pr_dmu_list.append(tmp_pr[i1][0])

                tr_dic.update({tmp_tr[i1][0]: (i1+1, tmp_tr[i1][1]) })
                tr_rank_list.append(i1+1)
                tr_dmu_list.append(tmp_tr[i1][0])

            else:

                if pr_dic[pr_dmu_list[-1]][1] == tmp_pr[i1][1]:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1], tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1])
                    pr_dmu_list.append(tmp_pr[i1][0])

                else:
                    pr_dic.update({tmp_pr[i1][0]: (pr_rank_list[-1]+1, tmp_pr[i1][1]) })
                    pr_rank_list.append(pr_rank_list[-1]+1)
                    pr_dmu_list.append(tmp_pr[i1][0])

                if tr_dic[tr_dmu_list[-1]][1] == tmp_tr[i1][1]:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1], tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1])
                    tr_dmu_list.append(tmp_tr[i1][0])

                else:
                    tr_dic.update({tmp_tr[i1][0]: (tr_rank_list[-1] + 1, tmp_tr[i1][1]) })
                    tr_rank_list.append(tr_rank_list[-1]+1)
                    tr_dmu_list.append(tmp_tr[i1][0])

        # assuming there are dmus
        dmu_rank_compare = []
        for dmui in range(dmu_size):
            if pr_dic[dmui+1][0] == tr_dic[dmui+1][0]:
                dmu_rank_compare.append(1)
            else:
                dmu_rank_compare.append(0)

        gama_comp.append(dmu_rank_compare)

    return gama_comp


def tfunc(lamba, num_simul, xdata1, data_real):
    #
    gama_dmu = []
    #
    for gam_iter in range(0, gama.shape[0]): # iteration for gamma
        #
        sim_dmu = []
        #

        # random number generation
        # x_dim = xdata.shape[1]//4 # [x_l, x_u], [y_l, y_u]
        # y_dim = num_output
        #

        X = np.array( [[ np.random.uniform(xdata1[j][2*i], xdata1[j][2*i+1], num_simul) for i in range(num_input)] for j in range(lamba.shape[0])] , dtype = np.float64)
        Y = np.array( [[ np.random.uniform(xdata1[j][2*i + 2*num_input], xdata1[j][2*i + 1 + 2*num_input], num_simul) for i in range(num_output) ] for j in range(lamba.shape[0])] , dtype = np.float64)

        #
        for k in range(num_simul):
            #
            dmus = []
            # for i in range(x_dim):
            for idmu in range(dmu_size): # iteration for DMU
                uy = 0.
                vx = 0.
                # computing the indexes: indexes is correspondent to the gamma and DMU iterations:
                #
                gmdm_ind = lamba.shape[0] * (gam_iter ) + idmu
                # -------------
                # for idd in range(lamba.shape[0]):

                #     uy += np.array([ (X[idd,i, k] * lamba[idd, gmdm_ind ]) / X[idmu, i, k]  for i in range(x_dim) ]).sum()
                #     vx += np.array([ (Y[idd,i, k] * lamba[idd, gmdm_ind ]) / Y[idmu, i, k]  for i in range(x_dim) ]).sum()
                # print('lambda idx {0}'.format(gmdm_ind))
                # try:
                uy = np.array([( np.dot( X[:,i,k] , lamba[:, gmdm_ind ]) / X[idmu, i, k] ) for i in range(num_input)]).sum() / X.shape[1]
                vx = np.array([( np.dot( Y[:,i,k] , lamba[:, gmdm_ind ]) / Y[idmu, i, k] ) for i in range(num_output)]).sum() / Y.shape[1]
                # except:
                #     print('here')
                # target = 1. if uy/vx > 1. else np.divide(uy,vx)
                target = np.divide(uy,vx)
                #
                dmus.append(target)
            #
            sim_dmu.append(dmus)

        gama_dmu.append(sim_dmu)

    gama_dmu = np.array(gama_dmu).transpose(1,0,2)
    # ranking the simulations
    sim = []
    for j in range(gama_dmu.shape[0]): # iterate ove the simulations
        gama_ranking_compare = check_reanking(gama_dmu[j,:,:], data_real)
        sim.append(gama_ranking_compare)

    return sim


sim_result = tfunc(lambdas1, num_simul, xdata, data_real)

sim_res = np.array(sim_result).transpose(2,1,0) # DMU X GAMMA X simulations

total_res = []

total_sims = sim_res.shape[2]
for i in range(sim_res.shape[0]): # iterate for dmus
    tmp_res = []
    for j in range(sim_res.shape[1]): # iterate over gamma
        #        for k in range(sim_result.shape[2]):
        tmp_res.append(sim_res[i,j].sum()/sim_res.shape[2])
    total_res.append(tmp_res)

final_res = np.array(total_res).transpose(1,0)
pd.DataFrame(final_res).to_csv("results/final_Russell_Sim_Trial1-ERM-New" +  str(num_simul) + ".csv")
average = final_res.sum(axis=1)/final_res.shape[1]
pd.DataFrame(average).to_csv("results/average_Russell_Sim_Trial1-ERM-New" +  str(num_simul) + ".csv")
# with open('total_res.npy', 'wb') as f:
#     pickle.dump(total_res, f)


# checking the results for the simulations
print('done!')





