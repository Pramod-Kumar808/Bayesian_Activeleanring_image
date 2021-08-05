import matplotlib.pyplot as plt

acc_rand = [0.17299999296665192, 0.17100000381469727, 0.1706666648387909, 0.1693333387374878, 0.18466666340827942, 0.16766667366027832]
loss_rand = [1.8646432161331177, 1.8059005737304688, 1.7890431880950928, 1.7969996929168701, 1.798080563545227, 1.7906302213668823]



acc_var = [0.17399999499320984, 0.17366667091846466, 0.1756666600704193, 0.17399999499320984, 0.17499999701976776, 0.17466667294502258]
loss_var = [2.1828062534332275, 1.806448221206665, 1.796953797340393, 1.7937910556793213, 1.7975748777389526, 1.7992421388626099]

loop = [i for i in range(len(acc_rand))]
plt.plot(loop, acc_rand, label='accuracy random')
plt.plot(loop, acc_var, label='accuracy var_ratio')
plt.legend()
plt.show()

loop = [i for i in range(len(loss_rand))]
plt.plot(loop, loss_rand, label='loss random')
plt.plot(loop, loss_var, label='loss var_ratio')
plt.legend()
plt.show()

