from copy import copy

if __name__=="__main__":
    n_sites = 5

    h_contr = list(range(1, 2*n_sites+1))
    print('Contr_h: ', h_contr)
    start_i = h_contr[-1] + 1

    for site in range(n_sites):
        i = start_i
        nL = site
        nR = n_sites - 1 - site

        print('Site no: ', site)
        print(f'   (nL, nR): {nL}, {nR}')

        h_i = 1
        h_i_dag = h_contr[n_sites]

        Al_contr = []
        Al_dag_contr = []
        Ar_contr = []
        Ar_dag_contr = []

        for _ in range(nL):
            Al_contr.append([i, h_i, i + 2])
            Al_dag_contr.append([i+1, h_i_dag, i + 3])

            h_i += 1
            h_i_dag += 1
            i += 2

        # Skip over the site
        h_i += 1
        h_i_dag += 1

        for _ in range(nR):
            Ar_contr.append([i, h_i, i + 2])
            Ar_dag_contr.append([i+1, h_i_dag, i + 3])

            h_i += 1
            h_i_dag += 1
            i += 2


        h_contr_ = copy(h_contr)
        h_contr_[site] = -5
        h_contr_[n_sites + site] = -2

        if len(Al_contr) > 0:
            Al_dag_contr[0][0] = Al_contr[0][0]
            Al_contr[-1][-1] = -1
            Al_dag_contr[-1][-1] = -4

        if len(Ar_contr) > 0:
            Ar_dag_contr[-1][-1] = Ar_contr[-1][-1]
            Ar_contr[0][0] = -3
            Ar_dag_contr[0][0] = -6

        I_contr = []
        if site == 0:
            I_contr.append([-1, -4])
        elif site == n_sites-1:
            I_contr.append([-3, -6])

        print('   Al: ', Al_contr)
        print('   Al_dag: ', Al_dag_contr)
        print('   Ar: ', Ar_contr)
        print('   Ar_dag: ', Ar_dag_contr)
        print('   h: ', h_contr_)
        print('   I: ',I_contr)



