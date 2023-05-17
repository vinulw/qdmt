
if __name__=="__main__":
    n_sites = 5

    contr_h = list(range(1, 2*n_sites+1))
    print('Contr_h: ', contr_h)
    start_i = contr_h[-1] + 1

    for nL in range(1, n_sites):
        nR = n_sites - nL
        print('Current pair: ', (nL, nR))
        i_ = start_i
        h_i = contr_h[0]
        h_i_dag = contr_h[n_sites]
        contr_Al = [[i_, h_i , i_+1]]
        contr_Al_dag = [[i_, h_i_dag, i_+2]]

        i_ = i_+1
        h_i += 1
        h_i_dag += 1

        for _ in range(nL - 1):
            contr_Al.append([i_, h_i, i_+2])
            contr_Al_dag.append([i_+1, h_i_dag, i_+3])

            i_ += 2
            h_i += 1
            h_i_dag += 1

        contr_Al[-1][-1] = -3
        contr_Al_dag[-1][-1] = -1

        contr_Ar = []
        contr_Ar_dag = []

        for _ in range(nR):
            contr_Ar.append([i_, h_i, i_+2])
            contr_Ar_dag.append([i_+1, h_i_dag, i_+3])

            i_ += 2
            h_i += 1
            h_i_dag += 1

        contr_Ar[0][0] = -4
        contr_Ar_dag[0][0] = -2
        contr_Ar_dag[-1][-1] = contr_Ar[-1][-1]

        print(contr_Al)
        print(contr_Al_dag)
        print(contr_Ar)
        print(contr_Ar_dag)

        breakpoint()
