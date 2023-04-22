
def minAcC_svd(Ac, C, errors=False):
    print('Using svd strategy...')
    AcC = ncon([Ac, C.conj().T], ((-1, -2, 1), (1, -3))).reshape(d*D, D)
    Ul, _, Vl = la.svd(AcC, full_matrices=False)
    AL =  (Ul @ Vl).reshape(D, d, D)

    CAc = ncon([C.conj().T, Ac], ((-1, 1), (1, -2, -3))).reshape(D, d*D)
    Ur, _, Vr = la.svd(CAc, full_matrices=False)
    AR = (Ur @ Vr).reshape(D, d, D)

    print('Verigying Al canonical...')
    ALAL = ncon([AL, AL.conj()], ((1, 2, -1), (1, 2, -2)))
    print(np.allclose(ALAL, np.eye(D)))

    print('Verigying Ar canonical...')
    ARAR = ncon([AR, AR.conj()], ((-1, 1, 2), (-2, 1, 2)))
    print(np.allclose(ARAR, np.eye(D)))

    AlC = ncon([AL, C], ((-1, -2, 1), (1, -3)))
    ϵL = np.linalg.norm(AlC - Ac) # Error in Al, should converge near optima

    CAr = ncon([C, AR], ((-1, 1), (1, -2, -3)))
    ϵR = np.linalg.norm(CAr - Ac) # Error in Ar should converge near optima

    if errors:
        return AL, AR, ϵL, ϵR
    return AL, AR


