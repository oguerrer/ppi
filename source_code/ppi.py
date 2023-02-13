"""Policy Priority Inference (PPI)

Complexity Economics and Sustainable Development: A Computational Framework for Policy Priority Inference

Authors: Omar A. Guerrero & Gonzalo Castañeda
Written in Python 3.7


Example
-------



Dependencies
--------------------------
- Numpy
- Joblib (optional, for parallel computing)


"""

# import necessary libraries
import numpy as np
from joblib import Parallel, delayed
import warnings
warnings.simplefilter("ignore")


def run_ppi(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None,
            Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=None, frontier=None):
    
    """Function to run one simulation of the Policy Priority Inference model.

    Parameters
    ----------
        I0: numpy array 
            Vector with the initial values of the development indicators.
        alphas: numpy array
            Vector with parameters representing the size of a positive change 
            in the indicators.
        alphas_prime: numpy array
            Vector with parameters representing the size of a negative change 
            in the indicators.
        betas: numpy array
            Vector with parameters that normalise the contribution of public 
            expenditure and spillovers to the probability of a positive change
            in the indicators.
        A:  2D numpy array (optional)
            The adjacency matrix of the interdependency network of development 
            indicators. The rows denote the origins of dependencies and the columns
            their destinations. Self-loops are not allowed, so PPI turns A's
            diagonal into zeros. If not provided, the default value is a matrix 
            full of zeros.
        R:  numpy array (optional)
            Vector that specifies whether an indicator is instrumental
            or collateral. Instrumental indicators have value 1 and collateral
            ones have zero. If not provided, the default value is
            a vector of ones.
        bs: numpy array (optional)
            Vector with modulating factors for the budgetary allocation of
            each instrumental indicator. Its size should be equal to the number
            of instrumental indicators. If not provided, the default value is
            a vector of ones.
        qm: A floating point, an integer, or a numpy array (optional)
            Captures the quality of the monitoring mechanisms that procure 
            public governance. There are three options to specify qm:
                - A floating point: it assumes that the quality of monitoring is
                the same for every indicator, that it is exogenous, 
                and that it remains constant through time. In this case, qm 
                should have a value between 0 and 1.
                - An integer: it assumes that the quality of monitoring is captured 
                by one of the indicators, so qm gives the index in I0 where such 
                indicator is located. Here, the quality of monitoring is endogenous, 
                dynamic, and homogenous across the indicators.
                - A vector: it assumes that the quality of monitoring is
                heterogeneous across indicators, exogenous, and constant. Thus,
                qm must have a size equal to the number of instrumental
                indicators. Each entry in qm denotes the quality of monitoring
                related to a particular indicator. Each value in this vector should be
                between 0 and 1.
            If not provided, the default values is qm=0.5.
        rl: A floating point, an integer, or a numpy array (optional)
            Captures the quality of the rule of law. There are three options to
            specify rl:
                - A floating point: it assumes that the quality of the rule of law 
                is the same for every indicator, that it is exogenous, 
                and that it remains constant through time. In this case, rl 
                should have a value between 0 and 1.
                - An integer: it assumes that the quality of the rule of law is 
                captured by one of the indicators, so rl gives the index in I0 
                where such indicator is located. Here, the quality of the rule 
                of law is endogenous, dynamic, and homogenous across the indicators.
                - A vector: it assumes that the quality of the rule of law is
                heterogeneous across indicators, exogenous, and constant. Thus,
                rl has to have a size equal to the number of instrumental
                indicators. Each entry in rl denotes the quality of monitoring
                in a particular indicator. Each value in this vector should be
                between 0 and 1.
            If not provided, the default values is rl=0.5.
        Imax: numpy array (optional)
            Vector with the theoretical upper bound of the indicators. If an entry
            contains a missing value (NaN), then there is no upper bound defined
            for that indicator and it will grow indefinitely. If not provided,
            no indicator will have upper bound.
        Imin: numpy array (optional)
            Vector with the theoretical lower bound of the indicators. If an entry
            contains a missing value (NaN), then there is no lower bound defined
            for that indicator and it will decrease indefinitely. If not provided,
            no indicator will have lower bound.
        Bs: numpy ndarray
            Disbursement schedule across expenditure programs. There are three 
            options to specify Bs:
                - A matrix: this is a disaggregated specification of the 
                disbursement schedule across expenditure programs and time. 
                The rows correspond to expenditure programs and the columns
                to simulation periods. Since there may be more or less expenditure 
                programs than indicators, the number of rows in Bs should be
                consistent with the information contained in parameter B_dict,
                otherwise PPI will throw and exception. Since the number of 
                columns denotes the number of simulation periods, parameter T
                will be overridden.
                - A vector: this would be equivalent to a matrix with a single
                row, i.e. to having a single expenditure program. This representation
                is useful when there is no information available across programs,
                but there is across time. Like in the matrix representation, 
                this input should be consistent with B_dict.
            If not provided, the default value is Bs=100 in every period.
        B_dict: dictionary (optional)
            A dictionary that maps the indices of every indicator into the 
            expenditure program(s) designed to affect them. Since there may be
            multiple programs designed to impact and indicator, or multiple
            indicators impacted by the same program, this mapping is not 
            one to one. To account for this, B_dict has, as keys, the indices 
            of the instrumental indicators and, as values, lists containing
            the indices of the expenditure programs designed to impact them.
            The indices of the programmes correspond to the rows of parameter 
            Bs in its matrix form.  The user should make sure that the keys are 
            consistent with the indices of those indicators that are instrumental. 
            Likewise, the indices of the expenditure programs should be consistent 
            with the number of rows in Bs, otherwise PPI will throw an exception.
            Providing B_dict is necessary if Bs is a matrix with more than one 
            row.
        G: numpy array (optional)
            The development goals to be achieved for each indicator. These are
            used only to calculate the initial development gaps, which affect 
            the allocation decision of the government agent. If not provided,
            the default initial allocations are determined randomly.
        T: int (optional)
            The maximum number of simulation periods. If Bs is provided, then T
            is overridden by the number of columns in Bs. If not provided, the
            default value in T=50.
        frontier: numpy array (optional)
            A vector with exogenous probabilities of positive growth for each
            indicator. If an entry contains NaN, then the corresponding
            probability is endogenous. This vector is typically used to perform
            analysis on the budgetary frontier, in which the probability of 
            success of the instrumental indicators is set to 1. Alternatively,
            the 'relaxed' frontier consists of imposing a high (but less than 1)
            probability of growth. If not provided, the default behaviour is that
            all the probabilities of success are endogenous. It is recommended
            to not provide this parameter unless the user understands the
            budgetary frontier analysis and idiosyncratic bottlenecks.
            
        
    Returns
    -------
        tsI: 2D numpy array
            Matrix with the time series of the simulated indicators. Each row
            corresponds to an indicator and each column to a simulation step.
        tsC: 2D numpy array
            Matrix with the time series of the simulated contributions. Each row
            corresponds to an indicator and each column to a simulation step.
        tsF: 2D numpy array
            Matrix with the time series of the simulated benefits. Each row
            corresponds to an indicator and each column to a simulation step.
        tsP: 2D numpy array
            Matrix with the time series of the simulated allocations. Each row
            corresponds to an indicator and each column to a simulation step.
        tsS: 2D numpy array
            Matrix with the time series of the simulated spillovers. Each row
            corresponds to an indicator and each column to a simulation step.
        tsG: 2D numpy array
            Matrix with the time series of the simulated growth probabilities. 
            Each row corresponds to an indicator and each column to a simulation step.
    """
    
    
    
    
    ## SET DEFAULT PARAMETERS & CHECK INPUT INTEGRITY 
    
    # Number of indicators
    assert np.sum(np.isnan(I0)) == 0, 'I0 should not contain missing values'
    N = len(I0) 
    
    # Structural factors
    assert np.sum(np.isnan(alphas)) == 0, 'alphas should not contain missing values'
    assert len(alphas) == N, 'alphas should have the same size as I0'
    assert np.sum(np.isnan(alphas_prime)) == 0, 'alphas_prime should not contain missing values'
    assert len(alphas_prime) == N, 'alphas_prime should have the same size as I0'
    
    # Normalising factors
    assert np.sum(np.isnan(betas)) == 0, 'betas should not contain missing values'
    assert len(betas) == N, 'betas should have the same size as I0'
    
    # Interdependency network
    if A is None:
        A = np.zeros((N,N))
    else:
        assert np.sum(np.isnan(A)) == 0, 'A should not contain missing values'
        assert len(A.shape) == 2 and A.shape[0] == N and A.shape[1] == N, 'A should have as many rows and columns as the number of indicators'
        A = A.copy()
        np.fill_diagonal(A, 0) # make sure there are no self-loops
    
    # Instrumental indicators
    if R is None:
        R = np.ones(N).astype(bool)
    else:
        R[R!=1] = 0
        R = R.astype(bool)
        assert np.sum(R) > 0, 'PPI needs at least one instrumental indicator, make sure that at least one entry in R is 1'
        assert len(R) == N, 'R should have the same size as I0'
    
    # Number of instrumental indicators
    n = int(R.sum())
        
    # Modulating factors
    if bs is None:
        bs = np.ones(n)
    else:
        assert np.sum(np.isnan(bs)) == 0, 'bs should not contain missing values'
        assert len(bs) == n, 'bs should have the same size as the number of instrumental indicators (ones in R)'
        
    # Quality of monitoring
    if qm is None:
        qm = np.ones(n)*.5
    elif type(qm) is np.ndarray:
        assert np.sum(np.isnan(qm)) == 0, 'qm should not contain missing values'
        assert len(qm) == n, 'qm should have the same size as the number of instrumental indicators (ones in R)'
        
    # Quality of the rule of law
    if rl is None:
        rl = np.ones(n)*.5
    elif type(rl) is np.ndarray:
        assert np.sum(np.isnan(rl)) == 0, 'rl should not contain missing values'
        assert len(rl) == n, 'rl should have the same size as the number of instrumental indicators (ones in R)'
        
    # Theoretical upper bounds
    if Imax is not None:
        assert len(Imax) == N, 'Imax should have the same size as I0'
        if np.sum(~np.isnan(Imax)) > 0:
            assert np.sum(Imax[~np.isnan(Imax)] < I0[~np.isnan(Imax)]) == 0, 'All entries in Imax should be greater than their corresopnding value in I0'

    # Theoretical lower bounds
    if Imin is not None:
        assert len(Imin) == N, 'Imin should have the same size as I0'
        if np.sum(~np.isnan(Imin)) > 0:
            assert np.sum(Imin[~np.isnan(Imin)] > I0[~np.isnan(Imin)]) == 0, 'All entries in Imin should be lower than their corresopnding value in I0'

    # Payment schedule
    assert type(Bs) is np.ndarray, 'Bs must be a numpy vector or a matrix'
    if T is None:
        T = 50
    if Bs is None:
        Bs = np.array([np.ones(T)*100])
        B_dict = dict([(i,[0]) for i in range(N) if R[i]])
    elif type(Bs) is np.ndarray and len(Bs.shape) == 1:
        Bs = np.array([Bs])
        B_dict = dict([(i,[0]) for i in range(N) if R[i]])
        T = Bs.shape[1]
    else:
        T = Bs.shape[1]
    
    assert np.sum(np.isnan(Bs)) == 0, 'Bs should not contain missing values'
    
    # Dictionary linking indicators to expenditure programs
    assert B_dict is not None, 'If you provide Bs, you must provide B_dict as well'
    assert len(B_dict) == n, 'The number of keys in B_dict should be the same as the number of ones in R'
    assert np.sum(np.in1d(np.array(list(B_dict.keys())), np.arange(N))) == n, 'The keys in B_dict must match the indices of the entries in R that contain ones'
    assert sum([type(val) is list for val in B_dict.values()]) == n, 'Every value in B_dict dictionary must be a list'
    assert sum([True for i in range(N) if R[i] and i not in B_dict]) == 0, 'Every key in B_dict must be mapped into a non-empty list'
    assert sum([True for i in B_dict.keys() if not R[i]]) == 0, 'The keys in B_dict must match the indices of the entries in R that contain ones'
    
    # Create reverse dictionary linking expenditure programs to indicators
    programs = sorted(np.unique([item for sublist in B_dict.values() for item in sublist]).tolist())
    assert Bs.shape[0] == len(programs), 'The number of unique expenditure programs in B_dict do not match the number of rows in Bs'
    program2indis = dict([(program, []) for program in programs])
    sorted_programs = sorted(program2indis.keys())
    for indi, programs in B_dict.items():
        for program in programs:
            if R[indi]:
                program2indis[program].append( indi )
    inst2idx = np.ones(N)*np.nan
    inst2idx[R] = np.arange(n)
    
    # Create initial allocation profile
    if G is not None:
        gaps = G-I0
        gaps[G<I0] = 0
        p0 = gaps/gaps.sum()
        P0 = np.zeros(n)
    else:
        P0 = np.zeros(n)
        p0 = np.random.rand(n)
    i=0
    for program in sorted_programs:
        indis = program2indis[program]
        relevant_indis = inst2idx[indis].astype(int)
        P0[relevant_indis] += Bs[i,0]*p0[relevant_indis]/p0[relevant_indis].sum()
        i+=1
    
    # Prevent null allocations
    P0 = Bs[:,0].sum()*P0/P0.sum()
    Bs[Bs==0] = 10e-12
    P0[P0==0] = 10e-12
    
    
    
    ## INSTANTIATE ALL VARIABLES AND CREATE CONTAINERS TO STORE DATA
        
    P = P0.copy() # first allocation
    F = np.random.rand(n) # policymakers' benefits
    Ft = np.random.rand(n) # lagged benefits
    X = np.random.rand(n)-.5 # policymakers' actions
    Xt = np.random.rand(n)-.5 # lagged actions
    H = np.ones(n) # cumulative spotted inefficiencies
    HC = np.ones(n) # number of times spotted so far
    signt = np.sign(np.random.rand(n)-.5) # direction of previous actions
    changeFt = np.random.rand(n)-.5 # change in benefits
    C = np.random.rand(n)*P # contributions
    I = I0.copy() # initial levels of the indicators
    It = np.random.rand(N) # lagged indicators

    tsI = np.empty(N,T) # stores time series of indicators
    tsC = np.empty(N,T) # stores time series of contributions
    tsF = np.empty(N,T) # stores time series of benefits
    tsP = np.empty(N,T) # stores time series of allocations
    tsS = np.empty(N,T) # stores time series of spillovers
    tsG = np.empty(N,T) # stores time series of gammas
    
    
    
    ## MAIN LOOP
    for t in range(T):
        
        tsI[:,T] = I # store this period's indicators
        tsP[:,T] = P # store this period's allocations


        ### REGISTER INDICATOR CHANGES ###
        deltaBin = (I>It).astype(int) # binary for computing spillovers
        deltaIIns = I[R]-It[R] # instrumental indicators' changes
        if np.sum(np.abs(deltaIIns)) > 0: # relative change of instrumental indicators
            deltaIIns = deltaIIns/np.sum(np.abs(deltaIIns))
        

        ### DETERMINE CONTRIBUTIONS ###
        changeF = F - Ft # change in benefits
        changeX = X - Xt # change in actions
        sign = np.sign(changeF*changeX) # direction of the next action
        changeF[changeF==0] = changeFt[changeF==0] # if the benefit did not change, keep the last change
        sign[sign==0] = signt[sign==0] # if the sign is undefined, keep the last one
        Xt = X.copy() # update lagged actions
        X = X + sign*np.abs(changeF) # determine current action
        assert np.sum(np.isnan(X)) == 0, 'X has invalid values!'
        C = P/(1 + np.exp(-X)) # map action into contribution
        assert np.sum(np.isnan(C)) == 0, 'C has invalid values!'
        assert np.sum(P < C)==0, 'C cannot be larger than P!'
        signt = sign.copy() # update previous signs
        changeFt = changeF.copy() # update previous changes in benefits
        
        tsC[:,T] = C # store this period's contributions
        tsF[:,T] = F # store this period's benefits
                
        
        ### DETERMINE BENEFITS ###
        if type(qm) is int or type(qm) is np.int64: # if the quality of monitoring is endogenous
            trial = (np.random.rand(n) < (I[qm]/1) * P/P.max() * (P-C)/P) # monitoring outcomes
        else: # if the quality of monitoring is exogenous
            trial = (np.random.rand(n) < qm * P/P.max() * (P-C)/P) # monitoring outcomes
        theta = trial.astype(float) # indicator function of uncovering inefficiencies
        H[theta==1] += (P[theta==1] - C[theta==1])/P[theta==1] # cumulative inefficiencies spotted
        HC[theta==1] += 1 # number of times spotted so far being inefficient
        if type(rl) is int or type(rl) is np.int64: # if the quality of the rule of law is endogenous
            newF = deltaIIns*C/P + (1-theta*(I[rl]/1))*(P-C)/P # compute benefits
        else: # if the quality of the rule of law is exogenous
            newF = deltaIIns*C/P + (1-theta*rl)*(P-C)/P # compute benefits
        Ft = F.copy() # update lagged benefits
        F = newF # update benefits
        assert np.sum(np.isnan(F)) == 0, 'F has invalid values!'
        
        
        ### DETERMINE INDICATORS ###
        deltaM = np.array([deltaBin,]*len(deltaBin)).T # reshape deltaIAbs into a matrix
        S = np.sum(deltaM*A, axis=0) # compute spillovers
        assert np.sum(np.isnan(S)) == 0, 'S has invalid values!'
        tsS[:,T] = S # store spillovers
        cnorm = np.zeros(N) # initialise a zero-vector to store the normalised contributions
        cnorm[R] = C # compute contributions only for instrumental nodes
        gammas = ( betas*(cnorm + C.sum()/(P.sum()+1)) )/( 1 + np.exp(-S) ) # compute probability of successful growth
        assert np.sum(np.isnan(gammas)) == 0, 'some gammas have invalid values!'
        assert np.sum(gammas==0) == 0, 'some gammas have zero value!'
        
        if frontier is not None: # if the user wants to perform frontier analysis
            gammas = frontier
            
        tsG[:,T] = gammas # store gammas
        success = (np.random.rand(N) < gammas) # determine if there is successful growth
        newI = I.copy() # compute potential new values
        newI[success] = I[success] + alphas[success] # update growing indicators
        newI[~success] = I[~success] - alphas_prime[~success] # update decreasing indicators
        
        # if theoretical maximums are provided, make sure the indicators do not surpass them
        if Imax is not None:
            with_bound = ~np.isnan(Imax)
            newI[with_bound & (newI[with_bound] > Imax[with_bound])] = Imax[with_bound & (newI[with_bound] > Imax[with_bound])]
            assert np.sum(newI[with_bound] > Imax[with_bound])==0, 'some indicators have surpassed their theoretical upper bound!'
        
        # if theoretical minimums are provided, make sure the indicators do not become lower than them
        if Imin is not None:
            with_bound = ~np.isnan(Imin)
            newI[with_bound & (newI[with_bound] < Imin[with_bound])] = Imin[with_bound & (newI[with_bound] < Imin[with_bound])]
            assert np.sum(newI[with_bound] < Imin[with_bound])==0, 'some indicators have surpassed their theoretical lower bound!'
                        
        # if governance parameters are endogenous, make sure they are not larger than 1
        if (type(qm) is int or type(qm) is np.int64) and newI[qm] > 1:
            newI[qm] = 1
        
        if (type(rl) is int or type(rl) is np.int64) and newI[rl] > 1:
            newI[rl] = 1
            
        # if governance parameters are endogenous, make sure they are not smaller than 0
        if (type(qm) is int or type(qm) is np.int64) and newI[qm] < 0:
            newI[qm] = 0
        
        if (type(rl) is int or type(rl) is np.int64) and newI[rl] < 0:
            newI[rl] = 0
            
        It = I.copy() # update lagged indicators
        I =  newI.copy() # update indicators
        
        
        ### DETERMINE ALLOCATION PROFILE ###
        P0 += np.random.rand(n)*H/HC # interaction between random term and inefficiency history
        assert np.sum(np.isnan(P0)) == 0, 'P0 has invalid values!'
        assert np.sum(P0==0) == 0, 'P0 has a zero value!'
        
        P = np.zeros(n)
        # iterate over the expenditure programs
        for i, program in enumerate(sorted_programs):
            indis = program2indis[program]
            relevant_indis = inst2idx[indis].astype(int)
            q = P0[relevant_indis]/P0[relevant_indis].sum() # compute expenditure propensities
            assert np.sum(np.isnan(q)) == 0, 'q has invalid values!'
            assert np.sum(q == 0 ) == 0, 'q has zero values!'
            qs_hat = q**bs[relevant_indis] # modulate expenditure propensities
            P[relevant_indis] += Bs[i,t]*qs_hat/qs_hat.sum()
            
        # optional assertion that checks for consistency between the total budget and the sum of the allocations
        # assert abs(P.sum() - Bs[:,t].sum()) < 1e-6, 'unequal budgets ' + str(abs(P.sum() - Bs[:,t].sum()))
        
        assert np.sum(np.isnan(P)) == 0, 'P has invalid values!'
        assert np.sum(P==0) == 0, 'P has zero values!'

    # return time series
    return tsI, tsC, tsF, tsP, tsS, tsG



    










## Calibrates PPI automatically and return a Pandas DataFrame with the parameters, errors, and goodness of fit
def calibrate(I0, IF, success_rates, A=None, R=None, qm=None, rl=None,  Bs=None, B_dict=None, 
              T=None, threshold=.8, parallel_processes=None, 
              verbose=False, low_precision_counts=101, increment=1000):

    """Function to calibrate the model parameters.

    Parameters
    ----------
        I0: numpy array 
            See run_ppi function.
        IF: numpy array 
            Vector with final values of the development indicators. These are
            used to compute one type of error: whether the simulated values
            end at the same levels as the empirical ones.
        success_rates: numpy array 
            Vector with the empirical rate of growth of the development indicators.
            A growth rate, for an indicator, is the number of times a positive 
            change is observed, divided by the the total number of changes 
            (which should be the number of observations in a time series minus one).
            These rates must be greater than zero and less or equal to 1.
            If an indicator does not show positive changes, it is suggested
            to assign a rate close to zero. If success_rates contains values 
            that are zero (or less) or ones, they will be automatically replaced
            by 0.01 and 1.0 respectively. This input is used to compute another 
            type of error: whether the endogenous probability of success of 
            each indicator matches its empirical rate of growth.
        A:  2D numpy array (optional)
            See run_ppi function.
        R:  numpy array (optional)
            See run_ppi function.
        bs: numpy array (optional)
            See run_ppi function.
        qm: A floating point, an integer, or a numpy array (optional)
            See run_ppi function.
        rl: A floating point, an integer, or a numpy array (optional)
            See run_ppi function.
        Bs: numpy ndarray (optional)
            See run_ppi function.
        B_dict: dictionary (optional)
            See run_ppi function.
        T: int (optional)
            See run_ppi function.
        threshold: float (optional)
            The goodness-of-fit threshold to stop the calibration routine. This
            consists of the worst goodness-of-fit metric (across indicators) that
            the user would like to obtain. The best possible metric is 1, but it
            is impossible to achieve due to the model's stochasticity. Higher
            thresholds demand more computing because more Monte Carlo simulations 
            are necessary to achieve high precision. If not provided, the default
            value is 0.8.
        parallel_processes: integer (optional)
            The number of processes to be run in parallel. Each process carries
            a work load of multiple Monte Carlo simulations of PPI. Parallel
            processing is optional and requires the JobLib library. 
            If not provided, the Monte Carlo simulations are run in a serial fashion.
        verbose: boolean (optional)
            Whether to print the calibration progress. If not provided, the
            default value is False.
        low_precision_counts: integer (optional)
            A hyperparameter of how many low-precision iterations will be run.
            Low precision means that only 10 Monte Carlo simulations are performed
            for each evaluation. Once low_precision_counts has been met,
            the number of Monte Carlo simulations increases in each iteration
            by the amount specified in the hyperparameter: increment. If not
            provided, the default value is 100.
        increment: integer (optional)
            A hyperparameter that sets the number of Montecarlo Simulations to
            increase with each iteration, once low_precision_counts has been
            reached. If not provided, the default value is 1000.
        
    Returns
    -------
        output: 2D numpy array
            A matrix with the calibration results, organised in the following 
            columns (the matrix includes the column titles):
                - alpha: The structural parameter of each indicator associated
                    with positive changes.
                - alpha_prime: The structural parameter of each indicator associated
                    with negative changes.
                - beta: The normalising constant of each parameter that helps 
                    mapping the expenditure and spillovers into a probability.
                - T: The number of simulation periods ran in in each Monte Carlo 
                    simulation. It only appears in the first row of the column, 
                    while the rest remain empty.
                - error_alpha: The indicator-specific error related to the 
                    final value.
                - error_beta: The indicator-specific error related to the 
                    rate of positive changes.
                - GoF_alpha: The indicator-specific goodness-of-fit metric 
                    related to the final value.
                - GoF_beta: The indicator-specific goodness-of-fit metric 
                    related to the rate of positive changes.
    """
    
    
    
    # Check data integrity
    success_rates[success_rates<=0] = 0.05
    success_rates[success_rates>=1] = .95
    assert threshold < 1, 'the threshold must be lower than 1'
    assert len(I0) == len(IF), 'I0 and IF must have the same size'
    
    
    # Initialise hyperparameters and containers
    N = len(I0)
    params = np.ones(3*N)*.5 # vector containing all the parameters that need calibration
    sample_size = 10
    counter = 0
    GoF_alpha = np.zeros(N)
    GoF_beta = np.zeros(N)
    
    # Main iteration of the calibration
    # Iterates until the minimum threshold criterion has been met, and at least 100 iterations have taken place
    while np.sum(GoF_alpha<threshold) > 0 or np.sum(GoF_beta<threshold) > 0 or counter < low_precision_counts:

        counter += 1 # Makes sure at least 100 iterations are performed
        
        # unpack the parameter vector into 3 vectors that correspond to the different parameter types
        alphas = params[0:N] 
        alphas_prime = params[N:2*N]
        betas = params[2*N::]
    
        # compute the errors for the specified parameter vector
        errors_all, TF = compute_error(I0=I0, IF=IF, success_rates=success_rates, alphas=alphas, alphas_prime=alphas_prime, betas=betas, A=A, 
                                        R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict, 
                                        T=T, parallel_processes=parallel_processes, 
                                        sample_size=sample_size)
        
        # unpack the error vector
        errors_alpha = errors_all[0:N]
        errors_beta = errors_all[N::]
        
        # normalise the errors
        abs_errors_alpha = np.abs(errors_alpha)
        gaps = IF-I0
        normed_errors_alpha = abs_errors_alpha/np.abs(gaps)
        abs_normed_errors_alpha = np.abs(normed_errors_alpha)
        abs_errors_beta = np.abs(errors_beta)
        normed_errors_beta = abs_errors_beta/success_rates
        abs_normed_errrors_beta = np.abs(normed_errors_beta)
        
        # apply the gradient descent and update the parameters
        params[0:N][(errors_alpha<0) & (IF>I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha<0) & (IF>I0)], .25, .99)
        params[0:N][(errors_alpha>0) & (IF>I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha>0) & (IF>I0)], 1.01, 1.5)
        params[N:2*N][(errors_alpha<0) & (IF>I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha<0) & (IF>I0)], 1.01, 1.5)
        params[N:2*N][(errors_alpha>0) & (IF>I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha>0) & (IF>I0)], .25, .99)
        params[0:N][(errors_alpha>0) & (IF<I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha>0) & (IF<I0)], 1.01, 1.5)
        params[0:N][(errors_alpha<0) & (IF<I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha<0) & (IF<I0)], .25, .99)
        params[N:2*N][(errors_alpha>0) & (IF<I0)] *= np.clip(1-abs_normed_errors_alpha[(errors_alpha>0) & (IF<I0)], .25, .99)
        params[N:2*N][(errors_alpha<0) & (IF<I0)] *= np.clip(1+abs_normed_errors_alpha[(errors_alpha<0) & (IF<I0)], 1.01, 1.5)
        params[2*N::][errors_beta<0] *= np.clip(1-abs_normed_errrors_beta[errors_beta<0], .25, .99)
        params[2*N::][errors_beta>0] *= np.clip(1+abs_normed_errrors_beta[errors_beta>0], 1.01, 1.5)
        
        # compute the goodness of fit
        GoF_alpha = 1 - normed_errors_alpha
        GoF_beta = 1 - abs_normed_errrors_beta
        
        # check low_precision_counts iterations have been reached
        # after low_precision_counts iterations, increase the number of Monte Carlo simulations by
        # 1000 in every iterations in order to achieve higher precision and 
        # minimise the error more effectively
        if counter >= low_precision_counts:
            sample_size += increment
            
        # prints the calibration iteration and the worst goodness-of-fit metric
        if verbose:
            print( 'Iteration:', counter, '.    Worst goodness of fit:', np.min(GoF_alpha.tolist()+GoF_beta.tolist()) )
    
    # save the last parameter vector and de associated errors and goodness-of-fit metrics
    output = np.array([['alpha', 'alpha_prime', 'beta', 'T', 'error_alpha', 
            'error_beta', 'GoF_alpha', 'GoF_beta']] + [[alphas[i], alphas_prime[i], betas[i], TF, 
            errors_alpha[i], errors_beta[i], GoF_alpha[i], GoF_beta[i]] \
            if i==0 else [alphas[i], alphas_prime[i], betas[i], 
             np.nan, errors_alpha[i], errors_beta[i], 
             GoF_alpha[i], GoF_beta[i]] \
            for i in range(N)])
        
    return output
    










## Computes a set of Monte Carlo simulations of PPI, obtains their average statistics, 
## and computes the error with respect to IF and success_rates. 
## Called by the calibrate function.
def compute_error(I0, IF, success_rates, alphas, alphas_prime, betas, A=None, 
                  R=None, qm=None, rl=None, Bs=None, B_dict=None, T=None, 
                  parallel_processes=None, sample_size=1000):

    """Function to evaluate the model and compute the errors.

    Parameters
    ----------
        I0: numpy array 
            See run_ppi function.
        IF: numpy array 
            See calibrate function.
        success_rates: numpy array 
            See calibrate function.
        alphas: numpy array
            See run_ppi function.
        alphas_prime: numpy array
            See run_ppi function.
        betas: numpy array
            See run_ppi function.
        A:  2D numpy array (optional)
            See run_ppi function.
        R:  numpy array (optional)
            See run_ppi function.
        bs: numpy array (optional)
            See run_ppi function.
        qm: A floating point, an integer, or a numpy array (optional)
            See run_ppi function.
        rl: A floating point, an integer, or a numpy array (optional)
            See run_ppi function.
        Bs: numpy ndarray (optional)
            See run_ppi function.
        B_dict: dictionary (optional)
            See run_ppi function.
        T: int (optional)
            See run_ppi function.
        parallel_processes: integer (optional)
            See calibrate function.
        sample_size: integer (optional)
            Number of Monte Carlo simulations to be ran.
        
    Returns
    -------
        errors: 2D numpy array
            A matrix with the error of each parameter. The first column contains
            the errors associated to the final values of the indicators. The second
            provides the errors related to the empirical probability of growth.
        TF: integer
            The number of periods that the model ran in each Monte Carlo simulation.
    """
    
    
    
    if parallel_processes is None:
        sols = np.array([run_ppi(I0=I0, alphas=alphas, alphas_prime=alphas_prime, 
                          betas=betas, A=A, R=R, qm=qm, rl=rl,
                          Bs=Bs, B_dict=B_dict, T=T) for itera in range(sample_size)])
    else:
        sols = np.array(Parallel(n_jobs=parallel_processes, verbose=0)(delayed(run_ppi)\
                (I0=I0, alphas=alphas, alphas_prime=alphas_prime, betas=betas, 
                 A=A, R=R, qm=qm, rl=rl, Bs=Bs, B_dict=B_dict, T=T) for itera in range(sample_size)))
        
    tsI, tsC, tsF, tsP, tsS, tsG = zip(*sols)
    I_hat = np.mean(tsI, axis=0)[:,-1]
    gamma_hat = np.mean(tsG, axis=0).mean(axis=1)
    error_alpha = IF - I_hat
    error_beta = success_rates - gamma_hat
    
    errors = np.array(error_alpha.tolist() + error_beta.tolist())
    TF = tsI[0].shape[1]
    
    return errors, TF











































