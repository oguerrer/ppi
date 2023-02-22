# PPI Source Code

The `ppi.py` file contains all the code of PPI's agent-computing model, as well as helper functions to parallelise the Monte Carlo simulations, and the calibration algorithm.


FUNCTIONS:


    calibrate(I0, IF, success_rates, A=None, R=None, bs=None, qm=None, rl=None, Bs=None, B_dict=None, T=None, threshold=0.8, parallel_processes=None, verbose=False, low_precision_counts=101, increment=1000)
        Function to calibrate the model parameters.
        
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
    
    compute_error(I0, IF, success_rates, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None, Bs=None, B_dict=None, G=None, T=None, parallel_processes=None, sample_size=1000)
        Function to evaluate the model and compute the errors.
        
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
    
    run_ppi(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None, Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=None, frontier=None)
        Function to run one simulation of the Policy Priority Inference model.
        
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
    
    run_ppi_parallel(I0, alphas, alphas_prime, betas, A=None, R=None, bs=None, qm=None, rl=None, Imax=None, Imin=None, Bs=None, B_dict=None, G=None, T=None, frontier=None, parallel_processes=4, sample_size=1000)
        Function to run a sample of evaluations in parallel. As opposed to the function
        run_ppi, which returns the output of a single realisation, this function returns
        a set of time series (one for each realisation) of each output type.
        
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
            G: numpy array (optional)
                See run_ppi function.
            T: int (optional)
                See run_ppi function.
            frontier: boolean
                See run_ppi function.
            parallel_processes: integer (optional)
                See calibrate function.
            sample_size: integer (optional)
                Number of Monte Carlo simulations to be ran.
            
        Returns
        -------
            tsI: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.
            tsC: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.
            tsF: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.
            tsP: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.
            tsS: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.
            tsG: list
                A list with multiple matrices, each one containing the time series 
                of multiple realisations of the simulated indicators. In each matrix, 
                each row corresponds to an indicator and each column to a simulation step.

