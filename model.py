import numpy as np
from copy import deepcopy
import scipy.io
import pandas as pd
from tqdm import tqdm
import subprocess
import os

class model:
    """
    Class for the Talento-Ganopolski (2021) model.
    """

    def __init__(self, 
                params=None, 
                sigma=0.0, 
                sigma_start=200, 
                sigma_end = -1, 
                seed=-1,
                kyr=1000,
                t_start=-1000,
                v_0=0.05, 
                CO2_0=300., 
                CO2_min=150.,
                dt=1, 
                memory_length=30, 
                Mv_0=0., 
                CO2_forcing=None,
                v_forcing=None,
                T_forcing=None,
                mid_bruhnes=True,
                free_model=False
        ):
        """Init Talento-Ganopolski (2021) model

        Args:
            params (dict, optional): changes to params. Defaults to None.
            sigma (float, optional): Set noise factor. I often used 0.001. Defaults to 0.
            sigma_start (int, optional): When to start adding noise in units kyr from start of simulation. Defaults to 200.
            sigma_end (int, optional): When to stop adding noise in units kyr from start of simulation. Defaults to -1, meaning end of simulation.
            seed (int, optional): Seed for the random number generator. Defaults to 1.
            kyr (int, optional): Length of the simulation in kyr. Defaults to 1000.
            t_start (int, optional): Starting time. Will load orbital forcing from this point onwards. Defaults to -1000 kyr.
            v_0 (float, optional): Init ice-volume value. Defaults to 0.05.
            CO2_v (float, optional): Init CO2 value. Defaults to 300..
            CO2_min (float, optional): Minimum CO2 value, constraint 11 from paper. Defaults to 150. 
            dt (float, optional): Time step size in kyr. Defaults to 1.
            memory_length (int, optional): Length of the memory term in the ice equation.
            Mv (float, optional): Init value for memory term. Defaults to 0. Used for starting simulation from a given state.
            CO2_forcing (float-array, optional): Array of CO2 forcing values. Defaults to None.
            v_forcing (float-array, optional): Array of v forcing values. Defaults to None.
            T_forcing (float-array, optional): Array of T forcing values. Defaults to None.
            mid_bruhnes (bool, optional): If set to true, ice-volume is kept above 0.05 from -410kyr to 0. Automatically laods appropriate params. Defaults to True.
            noise_type (str, optional): Type of noise. Defaults to NOISE_TYPE_ADDITIVE.
            free_model (bool, optional): If set to true, we set params=get_MCMC_params_no_MB_final, CO2_min=100, mid_bruhnes=False. Defaults to False.
        """

        self.sigma = sigma
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.v_0 = v_0
        self.CO2_0 = CO2_0
        self.CO2_min = CO2_min
        self.dt = dt
        self.seed = seed
        self.CO2_forcing = CO2_forcing
        self.v_forcing = v_forcing
        self.T_forcing = T_forcing
        self.mid_bruhnes = mid_bruhnes
        self.Mv_0 = Mv_0
        self.kyr = kyr
        self.t_start = t_start
        self.ensemble = None  # for later initialization
        
        # parameters fix by constraints
        self.c4 = 278. 
        self.d1 = -3.
        self.d2 = 5.56
        
        # Init params
        # defaults = get_MCMC_params()        # load default params
        defaults = get_MCMC_params_final()  # load default params
        
        # update params with user params
        if params is not None:              # Update defaults with any provided parameters
            defaults.update(params)         #
        for key, value in defaults.items(): # Assign to instance variables
            setattr(self, key, value)       #

        # set random seed if nothing provided
        if self.seed == -1:
            self.seed = np.random.randint(0, int(1e8))

        # Initialize state variables 
        self.init_state_variables(seed=self.seed)

        # divide sigma start and end with dt such that it is specified in kyr
        self.sigma_start = sigma_start / self.dt
        self.sigma_end = sigma_end / self.dt

        # set noise range end to last time step if nothing provided
        if self.sigma_end == -1/self.dt:
            self.sigma_end = self.N_time_steps

        # Divide memory length by dt such that it is specified in kyr
        self.tau = int(memory_length / self.dt)

        # set up orbital forcing
        self.f = load_orbital_forcing()
        self.f_mean = np.mean(self.f)
        
        assert self.N_time_steps <= len(self.f), "Runtime exceeds available orbital forcing data"


    def init_state_variables(self, seed):
        ''' Init state variables, called seperatedly from __init__ to allow for reinitialization of the model'''
        
        # time steps
        self.N_time_steps = int(self.kyr/self.dt)
        self.t_range = np.arange(-1000, self.kyr - 1000, self.dt)  # from -1Myr to 0

        # state vars
        self.v = np.full(self.N_time_steps, self.v_0)          # Global ice volume anomaly rel. to preindustrial state
        self.CO2 = np.full(self.N_time_steps, self.CO2_0)
        self.T = np.zeros(self.N_time_steps)
        self.dvdt = np.zeros(self.N_time_steps)
        self.Mv = np.full(self.N_time_steps, self.Mv_0)

        # init noise generator
        self.RandomGenerator = np.random.default_rng(seed=seed)
        self.additive_noise = self.RandomGenerator.normal(0, np.sqrt(self.dt), self.N_time_steps) * self.sigma

    def update(self, t): 
        
        # Equation: Delta T
        self.T[t] = self.d1 * self.v[t-1] + self.d2 * np.log(self.CO2[t-1] / self.c4);  # sign for d1 is in d1 itself
        
        # Equation: CO2
        dvdt = (self.v[t-1] - self.v[t-2])/self.dt
        self.CO2[t] = self.c1 * self.T[t] + self.c2 * self.v[t-1] + self.c3 * min(dvdt, 0) + self.c4
        self.CO2[t] = max(self.CO2[t], self.CO2_min) # constraint (13), should be 150ppm

        # Equation: ice mass-balance
        delta = 1 if dvdt < 0 else 0
        if self.tau > 0:
            M_v = delta * np.trapz(self.v[t-1-self.tau:t-1]) / self.tau  # memory term
        else:
            M_v = np.float16(0)  # needed for the M_v.copy() below
        self.dvdt[t] = (self.b1 * self.v[t-1] - self.b2 * self.v[t-1]**(3/2) - self.b3 * (self.f[t-1] - self.f_mean) - self.b4 * np.log(self.CO2[t-1])) / (1 - self.b5 * M_v) + self.b6
        
        # Additive ice volume noise
        if self.sigma > 0 and t > self.sigma_start and t < self.sigma_end:
            self.dvdt[t] += self.additive_noise[t]  # noise term according to Euler-Maruyama method

        # compute v and save memory term
        self.v[t] =  self.v[t-1] + self.dvdt[t] * self.dt
        self.Mv[t] = M_v.copy()

        # CO2 forcing
        if self.CO2_forcing:
            self.CO2[t] += self.CO2_forcing[t]

        # v (ice volume) forcing
        if self.v_forcing:
            self.v[t] += self.v_forcing[t]

        # T (temperature) forcing
        if self.T_forcing:
            self.T[t] += self.T_forcing[t]
 
        # mid bruhnes ice cover constraint
        min_v = 0.05 if self.t_range[t] < -410 and self.mid_bruhnes else 0.0
        self.v[t] = max(self.v[t], min_v)  # constraints (11, 12)



    def run(self, crop_to_paleo=True, crop_Mv=False):
        """Run the model
        Args:
            crop_to_paleo (bool, optional): Crop off first 200kyr if true. Defaults to True.
            crop_Mv (bool, optional): Crop off first 30 kyr or length of memory term. if crop_to_paleo is true, this is irrelevant. Defaults to False. 
        """

        custom_start = (int(1000/self.dt) + int(self.t_start/self.dt)) # deafults to 0
        start_after_memory_term = self.tau+1
        start = np.max([custom_start, start_after_memory_term])  # start after memory term or at custom start time
        for t in range(start, len(self.v)):
            self.update(t)

        # first 200kyr are just for getting the model to equilibrium
        if crop_to_paleo:
            kyr200 = int(200 / self.dt) + custom_start
            self.v = self.v[kyr200:]
            self.dvdt = self.dvdt[kyr200:]
            self.T = self.T[kyr200:]
            self.CO2 = self.CO2[kyr200:]
            self.t_range = self.t_range[kyr200:]
            self.Mv = self.Mv[kyr200:]
        # else, just crop off init time 
        elif crop_Mv:
            self.v = self.v[self.tau:]
            self.dvdt = self.dvdt[self.tau:]
            self.T = self.T[self.tau:]
            self.CO2 = self.CO2[self.tau:]
            self.t_range = self.t_range[self.tau:]
            self.Mv = self.Mv[self.tau:]

        return self
    

    def run_ensemble(self, N_runs, crop_to_paleo=True, crop_Mv=False, seed=1):
        """Run multiple simulations of the model with different random seeds.

        Args:
            N_runs (int): Number of ensemble members to generate
            crop_to_paleo (bool, optional): Crop off first 200kyr if true. Defaults to True.
            crop_Mv (bool, optional): Crop off first 30 kyr or length of memory term if true. Defaults to False.
            seed (int, optional): Random seed for reproducibility. Defaults to 1.

        Returns:
            model: Returns self with ensemble results stored in model variables (v, CO2, T, etc)
                  as numpy arrays of shape (N_runs, timesteps)
        """
        
        self.N_runs = N_runs
        M = []

        np.random.seed(seed)
        run = 0
        while run < N_runs:

            # make new rand seed
            randint = np.random.randint(0, int(1e8))

            # reset model vars
            self.init_state_variables(randint)
        
            # run again
            m = self.run(crop_to_paleo=crop_to_paleo, crop_Mv=crop_Mv)
            M.append(deepcopy(m))
            run += 1
            

        # save ensemble to model variables
        self.v = np.asarray([m.v for m in M])
        self.CO2 = np.asarray([m.CO2 for m in M])
        self.T = np.asarray([m.T for m in M])
        self.t_range = np.asarray([m.t_range for m in M])
        self.dvdt = np.asarray([m.dvdt for m in M])
        self.Mv = np.asarray([m.Mv for m in M])
        return self

    def ensemble_median(self, variable="v"):
        """Calculate the median across ensemble members at each timestep.

        Args:
            variable (str, optional): Name of model variable to calculate median for. Defaults to "v".

        Returns:
            numpy.ndarray: Array of median values at each timestep
        """
        median = []
        variable = getattr(self, variable)
        for i in range(len(variable[0])):
            median += [np.median(variable[:, i])]
        return np.asarray(median)

# end of model

######################################################
#     PARAMERTERS FOR THE TALENTO MODEL              #
######################################################


def get_param_names():
    return ['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'c1', 'c2', "c3"]

def get_MCMC_params_final():
    '''
    Gotten by running the MCMC fit 1000 times with 5000 itereations each.
    Identified in Talento_opt_many_runs.ipynb

    fits/MCMC_standard/fit533_description.txt
     
        n_iterations:           4999
        Accepted perturbations: 1064
        i_max:                  725
        Final correlation:      0.8745148687932637
        Highest correlation:    0.8920509488506244
        -------
        Details of last solution: 
        v[775]:                 0.9345223808026186
        v_min:                  0.0
        v_max:                  0.9645362137333332
        CO2_min                 150.0
        CO2_max                 277.9999989270859
        T_min                   -6.32404979814476
        T_max                   -5.897020334008767e-08
        mid_bruhnes             True
    
    '''

    return {'b1': 0.21225019974916653,
            'b2': 0.2836925345028339,
            'b3': 0.0008174423131295259,
            'b4': 0.095,
            'b5': 0.18964641973723284,
            'b6': 0.5261626610099003,
            'c1': 17.824476083652947,
            'c2': -24.49497200960442,
            'c3': -47.76302777426819}

######################################################
#     HELPER FUNCTIONS FOR THE TALENTO MODEL         #
######################################################

def load_orbital_forcing():
    smx_p = scipy.io.loadmat("./data/orbital_forcing/smx_p_2000kyr.mat")["smx_p"][0]
    smx_o = scipy.io.loadmat("./data/orbital_forcing/smx_o_2000kyr.mat")["smx_o"][0]
    val = 1.04
    f = smx_p + val * (smx_o - np.mean(smx_o))
    return f

def recovery_time(x, x_perturb, atol=0.000001):
    """Compute time it takes for a perturbed trajectory to return to the baseline trajectory.
       The code iterates over the trajectories and check if they are within relative tolerance.
       Only once the entire rest of the each trajectory match do I consider the system to have recovered.
       Trajectories may namely at times be close to each other, but then diverge again."""

    assert len(x) < 3000, "this is not updated for dt < 1"

    def equal(x, y):
        return np.allclose(x, y, atol=atol)

    # check if there is a perturbation, if not return 0
    if equal(x, x_perturb):
        return (0, 0)

    length = 0
    for i in range(len(x)):
        # if not in perturbation, check if ith values are equal
        if length == 0:
            if not equal(x[i], x_perturb[i]):
                length += 1

        # if in perturbation, check if the rest of the paths are equal
        else:
            if not equal(x[i:], x_perturb[i:]):
                length += 1
            else:
                # return length and the index of recovery
                return length, i

    return (
        10_000,
        -1,
    )  # if perturbation does not end in time series, return 10_000 and -1



def compute_recovery_times(
    var,
    m_baseline,
    perturbations,
    kyr=1000,
    mid_bruhnes=True,
    CO2_min=150,
    params=None,
    atol=0.000001,
    crop_to_paleo=True,
    perturb_range=(200, 1000),
    fast_recovery_threshold=10,
    record_v=False,
):
    """Compute recovery times for a given variable, baseline model, and perturbations.
    
    Args:
        var (str): Variable to compute recovery times for, must be one of 'CO2', 'v', or 'T'
        m_baseline (model): Baseline model instance
        perturbations (list): List of perturbation values to apply
        kyr (int, optional): Length of simulation in kyr. Defaults to 1000.
        mid_bruhnes (bool, optional): Whether to apply mid-Bruhnes constraint. Defaults to True.
        CO2_min (int, optional): Minimum allowed CO2 value. Defaults to 150.
        params (dict, optional): Model parameters to override defaults. Defaults to None.
        atol (float, optional): Absolute tolerance for recovery time calculation. Defaults to 0.000001.
        crop_to_paleo (bool, optional): Whether to crop output to paleo period. Defaults to True.
        perturb_range (tuple, optional): Range of times to start perturbations. Defaults to (200, 1000).
        fast_recovery_threshold (int, optional): Threshold for fast recovery. Defaults to 10.
        record_v (bool, optional): Whether to record ice volume trajectories. Defaults to False.

    Returns:
        tuple: Tuple containing DataFrame with recovery times, list of all recovery lengths, list of all recovery years, and list of all ice volume trajectories.
    """


    L_min, L_mean, L_median, L_max = [], [], [], []  # length
    Y_min, Y_mean, Y_median, Y_max = [], [], [], []  # year

    perturbation_times = np.arange(*perturb_range)

    # iterate to start perturbation at different times
    all_L = []
    all_Y = []
    all_v = []
    fast_rec_ratios = []
    for i in tqdm(perturbation_times):
        L = []
        Y = []
        for per in perturbations:
            # run model and compute recovery length and year
            m = run_with_forcing(
                var,
                i,
                per,
                kyr=kyr,
                mid_bruhnes=mid_bruhnes,
                CO2_min=CO2_min,
                params=params,
                crop_to_paleo=crop_to_paleo,
            )
            length, index = recovery_time(
                getattr(m_baseline, var), getattr(m, var), atol=atol
            )
            year = (
                m.t_range[index] if index > -1 else 10_000
            )  # index will be -1 if perturbation does not converge, set to a high number for year
            L += [length]
            Y += [year]
            if record_v:
                all_v += [m.v]  # can be quite memory intensive I think

        # compute min, mean, median and max
        L_min += [np.nanmin(L)]
        L_mean += [np.nanmean(L)]
        L_median += [np.nanmedian(L)]
        L_max += [np.nanmax(L)]
        Y_min += [np.nanmin(Y)]
        Y_mean += [np.nanmean(Y)]
        Y_median += [np.nanmedian(Y)]
        Y_max += [np.nanmax(Y)]
        all_L += [L]
        all_Y += [Y]

        # Count fraction of paths that recovered within fast_recovery_threshold = 10 kyr (default)
        fast_rec_ratios += [
            len(np.where(np.asarray(L) < fast_recovery_threshold)[0]) / len(L)
        ]

    df = pd.DataFrame(
        {
            "time": perturbation_times,
            "L_min": L_min,
            "L_mean": L_mean,
            "L_median": L_median,
            "L_max": L_max,
            "Y_min": Y_min,
            "Y_mean": Y_mean,
            "Y_median": Y_median,
            "Y_max": Y_max,
            f"fast_recovery_ratio_{fast_recovery_threshold}": fast_rec_ratios,
        }
    )

    return df, all_L, all_Y, all_v


def run_recovery_times_experiment(folder, steps = 20):
    """Run recovery times experiment.

    Args:
        folder (str): Folder to save results to
        steps (int, optional): Number of steps in perturbation range. Defaults to 20.
    """

    # create folder
    if os.path.exists(folder):
        print(f"Folder {folder} already exists — aborting")
        return
    else:
        os.makedirs(folder)

    perturb_range_lower, perturb_range_upper = -0.01, 0.01
    perturbations = np.linspace(perturb_range_lower, perturb_range_upper, steps)
    perturb_range = (200, 1000)
    kyr = 2000

    print("Computing recovery times")
    df, L, Y, V = compute_recovery_times(
        "v",
        model(kyr=kyr).run(),
        perturbations,
        kyr=kyr,
        perturb_range=perturb_range,
        record_v=True,
    )

    df.to_csv(f"{folder}/recovery_times.csv")
    filename = f"{folder}/recovery_times"

    filecomment = f"Ice volume perturbations in range v = [{perturb_range_lower}, {perturb_range_upper}] with {steps} steps, model runs for {kyr} kyr, perturbations in the time range {perturb_range}"
    save_txt(L, filename + "_L", filecomment="Length of all pertubations:\n" + filecomment)
    save_txt(
        Y,
        filename + "_Y",
        filecomment="Year of recovery for all perturbations:\n" + filecomment,
    )
    save_txt(
        V,
        filename + "_V",
        filecomment="Every ice-volume trajectory from perturbation computation:\n"
        + filecomment,
    )



def run_with_forcing(
    var,
    t,
    perturbation,
    kyr=1000,
    mid_bruhnes=True,
    crop_to_paleo=True,
    params=None,
    CO2_min=150,
):
    """Run model with a single perturbation applied to a specified variable at time t.

    Args:
        var (str): Variable to perturb, must be one of 'CO2', 'v', or 'T'
        t (int): Time step at which to apply the perturbation
        perturbation (float): Size of perturbation to apply
        kyr (int, optional): Length of simulation in kyr. Defaults to 1000.
        mid_bruhnes (bool, optional): Whether to apply mid-Bruhnes constraint. Defaults to True.
        crop_to_paleo (bool, optional): Whether to crop output to paleo period. Defaults to True.
        params (dict, optional): Model parameters to override defaults. Defaults to None.
        CO2_min (int, optional): Minimum allowed CO2 value. Defaults to 150.

    Returns:
        model: Model instance after running with the specified perturbation

    Raises:
        ValueError: If var is not one of 'CO2', 'v', or 'T'
    """

    var_forcing = np.zeros(2000)
    var_forcing[t] = perturbation

    if var == "CO2":
        m = model(
            sigma=0,
            kyr=kyr,
            params=params,
            CO2_min=CO2_min,
            CO2_forcing=list(var_forcing),
            mid_bruhnes=mid_bruhnes,
        ).run(crop_to_paleo=crop_to_paleo)
    elif var == "v":
        m = model(
            sigma=0,
            kyr=kyr,
            params=params,
            CO2_min=CO2_min,
            v_forcing=list(var_forcing),
            mid_bruhnes=mid_bruhnes,
        ).run(crop_to_paleo=crop_to_paleo)
    elif var == "T":
        m = model(
            sigma=0,
            kyr=kyr,
            params=params,
            CO2_min=CO2_min,
            T_forcing=list(var_forcing),
            mid_bruhnes=mid_bruhnes,
        ).run(crop_to_paleo=crop_to_paleo)
    else:
        raise ValueError("var must be one of CO2, v, T")
    return m


def save_txt(data, name, ext="txt", filecomment=None):
    """Save data to a text file.

    Args:
        data (numpy.ndarray): Data to save
        name (str): Name of file to save
        ext (str, optional): Extension of file. Defaults to "txt".
        filecomment (str, optional): Comment to add to file. Defaults to None.
    """

    filepath = f"{name}.{ext}"
    np.savetxt(filepath, data, delimiter=",")

    if filecomment:
        script = f'''
        tell application "Finder"
            set comment of (POSIX file "{filepath}" as alias) to "{filecomment}"
        end tell
        '''
        subprocess.run(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )



def compute_RAR(x_baseline, x_trajectories, atol=0.001):
    """Computes RAR between a baseline time series and a list of time series, given the supplied absolute tolerance.

    Args:
        x_baseline (numpy.ndarray): Baseline time series
        x_trajectories (list): List of time series to compare to baseline
        atol (float, optional): Absolute tolerance for comparison. Defaults to 0.001.
    """
    overlap = np.zeros_like(x_baseline)
    for i in range(len(x_baseline)):
        is_close_list = np.zeros(len(x_trajectories))
        for j in range(len(x_trajectories)):
            is_close_list[j] = np.abs(x_baseline[i] - x_trajectories[j][i]) < atol
        overlap[i] = np.sum(is_close_list) / len(x_trajectories)
    return overlap



def load_paleo_data():
    """Load paleo data.

    Returns:
        dict: Dictionary containing paleo data
    """

    # load Paleo temperature records
    paleo_T_snyder = scipy.io.loadmat("./data/paleodata/delta_t_snyder.mat")[
        "delta_t_snyder"
    ]
    paleo_T_snyder = np.concatenate(paleo_T_snyder)
    paleo_T_fridrich = scipy.io.loadmat("./data/paleodata/delta_t_fridrich.mat")[
        "delta_t_model_friedrich"
    ]
    paleo_T_fridrich = np.concatenate(paleo_T_fridrich)
    paleo_T_fridrich_x = np.arange(-784, 0)

    x_range = np.arange(-800, 0)

    # Load paleo co2 records
    paleo_co2 = scipy.io.loadmat("./data/paleodata/co2.mat")["co2"][0]

    # Load paleo ice volume records
    s = scipy.io.loadmat("./data/paleodata/sea-level.mat")["s"][0]  # Actually sea level
    paleo_v = (
        -s.T * 0.001 * 3.618e2
    )  # Convert to ice volume equivalent, by multiplication with ocean area. Although since we normalise, it's irrelevant
    paleo_v = paleo_v / np.max(paleo_v)  # Normalise

    # make a normalised version where all values are between 0 and 1
    vv = -s  # invert
    vv = vv - np.min(vv)  # make min = 0
    vv = vv / np.max(vv)  # normalise to [0, 1]

    return {
        "T_snyder": paleo_T_snyder,
        "T_fridrich": paleo_T_fridrich,
        "T_fridrich_time": paleo_T_fridrich_x,
        "CO2": paleo_co2,
        "ice_volume": paleo_v,
        "v_norm": vv,
        "sea_level": s,
        "time": x_range,
    }


def frequency_spectra(x):
    """Compute frequency spectra of a time series.

    Args:
        x (numpy.ndarray): Time series

    Returns:
        tuple: Tuple containing frequency range and power spectrum
    """

    T = 1  # Sampling period
    Fs = 1 / T  # Sampling frequency (1 every 1000 yr.)
    L = len(x)  # Length of signal

    # FFT
    n = (
        2 ** int(np.ceil(np.log2(L)))
    )  # this is to improve the FFT algorithm, complete with zeros until the next power of 2
    Y = np.fft.fft(x, n)
    P2 = np.abs(Y / n) ** 2
    P1 = P2[: n // 2 + 1]
    P1[1:-1] = 2 * P1[1:-1]
    ff = np.linspace(0, Fs / 2, n // 2 + 1)[
        :-1
    ]  # remove last element to match MATLAB's implementation
    with np.errstate(
        divide="ignore", invalid="ignore"
    ):  # this ignore divide by zero warnings
        aux = 1 / ff  # Period (kyr)

    P1 = P1[3 : n // 2]  # drop first couple of compnents
    range_ = aux[3 : n // 2]

    return range_, P1