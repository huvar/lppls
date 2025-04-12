from multiprocessing import Pool
from matplotlib import pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import random
from datetime import datetime as date
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import xarray as xr
from typing import Any, Dict, Optional
import warnings


class LPPLS(object):

    def __init__(self, observations):
        """
        Args:
            observations (np.array,pd.DataFrame): 2xM matrix with timestamp and observed value.
        """
        assert isinstance(
            observations, (np.ndarray, pd.DataFrame)
        ), f"Expected observations to be <pd.DataFrame> or <np.ndarray>, got :{type(observations)}"

        self.observations = observations
        self.coef_ = {}
        self.indicator_result = []

    @staticmethod
    @njit
    def lppls(t, tc, m, w, a, b, c1, c2):
        dt = np.abs(tc - t) + 1e-8
        return a + np.power(dt, m) * (
            b + ((c1 * np.cos(w * np.log(dt))) + (c2 * np.sin(w * np.log(dt))))
        )

    def func_restricted(self, x, *args):
        """
        Finds the least square difference.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
        Args:
            x(np.ndarray):  1-D array with shape (n,).
            args:           Tuple of the fixed parameters needed to completely specify the function.
        Returns:
            (float)
        """
        tc = x[0]
        m = x[1]
        w = x[2]
        observations = args[0]

        rM = self.matrix_equation(observations, tc, m, w)
        a, b, c1, c2 = rM[:, 0].tolist()
        # print('type', type(res))
        # print('func_restricted', res)

        delta = self.lppls(observations[0, :], tc, m, w, a, b, c1, c2)
        delta = np.subtract(delta, observations[1, :])
        delta = np.power(delta, 2)
        return np.sum(delta)

    @staticmethod
    @njit
    def matrix_equation(observations, tc, m, w):
        """
        Derive linear parameters in LPPLs from nonlinear ones.
        """
        T = observations[0]
        P = observations[1]
        N = len(T)

        dT = np.abs(tc - T) + 1e-8
        phase = np.log(dT)

        fi = np.power(dT, m)
        gi = fi * np.cos(w * phase)
        hi = fi * np.sin(w * phase)

        fi_pow_2 = np.power(fi, 2)
        gi_pow_2 = np.power(gi, 2)
        hi_pow_2 = np.power(hi, 2)

        figi = np.multiply(fi, gi)
        fihi = np.multiply(fi, hi)
        gihi = np.multiply(gi, hi)

        yi = P
        yifi = np.multiply(yi, fi)
        yigi = np.multiply(yi, gi)
        yihi = np.multiply(yi, hi)

        matrix_1 = np.array(
            [
                [N, np.sum(fi), np.sum(gi), np.sum(hi)],
                [np.sum(fi), np.sum(fi_pow_2), np.sum(figi), np.sum(fihi)],
                [np.sum(gi), np.sum(figi), np.sum(gi_pow_2), np.sum(gihi)],
                [np.sum(hi), np.sum(fihi), np.sum(gihi), np.sum(hi_pow_2)],
            ]
        )

        matrix_2 = np.array(
            [[np.sum(yi)], [np.sum(yifi)], [np.sum(yigi)], [np.sum(yihi)]]
        )

        matrix_1 += 1e-8 * np.eye(matrix_1.shape[0])

        return np.linalg.solve(matrix_1, matrix_2)


    def fit(self, max_searches, minimizer="Nelder-Mead", obs=None):
        """
        Fits the LPPLS model to the observations using multiple random seeds.

        MODIFIED: Now returns sse and residuals from estimate_params.

        Args:
            max_searches (int): Max optimization attempts before giving up.
            minimizer (str): Scipy optimization method.
            obs (Mx2 numpy array): Observed time-series data. Uses self.observations if None.

        Returns:
            tuple: (tc, m, w, a, b, c, c1, c2, O, D, sse, residuals) on success,
                   (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None) on failure.
                   - O (float): Number of oscillations.
                   - D (float): Damping factor.
                   - sse (float/None): Sum of Squared Errors of the fit.
                   - residuals (np.array/None): Array of residuals.
        """
        if obs is None:
            obs = self.observations
        # Ensure obs has at least 2 data points for fitting
        if obs.shape[1] < 2:
             # print("Warning: Need at least 2 data points to fit.")
             return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None


        search_count = 0
        t1 = obs[0, 0]
        t2 = obs[0, -1]

        while search_count < max_searches:
            # Set random initialization limits for non-linear params
            init_limits = [
                (t2 - 0.2 * (t2 - t1), t2 + 0.2 * (t2 - t1)), # tc
                (0.1, 1.0),                                  # m
                (6.0, 13.0),                                 # ω (Note: Original code default)
            ]
            # Randomly choose vals within bounds
            try:
                 tc_init = random.uniform(init_limits[0][0], init_limits[0][1])
                 m_init = random.uniform(init_limits[1][0], init_limits[1][1])
                 w_init = random.uniform(init_limits[2][0], init_limits[2][1])
                 seed = np.array([tc_init, m_init, w_init])
            except OverflowError:
                 # Handle potential OverflowError if t1, t2, or tc are extremely large/small
                 search_count += 1
                 continue


            try:
                # Call estimate_params (modified to return sse, residuals)
                tc, m, w, a, b, c, c1, c2, sse, residuals = self.estimate_params(obs, seed, minimizer)

                # Calculate O and D safely
                O = self.get_oscillations(w, tc, t1, t2) if tc not in [t1, t2] else np.nan
                D = self.get_damping(m, w, b, c) if c != 0 and w != 0 else np.nan # Avoid division by zero

                return tc, m, w, a, b, c, c1, c2, O, D, sse, residuals

            except (ValueError, UnboundLocalError, np.linalg.LinAlgError) as e:
                # Catch optimization failures or linear algebra errors
                # print(f"Fit attempt {search_count + 1} failed: {e}")
                search_count += 1
            except Exception as e:
                # Catch any other unexpected error during fitting
                # print(f"Unexpected error during fit attempt {search_count + 1}: {e}")
                search_count += 1 # Increment count to avoid infinite loop

        # Return zeros/None if max_searches reached without success
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None

    def estimate_params(self, observations, seed, minimizer):
        """
        Estimates LPPLS parameters using scipy.optimize.minimize.

        MODIFIED: Now also calculates and returns SSE and residuals.

        Args:
            observations (np.ndarray): The observed time-series data slice (2xN).
            seed (list): Initial guess for [tc, m, w].
            minimizer (str): Minimization algorithm for scipy.optimize.minimize.

        Returns:
            tuple: (tc, m, w, a, b, c, c1, c2, sse, residuals) if successful,
                   else raises UnboundLocalError or other optimization error.
                   - sse (float): Sum of Squared Errors of the fit.
                   - residuals (np.array): Array of residuals (obs - fit).
        """
        # Args observation slice is needed for residual calculation later
        cofs = minimize(
            args=observations, fun=self.func_restricted, x0=seed, method=minimizer
        )

        if cofs.success:
            tc = cofs.x[0]
            m = cofs.x[1]
            w = cofs.x[2]

            # Get linear params A, B, C1, C2
            rM = self.matrix_equation(observations, tc, m, w)
            a, b, c1, c2 = rM[:, 0].tolist()

            # Calculate derived param C
            c = self.get_c(c1, c2) # Assuming self.get_c exists

            # --- NEW: Calculate SSE and Residuals ---
            sse = cofs.fun # The final value of the objective function is the SSE
            lppls_fit_values = self.lppls(observations[0, :], tc, m, w, a, b, c1, c2)
            residuals = observations[1, :] - lppls_fit_values
            # --------------------------------------

            # Store coefficients if needed (original behavior)
            # @TODO consider if this self.coef_ update is desired during multi-fits
            # for coef in ["tc", "m", "w", "a", "b", "c", "c1", "c2"]:
            #     self.coef_[coef] = eval(coef)

            # Return parameters including sse and residuals
            return tc, m, w, a, b, c, c1, c2, sse, residuals
        else:
            # Raise error if minimization failed
            raise ValueError(f"Optimization failed: {cofs.message}")

    def plot_fit(self, show_tc=False):
        """
        Args:
            observations (Mx2 numpy array): the observed data
        Returns:
            nothing, should plot the fit
        """
        tc, m, w, a, b, c, c1, c2 = self.coef_.values()
        time_ord = [
            pd.Timestamp.fromordinal(d) for d in self.observations[0, :].astype("int32")
        ]
        t_obs = self.observations[0, :]
        # ts = pd.to_datetime(t_obs*10**9)
        # compatible_date = np.array(ts, dtype=np.datetime64)

        lppls_fit = [self.lppls(t, tc, m, w, a, b, c1, c2) for t in t_obs]
        price = self.observations[1, :]

        first = t_obs[0]
        last = t_obs[-1]

        O = (w / (2.0 * np.pi)) * np.log((tc - first) / (tc - last))
        D = (m * np.abs(b)) / (w * np.abs(c))

        fig, (ax1) = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(14, 8))
        # fig.suptitle(
        #     'Single Fit\ntc: {:.2f}, m: {:.2f}, w: {:.2f}, a: {:.2f}, b: {:.2f}, c: {:.2f}, O: {:.2f}, D: {:.2f}'.format(tc, m, w, a, b, c, O, D),
        #     fontsize=16)

        ax1.plot(time_ord, price, label="price", color="black", linewidth=0.75)
        ax1.plot(time_ord, lppls_fit, label="lppls fit", color="blue", alpha=0.5)
        # if show_tc:
        #     ax1.axvline(x=np.array(tc_ts, dtype=np.datetime64), label='tc={}'.format(ts), color='red', alpha=0.5)
        # set grids
        ax1.grid(which="major", axis="both", linestyle="--")
        # set labels
        ax1.set_ylabel("ln(p)")
        ax1.legend(loc=2)

        plt.xticks(rotation=45)
        # ax1.xaxis.set_major_formatter(months)
        # # rotates and right aligns the x labels, and moves the bottom of the
        # # axes up to make room for them
        # fig.autofmt_xdate()

    def compute_indicators(self, res, filter_conditions_config=None):
        pos_lst = []
        neg_lst = []
        pos_conf_lst = []
        neg_conf_lst = []
        price = []
        ts = []
        _fits = []

        if filter_conditions_config is None:
            # TODO make configurable again!
            m_min, m_max = (0.0, 1.0)
            w_min, w_max = (2.0, 15.0)
            O_min = 2.5
            D_min = 0.5
        else:
            # TODO parse user provided conditions
            pass

        for r in res:
            ts.append(r["t2"])
            price.append(r["p2"])
            pos_qual_count = 0
            neg_qual_count = 0
            pos_count = 0
            neg_count = 0
            # _fits.append(r['res'])

            for idx, fits in enumerate(r["res"]):
                t1 = fits["t1"]
                t2 = fits["t2"]
                tc = fits["tc"]
                m = fits["m"]
                w = fits["w"]
                b = fits["b"]
                c = fits["c"]
                O = fits["O"]
                D = fits["D"]

                # t_delta = t2 - t1
                # pct_delta_min = t_delta * 0.5
                # pct_delta_max = t_delta * 0.5
                # tc_min = t2 - pct_delta_min
                # tc_max = t2 + pct_delta_max

                # [max(t2 - 60, t2 - 0.5 * (t2 - t1)), min(252, t2 + 0.5 * (t2 - t1))]

                # print('lb: max({}, {})={}'.format(t2 - 60, t2 - 0.5 * (t2 - t1), max(t2 - 60, t2 - 0.5 * (t2 - t1))))
                # print('ub: min({}, {})={}'.format(t2 + 252, t2 + 0.5 * (t2 - t1), min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                #
                # print('{} < {} < {}'.format(max(t2 - 60, t2 - 0.5 * (t2 - t1)), tc, min(t2 + 252, t2 + 0.5 * (t2 - t1))))
                # print('______________')

                tc_in_range = (
                    max(t2 - 60, t2 - 0.5 * (t2 - t1))
                    < tc
                    < min(t2 + 252, t2 + 0.5 * (t2 - t1))
                )
                m_in_range = m_min < m < m_max
                w_in_range = w_min < w < w_max

                if b != 0 and c != 0:
                    O = O
                else:
                    O = np.inf

                O_in_range = O > O_min
                D_in_range = D > D_min  # if m > 0 and w > 0 else False

                if (
                    tc_in_range
                    and m_in_range
                    and w_in_range
                    and O_in_range
                    and D_in_range
                ):
                    is_qualified = True
                else:
                    is_qualified = False

                if b < 0:
                    pos_count += 1
                    if is_qualified:
                        pos_qual_count += 1
                if b > 0:
                    neg_count += 1
                    if is_qualified:
                        neg_qual_count += 1
                # add this to res to make life easier
                r["res"][idx]["is_qualified"] = is_qualified

            _fits.append(r["res"])

            pos_conf = pos_qual_count / pos_count if pos_count > 0 else 0
            neg_conf = neg_qual_count / neg_count if neg_count > 0 else 0
            pos_conf_lst.append(pos_conf)
            neg_conf_lst.append(neg_conf)

            # pos_lst.append(pos_count / (pos_count + neg_count))
            # neg_lst.append(neg_count / (pos_count + neg_count))

            # tc_lst.append(tc_cnt)
            # m_lst.append(m_cnt)
            # w_lst.append(w_cnt)
            # O_lst.append(O_cnt)
            # D_lst.append(D_cnt)

        res_df = pd.DataFrame(
            {
                "time": ts,
                "price": price,
                "pos_conf": pos_conf_lst,
                "neg_conf": neg_conf_lst,
                "_fits": _fits,
            }
        )
        return res_df
        # return ts, price, pos_lst, neg_lst, pos_conf_lst, neg_conf_lst, #tc_lst, m_lst, w_lst, O_lst, D_lst

    def plot_confidence_indicators(self, res):
        """
        Args:
            res (list): result from mp_compute_indicator
            condition_name (str): the name you assigned to the filter condition in your config
            title (str): super title for both subplots
        Returns:
            nothing, should plot the indicator
        """
        res_df = self.compute_indicators(res)
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(18, 10))

        ord = res_df["time"].astype("int32")
        ts = [pd.Timestamp.fromordinal(d) for d in ord]

        # plot pos bubbles
        ax1_0 = ax1.twinx()
        ax1.plot(ts, res_df["price"], color="black", linewidth=0.75)
        # ax1_0.plot(compatible_date, pos_lst, label='pos bubbles', color='gray', alpha=0.5)
        ax1_0.plot(
            ts,
            res_df["pos_conf"],
            label="bubble indicator (pos)",
            color="red",
            alpha=0.5,
        )

        # plot neg bubbles
        ax2_0 = ax2.twinx()
        ax2.plot(ts, res_df["price"], color="black", linewidth=0.75)
        # ax2_0.plot(compatible_date, neg_lst, label='neg bubbles', color='gray', alpha=0.5)
        ax2_0.plot(
            ts,
            res_df["neg_conf"],
            label="bubble indicator (neg)",
            color="green",
            alpha=0.5,
        )

        # if debug:
        #     ax3.plot(ts, tc_lst, label='tc count')
        #     ax3.plot(ts, m_lst, label='m count')
        #     ax3.plot(ts, w_lst, label='w count')
        #     ax3.plot(ts, O_lst, label='O count')
        #     ax3.plot(ts, D_lst, label='D count')

        # set grids
        ax1.grid(which="major", axis="both", linestyle="--")
        ax2.grid(which="major", axis="both", linestyle="--")

        # set labels
        ax1.set_ylabel("ln(p)")
        ax2.set_ylabel("ln(p)")

        ax1_0.set_ylabel("bubble indicator (pos)")
        ax2_0.set_ylabel("bubble indicator (neg)")

        ax1_0.legend(loc=2)
        ax2_0.legend(loc=2)

        plt.xticks(rotation=45)
        # format the ticks
        # ax1.xaxis.set_major_locator(years)
        # ax2.xaxis.set_major_locator(years)
        # ax1.xaxis.set_major_formatter(years_fmt)
        # ax2.xaxis.set_major_formatter(years_fmt)
        # ax1.xaxis.set_minor_locator(months)
        # ax2.xaxis.set_minor_locator(months)

        # rotates and right aligns the x labels, and moves the bottom of the
        # axes up to make room for them
        # fig.autofmt_xdate()

    def mp_compute_nested_fits(
        self,
        workers,
        window_size=80,
        smallest_window_size=20,
        outer_increment=5,
        inner_increment=2,
        max_searches=25,
        filter_conditions_config={},
    ):
        obs_copy = self.observations
        obs_opy_len = len(obs_copy[0]) - window_size
        func = self._func_compute_nested_fits

        # print('obs_copy', obs_copy)
        # print('obs_opy_len', obs_opy_len)

        func_arg_map = [
            (
                obs_copy[:, i : window_size + i],
                window_size,
                i,
                smallest_window_size,
                outer_increment,
                inner_increment,
                max_searches,
            )
            for i in range(0, obs_opy_len + 1, outer_increment)
        ]

        with Pool(processes=workers) as pool:
            self.indicator_result = list(
                tqdm(pool.imap(func, func_arg_map), total=len(func_arg_map))
            )

        return self.indicator_result

    def compute_nested_fits(
        self,
        window_size=80,
        smallest_window_size=20,
        outer_increment=5,
        inner_increment=2,
        max_searches=25,
    ):
        obs_copy = self.observations
        obs_copy_len = len(obs_copy[0]) - window_size
        window_delta = window_size - smallest_window_size
        res = []
        i_idx = 0
        for i in range(0, obs_copy_len + 1, outer_increment):
            j_idx = 0
            obs = obs_copy[:, i : window_size + i]
            t1 = obs[0][0]
            t2 = obs[0][-1]
            res.append([])
            i_idx += 1
            for j in range(0, window_delta, inner_increment):
                obs_shrinking_slice = obs[:, j:window_size]
                tc, m, w, a, b, c, _, _, _, _ = self.fit(
                    max_searches, obs=obs_shrinking_slice
                )
                res[i_idx - 1].append([])
                j_idx += 1
                for k in [t2, t1, a, b, c, m, 0, tc]:
                    res[i_idx - 1][j_idx - 1].append(k)
        return xr.DataArray(
            data=res,
            dims=("t2", "windowsizes", "params"),
            coords=dict(
                t2=obs_copy[0][(window_size - 1) :],
                windowsizes=range(smallest_window_size, window_size, inner_increment),
                params=["t2", "t1", "a", "b", "c", "m", "0", "tc"],
            ),
        )

    def _func_compute_nested_fits(self, args):
        """
        Internal helper function for mp_compute_nested_fits.
        Computes LPPLS fits for shrinking windows within a given observation slice.

        MODIFIED: Stores sse, residuals, and data indices in the result dict.
        """
        (
            obs,              # The current outer window slice (2xN)
            window_size,      # Original size of the outer window 'obs'
            n_iter,           # The starting index of 'obs' in the *full* observations array
            smallest_window_size,
            outer_increment,  # Not used within this function directly
            inner_increment,
            max_searches,
        ) = args

        window_delta = window_size - smallest_window_size
        res_inner = [] # Results for the nested fits of this outer window

        # Outer window metadata
        t1_outer = obs[0, 0]
        t2_outer = obs[0, -1]
        p2_outer = obs[1, -1]

        # Loop through nested shrinking windows
        # j is the number of points to shave off the beginning of the outer slice 'obs'
        for j in range(0, window_delta + 1, inner_increment): # Go up to window_delta inclusive
            # Define the shrinking slice for this nested fit
            # Start index within 'obs' is j, end index is window_size (exclusive)
            obs_shrinking_slice = obs[:, j:window_size]
            num_points_in_slice = obs_shrinking_slice.shape[1]

            # Calculate absolute start/end indices relative to the original full timeseries
            abs_start_idx = n_iter + j
            abs_end_idx = n_iter + window_size # End index (exclusive)

            # Ensure slice has enough points (e.g., >= 3 for matrix equation)
            if num_points_in_slice < 3:
                continue

            # Fit the model to the shrinking slice
            tc, m, w, a, b, c, c1, c2, O, D, sse, residuals = self.fit(
                max_searches, obs=obs_shrinking_slice
            )

            # Get the specific t1/t2 for this nested slice
            nested_t1 = obs_shrinking_slice[0, 0]
            nested_t2 = obs_shrinking_slice[0, -1] # Should match t2_outer

            # Store results for this nested fit
            fit_result_dict = {
                "tc": tc, "m": m, "w": w, "a": a, "b": b, "c": c, "c1": c1, "c2": c2,
                "t1": nested_t1, "t2": nested_t2, # t1/t2 of this specific nested window
                "O": O, "D": D,
                "sse": sse,                 # Store Sum of Squared Errors
                "residuals": residuals,     # Store residuals array
                "num_points": num_points_in_slice, # Store number of points in window
                "data_start_idx": abs_start_idx,   # Store absolute start index
                "data_end_idx": abs_end_idx,       # Store absolute end index (exclusive)
                "status": "success" if tc != 0 else "failed" # Basic status check
            }
            res_inner.append(fit_result_dict)

        # Return results for this outer window step
        return {"t1": t1_outer, "t2": t2_outer, "p2": p2_outer, "res": res_inner}

    def _get_tc_bounds(self, obs, lower_bound_pct, upper_bound_pct):
        """
        Args:
            obs (Mx2 numpy array): the observed data
            lower_bound_pct (float): percent of (t_2 - t_1) to use as the LOWER bound initial value for the optimization
            upper_bound_pct (float): percent of (t_2 - t_1) to use as the UPPER bound initial value for the optimization
        Returns:
            tc_init_min, tc_init_max
        """
        t_first = obs[0][0]
        t_last = obs[0][-1]
        t_delta = t_last - t_first
        pct_delta_min = t_delta * lower_bound_pct
        pct_delta_max = t_delta * upper_bound_pct
        tc_init_min = t_last - pct_delta_min
        tc_init_max = t_last + pct_delta_max
        return tc_init_min, tc_init_max

    def _is_O_in_range(self, tc, w, last, O_min):
        return ((w / (2 * np.pi)) * np.log(abs(tc / (tc - last)))) > O_min

    def _is_D_in_range(self, m, w, b, c, D_min):
        return False if m <= 0 or w <= 0 else abs((m * b) / (w * c)) > D_min

    def get_oscillations(self, w, tc, t1, t2):
        return (w / (2.0 * np.pi)) * np.log((tc - t1) / (tc - t2))

    def get_damping(self, m, w, b, c):
        return (m * np.abs(b)) / (w * np.abs(c))

    def get_c(self, c1, c2):
        if c1 and c2:
            # c = (c1 ** 2 + c2 ** 2) ** 0.5
            return c1 / np.cos(np.arctan(c2 / c1))
        else:
            return 0

    def ordinal_to_date(self, ordinal):
        # Since pandas represents timestamps in nanosecond resolution,
        # the time span that can be represented using a 64-bit integer
        # is limited to approximately 584 years
        try:
            return date.fromordinal(int(ordinal)).strftime("%Y-%m-%d")
        except (ValueError, OutOfBoundsDatetime):
            return str(pd.NaT)

    def detect_bubble_start_time_via_lagrange(
            self,
            max_window_size: int,
            min_window_size: int,
            step_size: int = 1,
            max_searches: int = 25,
        ) -> Optional[Dict[str, Any]]:

        window_sizes = []
        sse_list = []
        ssen_list = []
        lagrange_sse_list = []
        start_times = []
        n_params = 7 # The number of degrees of freedom used for this exercise as well as for the real-world time series is p = 8, which includes the 7 parameters of the LPPLS model augmented by the extra parameter t1

        total_obs = len(self.observations[0])

        lppls_params_list = []

        for window_size in range(max_window_size, min_window_size - 1, -step_size):
            start_idx = total_obs - window_size
            end_idx = total_obs
            obs_window = self.observations[:, start_idx:end_idx]

            start_time = self.observations[0][start_idx]
            start_times.append(start_time)
            t2 = self.observations[0][end_idx - 1]

            try:
                tc, m, w, a, b, _, c1, c2, _, _ = self.fit(max_searches, obs=obs_window)
                if tc == 0.0:
                    continue 

                # compute predictions and residuals
                Yhat = self.lppls(obs_window[0], tc, m, w, a, b, c1, c2)
                residuals = obs_window[1] - Yhat

                # compute SSE and normalized SSE
                sse = np.sum(residuals ** 2)
                n = len(obs_window[0])
                if n - n_params <= 0:
                    continue  # avoid division by zero or negative degrees of freedom
                ssen = sse / (n - n_params)

                window_sizes.append(window_size)
                sse_list.append(sse)
                ssen_list.append(ssen)
                lppls_params_list.append({
                    'tc': tc,
                    'm': m,
                    'w': w,
                    'a': a,
                    'b': b,
                    'c1': c1,
                    'c2': c2,
                    'obs_window': obs_window  # may be useful later
                })
            except Exception as e:
                print(e)
                continue

        if len(ssen_list) < 2:
            warnings.warn("Not enough data points to compute Lagrange regularization.")
            return None

        window_sizes_np = np.array(window_sizes).reshape(-1, 1)
        ssen_list_np = np.array(ssen_list)

        # fit linear regression to normalized SSE vs. window sizes
        reg = LinearRegression().fit(window_sizes_np, ssen_list_np)
        slope = reg.coef_[0]
        intercept = reg.intercept_

        # compute Lagrange-regularized SSE
        for i in range(len(sse_list)):
            lagrange_sse = ssen_list[i] - slope * window_sizes[i]
            lagrange_sse_list.append(lagrange_sse)

        # find the optimal window size
        min_index = np.argmin(lagrange_sse_list)
        optimal_window_size = window_sizes[min_index]
        optimal_params = lppls_params_list[min_index]  # get LPPLS parameters for optimal window

        # get tau (start time of the bubble)
        tau_idx = total_obs - optimal_window_size
        tau = self.observations[0][tau_idx]

        return {
            "tau": tau,
            "optimal_window_size": optimal_window_size,
            "tc": optimal_params['tc'],
            "m": optimal_params['m'],
            "w": optimal_params['w'],
            "a": optimal_params['a'],
            "b": optimal_params['b'],
            "c1": optimal_params['c1'],
            "c2": optimal_params['c2'],
            "window_sizes": window_sizes,
            "sse_list": sse_list,
            "ssen_list": ssen_list,
            "lagrange_sse_list": lagrange_sse_list,
            "start_times": start_times
        }
