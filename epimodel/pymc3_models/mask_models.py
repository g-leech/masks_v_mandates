import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano.tensor.signal.conv as C

from epimodel import EpidemiologicalParameters
from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace

#from .base_model import BaseCMModel


class RandomWalkMobilityModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_mean=0,
        wearing_mean_linear=0,
        wearing_mean_quadratic=0,
        wearing_sigma=0.4,
        wearing_sigma_linear=0.26,
        wearing_sigma_quadratic=0.13,
        mobility_mean=1.704,
        mobility_sigma=0.44,
        R_prior_mean_mean=1.07,
        R_prior_mean_scale=0.2,
        R_noise_scale=0.4,  
        cm_prior="skewed",
        gi_mean_mean=5.06,
        gi_mean_sd=0.33,
        gi_sd_mean=2.11,
        gi_sd_sd=0.5,
        cases_delay_mean_mean=10.92,
        cases_delay_disp_mean=5.41,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        mob_and_wearing_only=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=wearing_mean, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=wearing_mean_linear, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=wearing_mean_quadratic, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=wearing_mean_quadratic, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=mobility_mean, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs,))
            self.RegionR = pm.Deterministic(
                "RegionR", self.HyperRMean + self.RegionR_noise * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            if mob_and_wearing_only:
                growth_reduction = 0
            else:
                pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # if wearing_parameterisation is not None:
                # initial_wearing_reduction = growth_reduction_wearing[:, 0]
                # initial_wearing_reduction = T.reshape(initial_wearing_reduction, (self.nRs, 1))
                # pm.Deterministic("initial_wearing_reduction", initial_wearing_reduction)

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0
                # initial_wearing_reduction = 0
            else:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            if mobility_leaveout:
                growth_reduction_mobility = 0
                initial_mobility_reduction = 0
            else:
                initial_mobility_reduction = growth_reduction_mobility[:, 0]
                initial_mobility_reduction = T.reshape(initial_mobility_reduction, (self.nRs, 1))
                pm.Deterministic("initial_mobility_reduction", initial_mobility_reduction)

                pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                # - (growth_reduction_wearing - initial_wearing_reduction)
                # - growth_reduction_mobility
                - (growth_reduction_mobility - initial_mobility_reduction)
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class EverythingNormalisedModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.4,
        wearing_sigma_linear=0.26,
        wearing_sigma_quadratic=0.13,
        mobility_mean=1.704,
        mobility_sigma=0.44,
        R_prior_mean=1.07,
        R_noise_scale=0.32,
        cm_prior="skewed",
        gi_mean_mean=5.06,
        gi_mean_sd=0.33,
        gi_sd_mean=2.11,
        gi_sd_sd=0.5,
        cases_delay_mean_mean=10.92,
        cases_delay_disp_mean=5.41,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        mob_and_wearing_only=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=mobility_mean, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            if mob_and_wearing_only:
                growth_reduction = 0
            else:
                pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            if wearing_parameterisation is not None:
                initial_wearing_reduction = growth_reduction_wearing[:, 0]
                initial_wearing_reduction = T.reshape(initial_wearing_reduction, (self.nRs, 1))
                pm.Deterministic("initial_wearing_reduction", initial_wearing_reduction)

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0
                initial_wearing_reduction = 0
            else:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            if mobility_leaveout:
                growth_reduction_mobility = 0
                initial_mobility_reduction = 0
            else:
                initial_mobility_reduction = growth_reduction_mobility[:, 0]
                initial_mobility_reduction = T.reshape(initial_mobility_reduction, (self.nRs, 1))
                pm.Deterministic("initial_mobility_reduction", initial_mobility_reduction)

                pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                # - growth_reduction_wearing
                - (growth_reduction_wearing - initial_wearing_reduction)
                # - growth_reduction_mobility
                - (growth_reduction_mobility - initial_mobility_reduction)
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class RandomWalkMobilityModelWearingZeroed(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.4,
        wearing_sigma_linear=0.26,
        wearing_sigma_quadratic=0.13,
        mobility_mean=1.0,
        mobility_sigma=0.5,
        R_prior_mean=1.2,
        R_noise_scale=0.5,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        mob_and_wearing_only=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=mobility_mean, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            if mob_and_wearing_only:
                growth_reduction = 0
            else:
                pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            if wearing_parameterisation is not None:
                initial_wearing_reduction = growth_reduction_wearing[:, 0]
                initial_wearing_reduction = T.reshape(initial_wearing_reduction, (self.nRs, 1))
                pm.Deterministic("initial_wearing_reduction", initial_wearing_reduction)

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0
                initial_wearing_reduction = 0
            else:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            if mobility_leaveout:
                growth_reduction_mobility = 0
                # initial_mobility_reduction = 0
            else:
                # initial_mobility_reduction = growth_reduction_mobility[:, 0]
                # initial_mobility_reduction = T.reshape(initial_mobility_reduction, (self.nRs, 1))
                # pm.Deterministic("initial_mobility_reduction", initial_mobility_reduction)

                pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - (growth_reduction_wearing - initial_wearing_reduction)
                # - (growth_reduction_mobility - initial_mobility_reduction)
                - growth_reduction_mobility
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

class RandomWalkMobilityModelOld(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        mobility_sigma=0.5,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=0, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            if wearing_parameterisation:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )

            pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                - growth_reduction_mobility
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

class MandateMobilityModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        mobility_mean=1.704,
        mobility_sigma=0.44,
        R_prior_mean_mean=1.07,
        R_prior_mean_scale=0.2,
        R_noise_scale=0.4,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5.06,
        gi_mean_sd=0.33,
        gi_sd_mean=2.11,
        gi_sd_sd=0.5,
        mask_sigma=0.08,
        n_mandates=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10.92,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5.41,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # mob_feature = "avg_mobility_no_parks_no_residential"
        # assert self.d.CMs[-2] == mob_feature
        # assert self.d.CMs[-1] == "H6_Facial Coverings"

        with self.model:
            # build NPI Effectiveness priors:
            if intervention_prior == "AL":
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 3,),
                )
            else:
                self.CM_Alpha = pm.Normal(
                    "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 3,)
                )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            self.Mandate_Alpha_1 = pm.Normal(
                "Mandate_Alpha_1", mu=0, sigma=mask_sigma, shape=(1,)
            )
            self.Mandate_Alpha_2 = pm.Normal(
                "Mandate_Alpha_2", mu=0, sigma=mask_sigma, shape=(1,)
            )
            if n_mandates == 1:
                self.MandateReduction = pm.Deterministic(
                    "MandateReduction", T.exp((-1.0) * self.Mandate_Alpha_1)
                )
            else:
                self.MandateReduction = pm.Deterministic(
                    "MandateReduction", T.exp((-1.0) * (self.Mandate_Alpha_1 + self.Mandate_Alpha_2))
                )

            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=mobility_mean, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs,))
            self.RegionR = pm.Deterministic(
                "RegionR", self.HyperRMean + self.RegionR_noise * self.HyperRVar
            )

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-3, :])
            self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs - 3, 1)) * self.ActiveCMs
            )

            self.ActiveCMs_mandate_1 = pm.Data(
                "ActiveCMs_mandate_1", self.d.ActiveCMs[:, -3, :]
            )
            self.ActiveCMs_mandate_2 = pm.Data(
                "ActiveCMs_mandate_2", self.d.ActiveCMs[:, -1, :]
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            self.ActiveCMReduction_mandate_1 = T.reshape(
                self.Mandate_Alpha_1, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mandate_1,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )
            self.ActiveCMReduction_mandate_2 = T.reshape(
                self.Mandate_Alpha_2, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mandate_2,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mandate_1 = T.sum(self.ActiveCMReduction_mandate_1, axis=1)
            growth_reduction_mandate_2 = T.sum(self.ActiveCMReduction_mandate_2, axis=1)

            if n_mandates == 1:
                growth_reduction_mandate = growth_reduction_mandate_1
            else:
                growth_reduction_mandate = growth_reduction_mandate_1 + growth_reduction_mandate_2

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            if mobility_leaveout:
                growth_reduction_mobility = 0
                initial_mobility_reduction = 0
            else:
                initial_mobility_reduction = growth_reduction_mobility[:, 0]
                initial_mobility_reduction = T.reshape(initial_mobility_reduction, (self.nRs, 1))
                pm.Deterministic("initial_mobility_reduction", initial_mobility_reduction)

                pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_mandate
                - (growth_reduction_mobility - initial_mobility_reduction)
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_mandate
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


MandateModel = MandateMobilityModel


class RandomWalkModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=6,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            # pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            #             self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthNoiseBase = pm.Normal(
            #                 "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            #             )
            #             self.GrowthCasesNoise = pm.Deterministic(
            #                 "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            #             )

            #             self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 self.GrowthNoiseScale,
            #                 shape=(self.nRs, self.nDs)
            #             )

            #             self.Growth = pm.Deterministic(
            #                 "Growth",
            #                 T.inc_subtensor(self.ExpectedGrowth[:, :], self.GrowthCasesNoise),
            #             )
            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class RandomWalkModelFixedCaseDelay(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            # pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            #             self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthNoiseBase = pm.Normal(
            #                 "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            #             )
            #             self.GrowthCasesNoise = pm.Deterministic(
            #                 "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            #             )

            #             self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 self.GrowthNoiseScale,
            #                 shape=(self.nRs, self.nDs)
            #             )

            #             self.Growth = pm.Deterministic(
            #                 "Growth",
            #                 T.inc_subtensor(self.ExpectedGrowth[:, :], self.GrowthCasesNoise),
            #             )
            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayDisp = pm.Normal(
            #    "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class MobilityModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        wearing_parameterisation="exp",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        mobility_sigma=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        assert self.d.CMs[-2] == "avg_mobility_no_parks_no_residential"

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            # if we are not using wearing--just mandates, then:
            if wearing_parameterisation is None:
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 1,),
                )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 2,),
                )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=0, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )

            pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                - growth_reduction_mobility,
            )
            # self.Rt = pm.Deterministic("Rt", T.exp(self.ExpectedLogR))

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            self.GrowthNoiseBase = pm.Normal(
                "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            )

            #             self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 self.GrowthNoiseScale,
            #                 shape=(self.nRs, self.nDs)
            #             )

            self.Growth = pm.Deterministic(
                "Growth",
                T.inc_subtensor(self.ExpectedGrowth[:, :], self.GrowthCasesNoise),
            )

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class Base_Model(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        wearing_parameterisation="log_quadratic_2",
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        mobility_sigma=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior_scale=10,  # TODO: 20
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=5,
        log_init_sd=5,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        assert self.d.CMs[-3] == "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == "residential_percent_change_from_baseline"

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 2,),
                )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 3,),
                )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = AsymmetricLaplace(
                    "Wearing_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=0, sigma=mobility_sigma, shape=(2,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                1.0 - self.Mobility_Alpha[0] - self.Mobility_Alpha[1],
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-3, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 3, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -3:-1, :]
            )

            self.ActiveCMReduction_mobility = (
                T.reshape(self.Mobility_Alpha[0], (1, 1, 1))
                * T.reshape(
                    self.ActiveCMs_mobility[:, 0, :],
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                + T.reshape(self.Mobility_Alpha[1], (1, 1, 1))
                * T.reshape(
                    self.ActiveCMs_mobility[:, 1, :],
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                ** 2
            )
            eps = 10 ** (-20)
            growth_reduction_mobility = -1.0 * T.log(
                T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_mobility, axis=1)) + eps
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                - growth_reduction_mobility,
            )

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (
                np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR)
            )

            self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            self.GrowthNoiseBase = pm.Normal(
                "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            )

            #             self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 self.GrowthNoiseScale,
            #                 shape=(self.nRs, self.nDs)
            #             )

            self.Growth = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class ALCasesOnlyModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        wearing_parameterisation="log_quadratic_2",
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(self.nCMs,)
                )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                self.CM_Alpha = AsymmetricLaplace(
                    "CM_Alpha",
                    scale=cm_prior_scale,
                    symmetry=0.5,
                    shape=(self.nCMs - 1,),
                )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = AsymmetricLaplace(
                    "Wearing_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing,
            )

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                )
            )

            self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            self.GrowthNoiseBase = pm.Normal(
                "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            )
            self.Growth = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class CasesOnlyModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="log_quadratic_2",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )  # , in wrong place
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            # pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing,
            )
            # self.Rt = pm.Deterministic("Rt", T.exp(self.ExpectedLogR))

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            self.GrowthNoiseBase = pm.Normal(
                "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            )

            #             self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 self.GrowthNoiseScale,
            #                 shape=(self.nRs, self.nDs)
            #             )

            self.Growth = pm.Deterministic(
                "Growth",
                T.inc_subtensor(self.ExpectedGrowth[:, :], self.GrowthCasesNoise),
            )

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


NormCasesOnlyModel = CasesOnlyModel


class NoMasksModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        wearing_parameterisation="log_quadratic_2",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            self.CM_Alpha = AsymmetricLaplace(
                "CM_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(self.nCMs,)
            )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal(
                "RegionR_noise", 0, R_noise_scale, shape=(self.nRs),
            )
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionR_noise  # * self.HyperRVar
            )

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            growth_reduction_wearing = 0

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing,
            )

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.HyperGNScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            self.GrowthNoiseBase = pm.Normal(
                "GrowthNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise", 0 + self.GrowthNoiseBase * self.HyperGNScale
            )

            self.Growth = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )


class CasesDeathsModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.
        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []
        observed_deaths = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if self.d.NewCases.mask[r, d] == False and not np.isnan(
                    self.d.Confirmed.data[r, d]
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

                if self.d.NewDeaths.mask[r, d] == False and not np.isnan(
                    self.d.Deaths.data[r, d]
                ):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)
        self.all_observed_deaths = np.array(observed_deaths)

    @property
    def nRs(self):
        """
        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """
        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """
        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        R_prior_mean=1.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        deaths_delay_mean_mean=21,
        deaths_delay_mean_sd=1,
        deaths_delay_disp_mean=9,
        deaths_delay_disp_sd=1,
        cases_truncation=32,
        deaths_truncation=48,
        log_init_mean=9.9,  # cases
        log_init_sd=9.9,  # cases
        log_init_deaths_mean=5.29,
        log_init_deaths_sd=3.9,
        **kwargs,
    ):
        """
        Build PyMC3 model.
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        print("R_prior_mean:", R_prior_mean)

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            self.CM_Alpha = AsymmetricLaplace(
                "CM_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(self.nCMs,)
            )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar
            )

            # load CMs active, compute log-R reduction and region log-R based on NPIs active
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction,
            )

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (
                np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR)
            )
            # self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 growth_noise_scale,
            #                 shape=(self.nRs, self.nDs)
            #             )
            self.HyperGNScaleCases = pm.HalfNormal(
                "GrowthCasesNoiseScale", growth_noise_scale
            )
            self.GrowthCasesNoiseBase = pm.Normal(
                "GrowthCasesNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthCasesNoise = pm.Deterministic(
                "GrowthCasesNoise",
                0 + self.GrowthCasesNoiseBase * self.HyperGNScaleCases,
            )

            self.GrowthCases = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )

            # Originally N(0, 50)
            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal(
                "InitialSizeCases_log",
                log_init_mean,
                log_init_sd,
                shape=(self.nRs, 1),  # , 1
            )
            self.InfectedCases = pm.Deterministic(
                "InfectedCases",
                pm.math.exp(
                    self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)
                ),
            )

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.InfectedCases, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.PsiCases = pm.HalfNormal("PsiCases", 5.0)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

            """
                Deaths
            """
            # self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthDeathsNoise = pm.Normal(
            #                 "GrowthDeathsNoise",
            #                 0,
            #                 growth_noise_scale,
            #                 shape=(self.nRs, self.nDs)
            #             )
            self.HyperGNScaleDeaths = pm.HalfNormal(
                "GrowthDeathsNoiseScale", growth_noise_scale
            )
            self.GrowthDeathsNoiseBase = pm.Normal(
                "GrowthDeathsNoiseBase", 0, 1, shape=(self.nRs, self.nDs)
            )
            self.GrowthDeathsNoise = pm.Deterministic(
                "GrowthDeathsNoise",
                0 + self.GrowthDeathsNoiseBase * self.HyperGNScaleDeaths,
            )

            self.GrowthDeaths = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthDeathsNoise
            )

            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal(
                "InitialSizeDeaths_log",
                log_init_deaths_mean,
                log_init_deaths_sd,
                shape=(self.nRs, 1),
            )
            self.InfectedDeaths = pm.Deterministic(
                "InfectedDeaths",
                pm.math.exp(
                    self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)
                ),
            )

            self.DeathsDelayMean = pm.Normal(
                "DeathsDelayMean", deaths_delay_mean_mean, deaths_delay_mean_sd
            )

            self.DeathsDelayDisp = pm.Normal(
                "DeathsDelayDisp", deaths_delay_disp_mean, deaths_delay_disp_sd
            )
            deaths_delay_dist = pm.NegativeBinomial.dist(
                mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp
            )
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            # convolve with delay to production reports
            expected_deaths = C.conv2d(
                self.InfectedDeaths, fatality_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedDeaths = pm.Deterministic(
                "ExpectedDeaths", expected_deaths.reshape((self.nRs, self.nDs))
            )

            self.PsiDeaths = pm.HalfNormal("PsiDeaths", 5.0)

            # effectively handle missing values ourselves
            # death output distribution
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
            )


class CasesDeathsFixedScaleModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.
        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []
        observed_deaths = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if self.d.NewCases.mask[r, d] == False and not np.isnan(
                    self.d.Confirmed.data[r, d]
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

                if self.d.NewDeaths.mask[r, d] == False and not np.isnan(
                    self.d.Deaths.data[r, d]
                ):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)
        self.all_observed_deaths = np.array(observed_deaths)

    @property
    def nRs(self):
        """
        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """
        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """
        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        R_prior_mean=1.2,
        cm_prior_scale=10,
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        deaths_delay_mean_mean=21,
        deaths_delay_mean_sd=1,
        deaths_delay_disp_mean=9,
        deaths_delay_disp_sd=1,
        cases_truncation=32,
        deaths_truncation=48,
        log_init_mean=9.9,  # cases
        log_init_sd=9.9,  # cases
        log_init_deaths_mean=5.29,
        log_init_deaths_sd=3.9,
        **kwargs,
    ):
        """
        Build PyMC3 model.
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        print("R_prior_mean:", R_prior_mean)

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            self.CM_Alpha = AsymmetricLaplace(
                "CM_Alpha", scale=cm_prior_scale, symmetry=0.5, shape=(self.nCMs,)
            )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=0.5)

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar
            )

            # load CMs active, compute log-R reduction and region log-R based on NPIs active
            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            self.ActiveCMReduction = (
                T.reshape(self.CM_Alpha, (1, self.nCMs, 1)) * self.ActiveCMs
            )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)) - growth_reduction,
            )

            # self.GI_mean = gi_mean_mean
            # self.GI_sd = gi_sd_mean

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (
                np.exp(self.ExpectedLogR / gi_alpha) - T.ones_like(self.ExpectedLogR)
            )
            # self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", growth_noise_scale)
            #             self.GrowthCasesNoise = pm.Normal(
            #                 "GrowthCasesNoise",
            #                 0,
            #                 growth_noise_scale,
            #                 shape=(self.nRs, self.nDs)
            #             )
            self.GrowthCasesNoise = pm.Normal(
                "GrowthCasesNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs)
            )

            self.GrowthCases = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )

            # Originally N(0, 50)
            # Confirmed Cases
            # seed and produce daily infections which become confirmed cases
            self.InitialSizeCases_log = pm.Normal(
                "InitialSizeCases_log",
                log_init_mean,
                log_init_sd,
                shape=(self.nRs, 1),  # , 1
            )
            self.InfectedCases = pm.Deterministic(
                "InfectedCases",
                pm.math.exp(
                    self.InitialSizeCases_log + self.GrowthCases.cumsum(axis=1)
                ),
            )

            self.CasesDelayMean = pm.Normal(
                "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            )
            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=self.CasesDelayMean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.InfectedCases, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.PsiCases = pm.HalfNormal("PsiCases", 5.0)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

            """
                Deaths
            """

            self.GrowthDeathsNoise = pm.Normal(
                "GrowthDeathsNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs),
            )

            self.GrowthDeaths = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthDeathsNoise
            )

            # seed and produce daily infections which become confirmed cases
            self.InitialSizeDeaths_log = pm.Normal(
                "InitialSizeDeaths_log",
                log_init_deaths_mean,
                log_init_deaths_sd,
                shape=(self.nRs, 1),
            )
            self.InfectedDeaths = pm.Deterministic(
                "InfectedDeaths",
                pm.math.exp(
                    self.InitialSizeDeaths_log + self.GrowthDeaths.cumsum(axis=1)
                ),
            )

            self.DeathsDelayMean = pm.Normal(
                "DeathsDelayMean", deaths_delay_mean_mean, deaths_delay_mean_sd
            )

            self.DeathsDelayDisp = pm.Normal(
                "DeathsDelayDisp", deaths_delay_disp_mean, deaths_delay_disp_sd
            )
            deaths_delay_dist = pm.NegativeBinomial.dist(
                mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp
            )
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            # convolve with delay to production reports
            expected_deaths = C.conv2d(
                self.InfectedDeaths, fatality_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedDeaths = pm.Deterministic(
                "ExpectedDeaths", expected_deaths.reshape((self.nRs, self.nDs))
            )

            self.PsiDeaths = pm.HalfNormal("PsiDeaths", 5.0)

            # effectively handle missing values ourselves
            # death output distribution
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
            )


class AdditiveModel(pm.Model):
    def __init__(self, data, npis, cm_plot_style=None, name="", model=None):
        """
        Constructor.
        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        self.npi_names = npis

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []
        observed_deaths = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if self.d.NewCases.mask[r, d] == False and not np.isnan(
                    self.d.Confirmed.data[r, d]
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True

                if self.d.NewDeaths.mask[r, d] == False and not np.isnan(
                    self.d.Deaths.data[r, d]
                ):
                    observed_deaths.append(r * self.nDs + d)
                else:
                    self.d.NewDeaths.mask[r, d] = True

        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)
        self.all_observed_deaths = np.array(observed_deaths)

        # TODO: Remove masks from ActiveCMs
        if "percent_mc" in self.npi_names:
            i = self.npi_names.index("percent_mc")

    @property
    def nRs(self):
        """
        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """
        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """
        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        R_prior_mean=3.28,
        cm_prior_scale=10,
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        deaths_delay_mean_mean=21,
        deaths_delay_mean_sd=1,
        deaths_delay_disp_mean=9,
        deaths_delay_disp_sd=1,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=1,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        init_log_cases=9.9,
        deaths_truncation=48,
        cases_truncation=32,
        **kwargs,
    ):
        """
        Build NPI effectiveness model
        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale. For this model, this is the concentration parameter
                                dirichlet distribution, same for all NPIs.
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param deaths_delay_mean_mean: mean of normal prior placed over death delay mean
        :param deaths_delay_mean_sd: sd of normal prior placed over death delay mean
        :param deaths_delay_disp_mean: mean of normal prior placed over death delay dispersion (alpha / psi)
        :param deaths_delay_disp_sd: sd of normal prior placed over death delay dispersion (alpha / psi)
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        with self.model:
            self.AllBeta = pm.Dirichlet(
                "AllBeta",
                cm_prior_scale * np.ones((self.nCMs + 1)),
                shape=(self.nCMs + 1,),
            )
            self.CM_Beta = pm.Deterministic("CM_Beta", self.AllBeta[1:])
            self.Beta_hat = pm.Deterministic("Beta_hat", self.AllBeta[0])
            self.CMReduction = pm.Deterministic("CMReduction", self.CM_Beta)

            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=0.5)

            self.RegionR_noise = pm.Normal("RegionLogR_noise", 0, 1, shape=(self.nRs),)
            self.RegionR = pm.Deterministic(
                "RegionR", R_prior_mean + self.RegionLogR_noise * self.HyperRVar
            )

            self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs)

            active_cm_reduction = T.reshape(self.CM_Beta, (1, self.nCMs, 1)) * (
                T.ones_like(self.ActiveCMs) - self.ActiveCMs
            )

            growth_reduction = T.sum(active_cm_reduction, axis=1) + self.Beta_hat

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.log(
                    T.exp(T.reshape(pm.math.log(self.RegionR), (self.nRs, 1)))
                    * growth_reduction
                ),
            )

            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = gi_beta * (
                pm.math.exp(self.ExpectedLogR / gi_alpha)
                - T.ones_like(self.ExpectedLogR)
            )
            # Consider prior over
            # self.GrowthNoiseScale = pm.HalfNormal("GrowthNoiseScale", 0, growth_noise_scale)
            self.GrowthCasesNoise = pm.Normal(
                "GrowthCasesNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs),
            )
            self.GrowthDeathsNoise = pm.Normal(
                "GrowthDeathsNoise", 0, growth_noise_scale, shape=(self.nRs, self.nDs),
            )

            self.GrowthCases = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthCasesNoise
            )
            self.GrowthDeaths = T.inc_subtensor(
                self.ExpectedGrowth[:, :], self.GrowthDeathsNoise
            )

            self.PsiCases = pm.HalfNormal("PsiCases", 5.0)
            self.PsiDeaths = pm.HalfNormal("PsiDeaths", 5.0)

            self.InitialSizeCases_log = pm.Normal(
                "InitialSizeCases_log",
                init_log_cases,
                init_log_cases,
                shape=(self.nRs,),
            )
            self.InfectedCases_log = pm.Deterministic(
                "InfectedCases_log",
                T.reshape(self.InitialSizeCases_log, (self.nRs, 1))
                + self.GrowthCases.cumsum(axis=1),
            )

            self.InfectedCases = pm.Deterministic(
                "InfectedCases", pm.math.exp(self.InfectedCases_log)
            )

            self.CasesDelayDisp = pm.Normal(
                "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=self.CasesDelayDisp
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_cases = C.conv2d(
                self.InfectedCases, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_cases.reshape((self.nRs, self.nDs))
            )

            # learn the output noise for this.
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.PsiCases,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

            self.InitialSizeDeaths_log = pm.Normal(
                "InitialSizeDeaths_log",
                init_log_cases,
                init_log_cases,
                shape=(self.nRs,),
            )
            self.InfectedDeaths_log = pm.Deterministic(
                "InfectedDeaths_log",
                T.reshape(self.InitialSizeDeaths_log, (self.nRs, 1))
                + self.GrowthDeaths.cumsum(axis=1),
            )

            self.InfectedDeaths = pm.Deterministic(
                "InfectedDeaths", pm.math.exp(self.InfectedDeaths_log)
            )

            self.DeathsDelayMean = pm.Normal(
                "DeathsDelayMean", deaths_delay_mean_mean, deaths_delay_mean_sd
            )
            self.DeathsDelayDisp = pm.Normal(
                "DeathsDelayDisp", deaths_delay_disp_mean, deaths_delay_disp_sd
            )
            deaths_delay_dist = pm.NegativeBinomial.dist(
                mu=self.DeathsDelayMean, alpha=self.DeathsDelayDisp
            )
            bins = np.arange(0, deaths_truncation)
            pmf = T.exp(deaths_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            fatality_delay = pmf.reshape((1, deaths_truncation))

            expected_deaths = C.conv2d(
                self.InfectedDeaths, fatality_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedDeaths = pm.Deterministic(
                "ExpectedDeaths", expected_deaths.reshape((self.nRs, self.nDs))
            )

            # effectively handle missing values ourselves
            self.ObservedDeaths = pm.NegativeBinomial(
                "ObservedDeaths",
                mu=self.ExpectedDeaths.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
                alpha=self.PsiDeaths,
                shape=(len(self.all_observed_deaths),),
                observed=self.d.NewDeaths.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_deaths
                ],
            )


class RandomWalkMobilityHyperRModelOld(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.2,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        mobility_sigma=0.5,
        R_prior_mean_mean=1.64,
        R_prior_mean_scale=0.2,
        R_noise_scale=0.3,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=0, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )
            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs,))
            self.RegionR = pm.Deterministic(
                "RegionR", self.HyperRMean + self.RegionR_noise * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)
            pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0

            if wearing_parameterisation:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )
            pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                - growth_reduction_wearing
                - growth_reduction_mobility
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )

class RandomWalkMobilityHyperRModel(pm.Model):
    def __init__(self, data, cm_plot_style=None, name="", model=None):
        """
        Constructor.

        :param data: PreprocessedData object
        :param cm_plot_style: NPI data
        :param name: model name
        :param model: required for PyMC3, but never used.
        """
        super().__init__(name, model)
        self.d = data
        self.trace = None
        # self.CMDelayCut = 30

        # compute days to actually observe, looking at the data which is masked, and which isn't.
        # indices of active country-days in the 1D Rs*Ds vector
        observed_active = []

        for r in range(self.nRs):
            for d in range(self.nDs):
                # if its not masked, after the cut, and not before 100 confirmed
                if (
                    self.d.NewCases.mask[r, d] == False
                    # and d > self.CMDelayCut
                    and not np.isnan(self.d.Confirmed.data[r, d])
                ):
                    observed_active.append(r * self.nDs + d)
                else:
                    self.d.NewCases.mask[r, d] = True
        print(len(observed_active))
        self.all_observed_active = np.array(observed_active)

    @property
    def nRs(self):
        """

        :return: number of regions / countries
        """
        return len(self.d.Rs)

    @property
    def nDs(self):
        """

        :return: number of days
        """
        return len(self.d.Ds)

    @property
    def nCMs(self):
        """

        :return: number of countermeasures
        """
        return len(self.d.CMs)

    def build_model(
        self,
        r_walk_period=7,
        r_walk_noise_scale_prior=0.15,
        intervention_prior="AL",
        cm_prior_scale=10,
        wearing_parameterisation="exp",
        wearing_sigma=0.4,
        wearing_sigma_linear=0.15,
        wearing_sigma_quadratic=0.07,
        mobility_sigma=0.5,
        mobility_mean=1.0,
        R_prior_mean_mean=1.7,
        R_prior_mean_scale=0.3,
        R_noise_scale=0.5,  # 0.5
        cm_prior="skewed",
        gi_mean_mean=5,
        gi_mean_sd=1,
        gi_sd_mean=2,
        gi_sd_sd=2,
        growth_noise_scale=0.2,
        cases_delay_mean_mean=10,
        cases_delay_mean_sd=2,
        cases_delay_disp_mean=5,
        cases_delay_disp_sd=1,
        cases_truncation=32,
        log_init_mean=9.9,
        log_init_sd=9.9,
        IGNORE_START=10,
        IGNORE_END=10,
        mobility_leaveout=False,
        mob_and_wearing_only=False,
        **kwargs,
    ):
        """
        Build PyMC3 model.

        :param R_prior_mean: R_0 prior mean
        :param cm_prior_scale: NPI effectiveness prior scale
        :param cm_prior: NPI effectiveness prior type. Either 'normal', 'icl' or skewed (asymmetric laplace)
        :param gi_mean_mean: mean of normal prior placed over the generation interval mean
        :param gi_mean_sd: sd of normal prior placed over the generation interval mean
        :param gi_sd_mean: mean of normal prior placed over the generation interval sd
        :param gi_sd_sd: sd of normal prior placed over the generation interval sd
        :param growth_noise_scale: growth noise scale
        :param cases_delay_mean_mean: mean of normal prior placed over cases delay mean
        :param cases_delay_mean_sd: sd of normal prior placed over cases delay mean
        :param cases_delay_disp_mean: mean of normal prior placed over cases delay dispersion
        :param cases_delay_disp_sd: sd of normal prior placed over cases delay dispersion
        :param deaths_truncation: maximum death delay
        :param cases_truncation: maximum reporting delay
        """
        for key, _ in kwargs.items():
            print(f"Argument: {key} not being used")

        # Ensure mobility feature is in the right place
        mob_feature = "avg_mobility_no_parks_no_residential"
        assert self.d.CMs[-2] == mob_feature

        with self.model:
            # build NPI Effectiveness priors
            # TODO: Normal, narrower
            print(wearing_parameterisation)
            if wearing_parameterisation is None:
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 1,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 1,)
                    )
            else:
                assert self.d.CMs[-1] == "percent_mc"
                if intervention_prior == "AL":
                    self.CM_Alpha = AsymmetricLaplace(
                        "CM_Alpha",
                        scale=cm_prior_scale,
                        symmetry=0.5,
                        shape=(self.nCMs - 2,),
                    )
                else:
                    self.CM_Alpha = pm.Normal(
                        "CM_Alpha", mu=0, sigma=cm_prior_scale, shape=(self.nCMs - 2,)
                    )

            self.CMReduction = pm.Deterministic(
                "CMReduction", T.exp((-1.0) * self.CM_Alpha)
            )

            # prior specification for wearing options:
            if wearing_parameterisation == "exp":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", T.exp((-1.0) * self.Wearing_Alpha)
                )
            if wearing_parameterisation == "log_linear":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_linear, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(1,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction", 1.0 - 2.0 * self.Wearing_Alpha
                )
            if wearing_parameterisation == "log_quadratic_2":
                self.Wearing_Alpha = pm.Normal(
                    "Wearing_Alpha", mu=0, sigma=wearing_sigma_quadratic, shape=(2,)
                )
                self.WearingReduction = pm.Deterministic(
                    "WearingReduction",
                    1.0 - self.Wearing_Alpha[0] - self.Wearing_Alpha[1],
                )
            self.Mobility_Alpha = pm.Normal(
                "Mobility_Alpha", mu=mobility_mean, sigma=mobility_sigma, shape=(1,)
            )
            self.MobilityReduction = pm.Deterministic(
                "MobilityReduction",
                (2.0 * (T.exp(-1.0 * self.Mobility_Alpha)))
                / (1.0 + T.exp(-1.0 * self.Mobility_Alpha)),
            )

            self.HyperRMean = pm.TruncatedNormal(
                "HyperRMean", mu=R_prior_mean_mean, sigma=R_prior_mean_scale, lower=0.1
            )
            self.HyperRVar = pm.HalfNormal("HyperRVar", sigma=R_noise_scale)

            self.RegionR_noise = pm.Normal("RegionR_noise", 0, 1, shape=(self.nRs,))
            self.RegionR = pm.Deterministic(
                "RegionR", self.HyperRMean + self.RegionR_noise * self.HyperRVar
            )

            # load CMs active without wearing, compute log-R reduction and region log-R based on NPIs active
            if wearing_parameterisation is not None:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-2, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 2, 1)) * self.ActiveCMs
                )

                self.ActiveCMs_wearing = pm.Data(
                    "ActiveCMs_wearing", self.d.ActiveCMs[:, -1, :]
                )
            else:
                self.ActiveCMs = pm.Data("ActiveCMs", self.d.ActiveCMs[:, :-1, :])

                self.ActiveCMReduction = (
                    T.reshape(self.CM_Alpha, (1, self.nCMs - 1, 1)) * self.ActiveCMs
                )

            growth_reduction = T.sum(self.ActiveCMReduction, axis=1)

            if mob_and_wearing_only:
                growth_reduction = 0
            else:
                pm.Deterministic("growth_reduction", growth_reduction)

            # calculating reductions for each of the wearing parameterisations
            if wearing_parameterisation == "exp":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                growth_reduction_wearing = T.sum(self.ActiveCMReduction_wearing, axis=1)

            if wearing_parameterisation == "log_linear":
                self.ActiveCMReduction_wearing = T.reshape(
                    self.Wearing_Alpha, (1, 1, 1)
                ) * T.reshape(
                    self.ActiveCMs_wearing,
                    (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )

            if wearing_parameterisation == "log_quadratic":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha, (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # TODO: take out these reshapes. Can just add an axis manually.
            if wearing_parameterisation == "log_quadratic_2":
                self.ActiveCMReduction_wearing = (
                    T.reshape(self.Wearing_Alpha[0], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    + T.reshape(self.Wearing_Alpha[1], (1, 1, 1))
                    * T.reshape(
                        self.ActiveCMs_wearing,
                        (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
                    )
                    ** 2
                )
                eps = 10 ** (-20)
                growth_reduction_wearing = -1.0 * T.log(
                    T.nnet.relu(1.0 - T.sum(self.ActiveCMReduction_wearing, axis=1))
                    + eps
                )
            # if wearing_parameterisation is not None:
            #     initial_wearing_reduction = growth_reduction_wearing[:, 0]
            #     initial_wearing_reduction = T.reshape(initial_wearing_reduction, (self.nRs, 1))
            #     pm.Deterministic("initial_wearing_reduction", initial_wearing_reduction)

            if wearing_parameterisation is None:
                growth_reduction_wearing = 0
                # initial_wearing_reduction = 0
            else:
                pm.Deterministic("growth_reduction_wearing", growth_reduction_wearing)

            # make reduction for mobility
            self.ActiveCMs_mobility = pm.Data(
                "ActiveCMs_mobility", self.d.ActiveCMs[:, -2, :]
            )

            self.ActiveCMReduction_mobility = T.reshape(
                self.Mobility_Alpha, (1, 1, 1)
            ) * T.reshape(
                self.ActiveCMs_mobility,
                (self.d.ActiveCMs.shape[0], 1, self.d.ActiveCMs.shape[2]),
            )

            growth_reduction_mobility = -1.0 * T.log(
                T.sum(
                    (2.0 * T.exp(-1.0 * self.ActiveCMReduction_mobility))
                    / (1.0 + T.exp(-1.0 * self.ActiveCMReduction_mobility)),
                    axis=1,
                )
            )

            if mobility_leaveout:
                growth_reduction_mobility = 0
            else:
                pm.Deterministic("growth_reduction_mobility", growth_reduction_mobility)

            # random walk
            nNP = int(self.nDs / r_walk_period) - 1

            r_walk_noise_scale = pm.HalfNormal(
                "r_walk_noise_scale", r_walk_noise_scale_prior
            )
            # rescaling variables by 10 for better NUTS adaptation
            r_walk_noise = pm.Normal("r_walk_noise", 0, 1.0 / 10, shape=(self.nRs, nNP))

            expanded_r_walk_noise = T.repeat(
                r_walk_noise_scale * 10.0 * T.cumsum(r_walk_noise, axis=-1),
                r_walk_period,
                axis=-1,
            )[: self.nRs, : (self.nDs - 2 * r_walk_period)]

            full_log_Rt_noise = T.zeros((self.nRs, self.nDs))
            full_log_Rt_noise = T.subtensor.set_subtensor(
                full_log_Rt_noise[:, 2 * r_walk_period :], expanded_r_walk_noise
            )

            self.ExpectedLogR = pm.Deterministic(
                "ExpectedLogR",
                T.reshape(pm.math.log(self.RegionR), (self.nRs, 1))
                - growth_reduction
                # - (growth_reduction_wearing - initial_wearing_reduction)
                - growth_reduction_wearing
                - growth_reduction_mobility
                + full_log_Rt_noise,
            )

            self.Rt_walk = pm.Deterministic(
                "Rt_walk",
                T.exp(T.log(self.RegionR.reshape((self.nRs, 1))) + full_log_Rt_noise),
            )

            self.Rt_cm = pm.Deterministic(
                "Rt_cm",
                T.exp(
                    T.log(self.RegionR.reshape((self.nRs, 1)))
                    - growth_reduction
                    - growth_reduction_wearing
                ),
            )

            # convert R into growth rates
            self.GI_mean = pm.Normal("GI_mean", gi_mean_mean, gi_mean_sd)
            self.GI_sd = pm.Normal("GI_sd", gi_sd_mean, gi_sd_sd)

            gi_beta = self.GI_mean / self.GI_sd ** 2
            gi_alpha = self.GI_mean ** 2 / self.GI_sd ** 2

            self.ExpectedGrowth = pm.Deterministic(
                "ExpectedGrowth",
                gi_beta
                * (
                    np.exp(self.ExpectedLogR / gi_alpha)
                    - T.ones_like(self.ExpectedLogR)
                ),
            )

            self.Growth = self.ExpectedGrowth

            # Originally N(0, 50)
            self.InitialSize_log = pm.Normal(
                "InitialSize_log", log_init_mean, log_init_sd, shape=(self.nRs,)
            )
            self.Infected_log = pm.Deterministic(
                "Infected_log",
                T.reshape(self.InitialSize_log, (self.nRs, 1))
                + self.Growth.cumsum(axis=1),
            )

            self.Infected = pm.Deterministic("Infected", pm.math.exp(self.Infected_log))

            # self.CasesDelayMean = pm.Normal(
            #     "CasesDelayMean", cases_delay_mean_mean, cases_delay_mean_sd
            # )
            # self.CasesDelayDisp = pm.Normal(
            #     "CasesDelayDisp", cases_delay_disp_mean, cases_delay_disp_sd
            # )
            cases_delay_dist = pm.NegativeBinomial.dist(
                mu=cases_delay_mean_mean, alpha=cases_delay_disp_mean
            )
            bins = np.arange(0, cases_truncation)
            pmf = T.exp(cases_delay_dist.logp(bins))
            pmf = pmf / T.sum(pmf)
            reporting_delay = pmf.reshape((1, cases_truncation))

            expected_confirmed = C.conv2d(
                self.Infected, reporting_delay, border_mode="full"
            )[:, : self.nDs]

            self.ExpectedCases = pm.Deterministic(
                "ExpectedCases", expected_confirmed.reshape((self.nRs, self.nDs))
            )

            # Observation Noise Dispersion Parameter (negbin alpha)
            self.Psi = pm.HalfNormal("Psi", 5)

            # effectively handle missing values ourselves
            # likelihood
            self.ObservedCases = pm.NegativeBinomial(
                "ObservedCases",
                mu=self.ExpectedCases.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
                alpha=self.Psi,
                shape=(len(self.all_observed_active),),
                observed=self.d.NewCases.data.reshape((self.nRs * self.nDs,))[
                    self.all_observed_active
                ],
            )
