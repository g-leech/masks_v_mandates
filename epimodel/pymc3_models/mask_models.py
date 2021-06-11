import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano.tensor.signal.conv as C

from epimodel import EpidemiologicalParameters
from epimodel.pymc3_distributions.asymmetric_laplace import AsymmetricLaplace


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
            if wearing_parameterisation is None:
                growth_reduction_wearing = 0
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


