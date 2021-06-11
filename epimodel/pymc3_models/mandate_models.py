from pymc3 import Model
import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano.tensor.signal.conv as C

# All cases-only for now.

class HardcodedMandateModel(Model):
    
    def __init__(self, df, nRs, delay=True, name='', model=None) :
        super().__init__(name, model)
        
        self.df = df
        self.nRs = nRs
        self.regions = list(df.index.get_level_values("city").unique())
        self.response = 'cases_cumulative'
        
        if delay: 
            # Column with constant time shift
            self.feature = 'mandate_delayed'
        else : 
            self.feature = 'mandate_effected'
            
        intercept_mean_mean=20
        intercept_mean_sd=10
        intercept_var_sd=10

        ############
        # Intercept
        # Group mean
        a_grp = pm.Normal('a_grp', intercept_mean_mean, intercept_mean_sd)
        # Group variance
        a_grp_sigma = pm.HalfNormal('a_grp_sigma', intercept_var_sd)
        # Individual intercepts
        a_ind = pm.Normal('a_ind', 
                          mu=a_grp, sigma=a_grp_sigma, 
                          shape=nRs)

        # Group mean
        b_grp = pm.Normal('b_grp', 1.33, .5)
        # Group variance
        b_grp_sigma = pm.HalfNormal('b_grp_sigma', .5)
        # Individual slopes
        b_ind = pm.Normal('b_ind', 
                          mu=b_grp, sigma=b_grp_sigma, 
                          shape=nRs)

        # Group mean
        c_grp = pm.Normal('c_grp', 0, .5)
        # Group variance
        c_grp_sigma = pm.HalfNormal('c_grp_sigma', .5)
        # Individual slopes
        c_ind = pm.Normal('c_ind', 
                          mu=c_grp, sigma=c_grp_sigma,
                          shape=nRs)

        # Error
        sigma = pm.HalfNormal('sigma', 50., shape=nRs)

        # Create likelihood for each city
        for i, city in enumerate(self.regions):
            df_city = self.df.iloc[self.df.index.get_level_values('city') == city]

            # By using pm.Data we can change these values after sampling.
            # This allows us to extend x into the future so we can get
            # forecasts by sampling from the posterior predictive
            x = pm.Data(city + "x_data", 
                        np.arange(len(df_city)))
            confirmed = pm.Data(city + "y_data", df_city[self.response].astype('float64').values)

            # Likelihood
            pm.NegativeBinomial(
                city, 
                (a_ind[i] * (b_ind[i] + c_ind[i] * df_city[self.feature])** x), # Exponential regression
                sigma[i], 
                observed=confirmed)


class ConfirmationDelayMandateModel(Model):
    
    def __init__(self, df, nRs, delay=True, name='', model=None) :
        super().__init__(name, model)
        
        self.df = df
        self.nRs = nRs
        self.regions = list(df.index.get_level_values("city").unique())
        self.response = 'cases_cumulative'

        self.cases_delay_mean_mean = 10
        self.cases_delay_mean_sd = 1
        self.cases_truncation = 32
        
        
        # Intercept
        # Group mean
        a_grp = pm.Normal('a_grp', 2, 10)
        # Group variance
        a_grp_sigma = pm.HalfNormal('a_grp_sigma', 10)
        # Individual intercepts
        a_ind = pm.Normal('a_ind', 
                          mu=a_grp, sigma=a_grp_sigma, 
                          shape=nRs)
        
        # Group mean
        b_grp = pm.Normal('b_grp', 1.13, .5)
        # Group variance
        b_grp_sigma = pm.HalfNormal('b_grp_sigma', .5)
        # Individual slopes
        b_ind = pm.Normal('b_ind', 
                          mu=b_grp, sigma=b_grp_sigma, 
                          shape=nRs)
        
        # Group mean
        c_grp = pm.Normal('c_grp', 0, .5)
        # Group variance
        c_grp_sigma = pm.HalfNormal('c_grp_sigma', .5)
        # Individual slopes
        c_ind = pm.Normal('c_ind', 
                          mu=c_grp, sigma=c_grp_sigma,
                          shape=nRs)
        # Error
        sigma = pm.HalfNormal('sigma', 50., shape=nRs)
        
        cases_delay_dist = pm.NegativeBinomial.dist(mu=self.cases_delay_mean_mean, alpha=5)
        reporting_delay = self.truncate_and_normalise(cases_delay_dist)
        
        # Create likelihood for each city
        for i, city in enumerate(self.regions):
            df_city = df.iloc[df.index.get_level_values('city') == city]

            x = pm.Data(city + "x_data", 
                        np.arange(len(df_city)))
            # Exponential regression
            infected_cases = (a_ind[i] * (b_ind[i] + c_ind[i] * df_city['mandate_effected']) ** x)
            infected_cases = T.reshape(infected_cases, (1, len(df_city)))

            # convolve with delay to produce expectations
            expected_cases = C.conv2d(
              infected_cases,
              reporting_delay,
              border_mode="full"
            )[0, :len(df_city)]

            # By using pm.Data we can change these values after sampling.
            # This allows us to extend x into the future so we can get
            # forecasts by sampling from the posterior predictive
            confirmed = pm.Data(city + "y_data", df_city[self.response].astype('float64').values)

            # Likelihood
            pm.NegativeBinomial(
                city,
                mu=expected_cases,
                alpha=sigma[i],
                shape=len(df_city),
                observed=confirmed)
            
    
    def truncate_and_normalise(self, delay) :
        bins = np.arange(0, self.cases_truncation)
        pmf = T.exp(delay.logp(bins))
        pmf = pmf / T.sum(pmf)
        
        return pmf.reshape((1, self.cases_truncation))