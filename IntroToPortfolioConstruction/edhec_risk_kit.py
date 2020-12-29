import pandas as pd
from scipy.stats import norm
import numpy as np
from scipy.optimize import minimize

def drawdown(return_series: pd.Series):
    """
    Takes a time series of asset returns.
    Computes and returns a DF that contains:
    wealth index
    previous peaks
    percent drawdowns
    """
    
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    
    return pd.DataFrame({
        'Wealth': wealth_index,
        'Peaks': previous_peaks,
        'Drawdown': drawdowns
    })

def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                      header = 0, index_col=0, parse_dates = True, na_values=-99.99)

    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    
    return rets

def get_ind_returns():
    ind = pd.read_csv('data/ind30_m_vw_rets.csv', header = 0, index_col = 0, parse_dates = True)/100
    ind.index = pd.to_datetime(ind.index, format='%Y%m').to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind



def get_hfi_returns():
    hfi = pd.read_csv('data/edhec-hedgefundindices.csv', 
                      header = 0, index_col=0, parse_dates = True, na_values=-99.99)
    hfi = hfi / 100
    hfi.index = hfi.index.to_period('M')
    
    return hfi

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    computes skewness of a series or dataframe
    returns float or series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0) #using population SD
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    computes kurtosis of a series or dataframe
    returns float or series
    """
    demeaned_r = r - r.mean()
    sigma_r = r.std(ddof = 0) #using population SD
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level = .01):
    """
    Applies JB test to determine if series is normal or not
    applies test at 1% level by default
    Returns True if hypothsesis of normality is accepted, False otherwise
    """
    
    statistics, p_value = scipy.stats.jarque_bera(r)
    
    return p_value > level

def semideviation(r):
    """
    returns the semi-deviation aka negative semidiviation of r
    r must be a series or dataFrame
    """
    
    return r[r<0].std(ddof=0)

def var_historic(r, level=5):
    """
    Returns historic Value at Risk at a specified level
    i.e. returns the number such that "level" percent of the returns
    fall below that number, and the (100-level) percent are above
    i.e. There is a level% chance the the returns in any given time period will
    be X or worse
    """
    if isinstance(r, pd.DataFrame):
        return r.agg(var_historic, level = level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError('expected r to be pandas Series or DataFrame')
        

def var_gaussian(r, level=5, modified = False):
    """
    returns the parametric Gaussian VaR of a series or DF
    """
    z = norm.ppf(level/100)
    
    if modified:
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                 (z**2 - 1)*s/6 + 
                 (z**3 - 3*z)*(k-3)/24 - 
                 (2*z**3 - 5*z) * (s**2)/36
                        )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level = 5):
    """
    Computes Conditional VaR of series or DF
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.agg(cvar_historic, level = level)
    else:
        raise TypeError('expected r to be series or DF')
        
def annualize_rets(r, periods_per_year):
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1
        
def annualize_vol(r, periods_per_year):
    """
    Annualizes teh vol of a set of returns
    """
    return r.std()*(periods_per_year**.5)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    rf_per_period = (1+riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret / ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Vol
    """
    return (weights.T @ covmat @ weights)**(.5)

def plot_ef2(n_points, er, cov, style = '.-'):
    if er.shape[0] !=2 or er.shape[0] !=2:
        raise ValueError('plot_ef2 can only plot 2 asset frontiers')
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    return ef.plot.line('Volatility', 'Returns', style = style)


def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n) #equal weighting
    
    #Define Constraints
    bounds = ((0.0, 1.0), ) * n #makes n copies of the tuple
    return_is_target = { 
        'type': 'eq', #it's an equality constraint, should be 0 when succeeds
        'args': (er,), 
        'fun': lambda w, er: target_return - portfolio_return(w, er)
    }
    weights_sum_to_1= {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                      args = (cov,), method = 'SLSQP',
                      options = {'disp': False},
                      constraints = (return_is_target, weights_sum_to_1),
                      bounds = bounds
                      )
    return results.x


def optimal_weights(n_points, er, cov):
    """
    _> list of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    
    return weights

def plot_ef(n_points, er, cov, style = '.-'):
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({'Returns': rets, 'Volatility': vols})
    return ef.plot.line('Volatility', 'Returns', style = style)