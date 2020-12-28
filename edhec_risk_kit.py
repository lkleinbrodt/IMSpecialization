#lets create a drawdown function
import pandas as pd
import scipy.stats

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
test2 = 'yo'
def get_ffme_returns():
    me_m = pd.read_csv('data/Portfolios_Formed_on_ME_monthly_EW.csv', 
                      header = 0, index_col=0, parse_dates = True, na_values=-99.99)

    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets / 100
    rets.index = pd.to_datetime(rets.index, format = '%Y%m').to_period('M')
    
    return rets

test_object = 'present'

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