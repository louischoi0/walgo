import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES

if __name__ == "__main__":
    #df = pd.read_csv("../data/gs/yd/KSSCADA.701-367-PWI-2003.F_CV.csv")
    df = pd.read_csv("../data/gs/yd/KSSCADA.701-367-PWI-4005.F_CV.csv")
    result = []
    values = []

    def t(x): 
        v = x[1]
        ts = x[0]
        ts = pd.Timestamp(ts)
        values.append(v)
        result.append([ts.month, ts.day, ts.hour, ts.minute])

    for d in df.values[:26240]: t(d)
    #for d in df.values[:2624760]: t(d)

    nd = np.array(result)
    vd = np.array(values)
    h = vd
    #v = vd.reshape(10, 262476)
    #h = v.mean(axis=0)

    # additive model for fixed seasonal variation
    #fit6 = HWES(h[:-1000], seasonal_periods=144*60, trend='add', seasonal='add').fit(optimized=True, use_brute=True)

    #pred = fit6.forecast(1000)
    y = h[-1000:]

    #plt.plot(pred, label="p")
    plt.plot(y, label="y")

    plt.show()
    


