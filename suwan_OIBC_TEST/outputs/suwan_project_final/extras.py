"""Additional statistics of already issued weather forecasts, isolated by site/day."""
import numpy as np
import pandas as pd

RAW = ['r17_wa_cloud', 'r17_wb_cloud', 'r17_wb_uv_idx']
EXTRA = [f'{c}_{s}' for c in RAW for s in
         ['delta1', 'delta2', 'std3', 'range3', 'daylight_mean', 'daylight_deviation']]

def add_features(frame):
    # Must be called on complete operating days, before filtering training labels.
    d = frame.sort_values(['site_id','ts'], kind='stable').reset_index(drop=True).copy()
    ts = pd.DatetimeIndex(d.ts)
    assert len(d) % 24 == 0 and (ts.hour[::24] == 1).all()
    assert (ts.hour[23::24] == 0).all()
    dates = (ts - pd.Timedelta(hours=1)).normalize().asi8.reshape(-1,24)
    assert (dates == dates[:, :1]).all()
    assert (d.site_id.to_numpy().reshape(-1,24) == d.site_id.to_numpy().reshape(-1,24)[:, :1]).all()
    weight = np.maximum(d.cosz.to_numpy().reshape(-1,24), 0)
    new = {}
    for name in RAW:
        a = d[name].to_numpy(dtype=float).reshape(-1,24)
        for lag in [1,2]:
            delta = np.full(a.shape, np.nan)
            delta[:, lag:] = a[:,lag:] - a[:,:-lag]
            new[f'{name}_delta{lag}'] = delta.ravel()
        # Trailing within the issued target day; a missing hour is not compressed.
        std, span = np.full(a.shape,np.nan), np.full(a.shape,np.nan)
        for h in range(24):
            part = a[:,max(0,h-2):h+1]
            ok = np.isfinite(part).all(axis=1)
            std[ok,h] = part[ok].std(axis=1)
            span[ok,h] = np.ptp(part[ok],axis=1)
        finite = np.isfinite(a)
        denom = (weight*finite).sum(axis=1)
        mean = np.divide((np.where(finite,a,0)*weight).sum(axis=1),denom,
                         out=np.full(len(a),np.nan),where=denom>0)
        new[f'{name}_std3'],new[f'{name}_range3'] = std.ravel(),span.ravel()
        new[f'{name}_daylight_mean'] = np.repeat(mean,24)
        new[f'{name}_daylight_deviation'] = (a-mean[:,None]).ravel()
    return pd.concat([d,pd.DataFrame(new,dtype='float32')],axis=1)
