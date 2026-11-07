"""Issued-day features inspired by teammate source snapshot f33a5296.

Independent implementation; no observed weather, generation lags or external data.
Call on a complete site/day grid before filtering labels or training cutoffs.
"""
import numpy as np
import pandas as pd
import features as ft


def add_features(frame):
    d = frame.sort_values(['site_id', 'ts'], kind='stable').reset_index(drop=True)
    ts = pd.DatetimeIndex(d.ts)
    if len(d) % 24 or not len(d):
        raise ValueError('Complete operating days required')
    days = (ts-pd.Timedelta(hours=1)).normalize().asi8.reshape(-1, 24)
    ids = d.site_id.to_numpy().reshape(-1, 24)
    if not ((days == days[:, :1]).all() and (ids == ids[:, :1]).all()
            and (ts.hour.to_numpy().reshape(-1, 24) == np.r_[np.arange(1, 24), 0]).all()):
        raise ValueError('Nonconsecutive site/day grid')
    new = {}
    samples = [np.maximum(ft.solar(ts+pd.Timedelta(minutes=m), d.lat, d.lon)[2], 0)
               for m in (-25, -15, -5, 5, 15, 25)]
    q = np.mean(samples, axis=0)
    new['solar_hour_integral'] = q
    new['daylight_fraction'] = (np.asarray(samples) > 0).mean(axis=0)
    for col in ('r17_wa_cloud', 'r17_wb_cloud', 'r17_wb_uv_idx'):
        a = d[col].to_numpy().reshape(-1, 24)
        for lag in (1, 2):
            delta = np.full(a.shape, np.nan)
            delta[:, lag:] = a[:, lag:]-a[:, :-lag]
            new[f'{col}_delta{lag}'] = delta.ravel()
        std, span = np.full(a.shape, np.nan), np.full(a.shape, np.nan)
        for h in range(24):
            part = a[:, max(0, h-2):h+1]
            ok = np.isfinite(part).all(axis=1)
            std[ok, h], span[ok, h] = part[ok].std(axis=1), np.ptp(part[ok], axis=1)
        new[f'{col}_std3'], new[f'{col}_range3'] = std.ravel(), span.ravel()
        weights = q.reshape(-1, 24)*np.isfinite(a)
        mean = np.divide((np.nan_to_num(a)*weights).sum(axis=1), weights.sum(axis=1),
                         out=np.full(len(a), np.nan), where=weights.sum(axis=1) > 0)
        new[f'{col}_solar_mean'] = np.repeat(mean, 24)
        new[f'{col}_solar_deviation'] = (a-mean[:, None]).ravel()
    for provider in ('wa', 'wb'):
        temp = d[f'r17_{provider}_temp'].to_numpy()
        wind = d[f'r17_{provider}_wind_speed'].to_numpy()
        snow = d[f'r17_{provider}_snow'].to_numpy().reshape(-1, 24)
        rolling = np.full(snow.shape, np.nan)
        thaw = np.full(snow.shape, np.nan)
        t = temp.reshape(-1, 24)
        for h in range(24):
            s = snow[:, max(0, h-5):h+1]
            tt = t[:, max(0, h-5):h+1]
            rolling[:, h] = s.sum(axis=1)
            thaw[:, h] = np.maximum(tt, 0).sum(axis=1)
        new[f'{provider}_snow6'] = rolling.ravel()
        new[f'{provider}_thaw6'] = thaw.ravel()
        new[f'{provider}_cold_snow6'] = rolling.ravel()*(temp <= 0)
        new[f'{provider}_heat_wind'] = q*np.maximum(temp-25, 0)/(1+np.maximum(wind, 0))
    new['dewpoint_depression'] = d.r17_wb_temp.to_numpy()-d.r17_wb_dew_point.to_numpy()
    new['gust_excess'] = d.r17_wb_wind_gust_spd.to_numpy()-d.r17_wb_wind_speed.to_numpy()
    return pd.concat([d, pd.DataFrame(new, dtype='float32')], axis=1), list(new)
