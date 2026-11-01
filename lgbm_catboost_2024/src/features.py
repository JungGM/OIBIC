"""Local deterministic feature engine. No labels or fitted statistics."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pvlib

ROOT=Path(__file__).resolve().parents[1]
NAMES=[v["name"] for v in json.loads((ROOT/"configs/features_160.json").read_text())["features"]]
A="cloud temp humidity ground_press wind_speed wind_dir rain snow".split()
B="temp feel_temp wet_bulb_temp dew_point wind_speed wind_dir wind_gust_spd rel_hum vis ceiling uv_idx precip_prob rain_prob snow_prob ice_prob total_liq rain snow ice cloud".split()

def weather(path,source,index,stations):
    fields=A if source=="a" else B
    key="wa_station" if source=="a" else "wb_station"
    prefix="wa_" if source=="a" else "wb_"
    raw=pd.read_parquet(path)
    raw["ts"]=pd.to_datetime(raw.ts).dt.tz_convert("Asia/Seoul")
    raw=raw[raw.ts.isin(index)]
    if raw.duplicated([key,"round","ts"]).any(): raise ValueError("Duplicate forecast key")
    result={}
    for station in stations:
        out=pd.DataFrame(index=index)
        for rd in [10,17]:
            z=raw[(raw[key]==station)&(raw["round"]==rd)].set_index("ts")
            out[f"{source}{rd}_available"]=index.isin(z.index).astype(float)
            for v in fields:
                a=z[prefix+v].reindex(index).astype(float)
                if v in ["rain","snow","ice","total_liq","wind_speed","wind_gust_spd"]: a=a.where(a>=0)
                if v in ["cloud","humidity","rel_hum"]: a=a.clip(0,100)
                if source=="b" and v in ["wind_speed","wind_gust_spd"]: a=a/3.6
                if source=="b" and v=="snow": a=a*10
                if v=="wind_dir":
                    a=a.where(a.between(0,360));out[f"{source}{rd}_wind_sin"]=np.sin(np.deg2rad(a));out[f"{source}{rd}_wind_cos"]=np.cos(np.deg2rad(a))
                else: out[f"{source}{rd}_{v}"]=a
        result[station]=out
    return result

def build_site(site,index,wa,wb):
    d=pd.concat([wa,wb],axis=1).copy()
    target=(index-pd.Timedelta(hours=1)).normalize()
    for s in ["a","b"]:
        for v in "cloud temp humidity wind_speed rain snow".split():
            suffix="rel_hum" if s=="b" and v=="humidity" else v
            d[f"{s}_revision_{v}"]=d[f"{s}17_{suffix}"]-d[f"{s}10_{suffix}"]
    for v in "cloud temp humidity wind_speed".split():
        av=[d[f"{s}{rd}_{'rel_hum' if s=='b' and v=='humidity' else v}"] for s in ["a","b"] for rd in [10,17]]
        d[f"ab17_diff_{v}"]=av[1]-av[3]
        d[f"ab17_absdiff_{v}"]=(av[1]-av[3]).abs()
        d[f"ab_round_std_{v}"]=pd.concat(av,axis=1).std(axis=1,ddof=0).where(pd.concat(av,axis=1).count(axis=1)>=2)
    mid=index-pd.Timedelta(minutes=30)
    sol=pvlib.solarposition.get_solarposition(mid,site.lat,site.lon)
    alt=sol.elevation.to_numpy();az=np.deg2rad(sol.azimuth.to_numpy())
    samples=[]
    for minute in [5,15,25,35,45,55]:
        x=pvlib.solarposition.get_solarposition(index-pd.Timedelta(minutes=minute),site.lat,site.lon)
        samples.append(np.maximum(np.sin(np.deg2rad(x.elevation.to_numpy())),0))
    q=np.mean(samples,axis=0)
    d["solar_q"]=q;d["solar_mid_altitude"]=alt;d["solar_azimuth_sin"]=np.sin(az);d["solar_azimuth_cos"]=np.cos(az)
    d["daylight_fraction"]=np.mean(np.array(samples)>0,axis=0)
    h=mid.hour.to_numpy()+mid.minute.to_numpy()/60
    d["hour_sin"]=np.sin(2*np.pi*h/24);d["hour_cos"]=np.cos(2*np.pi*h/24)
    year=2*np.pi*(mid.dayofyear.to_numpy()-1)/np.where(mid.is_leap_year,366,365)
    d["year_sin"]=np.sin(year);d["year_cos"]=np.cos(year)
    days=pd.DatetimeIndex(target.unique())
    events=pvlib.solarposition.sun_rise_set_transit_spa(days,site.lat,site.lon)
    lengths=(events.sunset-events.sunrise).dt.total_seconds()/3600
    d["day_length"]=pd.Series(target,index=index).map(lengths)
    d["toa_horizontal"]=pvlib.irradiance.get_extra_radiation(mid).to_numpy()*q
    d["q_squared"]=q*q;d["q_cubed"]=q*q*q
    d["relative_airmass"]=pvlib.atmosphere.get_relative_airmass(90-alt)
    for s in ["a","b"]:
        for v in ["cloud","temp","humidity"]:
            val=d[f"{s}17_{'rel_hum' if s=='b' and v=='humidity' else v}"]
            d[f"{s}17_q_{v}"]=q*(1-val/100) if v=="cloud" else q*val
        wind=d[f"{s}17_wind_speed"];temp=d[f"{s}17_temp"]
        d[f"{s}17_q_wind"]=q*wind
        d[f"{s}17_heat_wind_proxy"]=q*(temp-25).clip(lower=0)/(1+wind)
    d["b17_dewpoint_depression"]=d.b17_temp-d.b17_dew_point
    d["b17_gust_excess"]=d.b17_wind_gust_spd-d.b17_wind_speed
    d["capacity_kw"]=site.capacity_kw;d["log_capacity"]=np.log(site.capacity_kw)
    for v in ["wa_station","wb_station","wa_dist","wb_dist"]: d[v]=getattr(site,v)
    # Vectorized day grouping preserves the original day-isolated formulas.
    daykey=pd.Series(target,index=index)
    def grp(v): return v.groupby(daykey,sort=False)
    position=daykey.groupby(daykey,sort=False).cumcount()
    length=daykey.groupby(daykey,sort=False).transform("size")
    def rolling(v,n,center=False,minimum=None):
        if minimum==1:
            # Boundary context has a fixed denominator, with outside-day slots absent.
            return v.fillna(0)+grp(v).shift(1).fillna(0)+grp(v).shift(-1).fillna(0)
        value=v.rolling(n,center=center,min_periods=n).sum()
        valid=(position>=n//2)&(position<length-n//2) if center else position>=n-1
        return value.where(valid)
    count=grp(pd.Series(1.,index=index)).transform("sum")
    additions={}
    for v in ["cloud","temp","uv_idx","wind_speed"]:
        a=d[f"b17_{v}"];g=grp(a)
        additions[f"b17_{v}_lagdiff"]=g.diff()
        additions[f"b17_{v}_leaddiff"]=g.shift(-1)-a
        additions[f"b17_{v}_mean3"]=(rolling(a,3,True)/3)
        additions[f"b17_{v}_std3"]=a.rolling(3,center=True,min_periods=3).std(ddof=0).where((position>=1)&(position<length-1))
        additions[f"b17_{v}_range3"]=(a.rolling(3,center=True,min_periods=3).max()-a.rolling(3,center=True,min_periods=3).min()).where((position>=1)&(position<length-1))
    for source in ["a","b"]:
        c=d[f"{source}17_cloud"];w=d.solar_q
        total=grp(w).transform("sum")
        good=(count==24)&(grp(c.notna().astype(int)).transform("sum")==24)&(total>0)
        mu=(grp(c*w).transform("sum")/total).where(good)
        additions[f"{source}17_day_cloud_qmean"]=mu
        additions[f"{source}17_day_cloud_qstd"]=(grp((c-mu)**2*w).transform("sum")/total).pow(.5).where(good)
        additions[f"{source}17_day_cloud_maxstep"]=grp(grp(c).diff().abs()).transform("max").where((count==24)&(grp(c.notna().astype(int)).transform("sum")==24))
        rain=d[f"{source}17_rain"];snow=d[f"{source}17_snow"];t=d[f"{source}17_temp"]
        additions[f"{source}17_day_rain_total"]=grp(rain).transform("sum").where((count==24)&(grp(rain.notna().astype(int)).transform("sum")==24))
        vals={"snow6":snow,"cold_snow6":snow.where(t.notna())*(t<=0),"thaw_degree_hours6":t.clip(lower=0),"wet_freezing_hours6":((t<=0)&((rain>0)|(snow>0))).astype(float).where(t.notna()&rain.notna()&snow.notna())}
        for op,val in vals.items(): additions[f"{source}17_{op}"]=rolling(val,6)
        fields=A if source=="a" else B
        cols=[f"{source}17_{'wind_sin' if v=='wind_dir' else v}" for v in fields]
        valid=d[cols].notna().sum(axis=1)
        additions[f"{source}17_day_coverage"]=grp(valid).transform("sum")/(24*len(cols))
        additions[f"{source}17_context3_coverage"]=rolling(valid,3,True,1)/(3*len(cols))
    d=pd.concat([d,pd.DataFrame(additions,index=index)],axis=1)
    missing=set(NAMES)-set(d.columns)
    if missing: raise ValueError(f"Missing feature definitions: {missing}")
    result=d[NAMES].copy()
    for c in result:
        if c not in ["wa_station","wb_station"]: result[c]=result[c].astype("float32")
    if np.isinf(result.select_dtypes(include="number").to_numpy()).any(): raise ValueError("Infinite feature")
    result=result.copy()
    result.insert(0,"ts",index);result.insert(0,"site_id",site.site_id)
    return result
