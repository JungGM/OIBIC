"""Build local monthly feature shards without reading generation or SMP."""
import argparse,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from features import weather,build_site,NAMES
import pandas as pd

def main():
    p=argparse.ArgumentParser();p.add_argument("--data",type=Path,required=True);p.add_argument("--out",type=Path,required=True);p.add_argument("--start",required=True);p.add_argument("--end",required=True);p.add_argument("--site-limit",type=int);a=p.parse_args()
    start=pd.Timestamp(a.start,tz="Asia/Seoul");end=pd.Timestamp(a.end,tz="Asia/Seoul")
    if start>end: raise ValueError("Reversed period")
    if start!=start.floor("h") or end!=end.floor("h"): raise ValueError("Hourly endpoints required")
    sites=pd.read_csv(a.data/"sites.csv")
    if a.site_limit: sites=sites.head(a.site_limit)
    if not sites.site_id.is_unique: raise ValueError("Duplicate site")
    first=(start-pd.Timedelta(hours=1)).normalize();last=(end-pd.Timedelta(hours=1)).normalize()
    days=pd.date_range(first,last,freq="D")
    a.out.mkdir(parents=True,exist_ok=True)
    report={"features":len(NAMES),"sites":len(sites),"rows":0,"shards":[],"labels_read":False,"requested_start":str(start),"requested_end":str(end)}
    for month in sorted(set(days.strftime("%Y-%m"))):
        ds=days[days.strftime("%Y-%m")==month]
        ix=pd.date_range(ds[0]+pd.Timedelta(hours=1),ds[-1]+pd.Timedelta(days=1),freq="h")
        wa=weather(a.data/"train/forecast_wa.parquet","a",ix,sites.wa_station.unique());wb=weather(a.data/"train/forecast_wb.parquet","b",ix,sites.wb_station.unique())
        for site in sites.itertuples(index=False):
            d=build_site(site,ix,wa[site.wa_station],wb[site.wb_station]);d=d[d.ts.between(start,end)]
            if d.duplicated(["site_id","ts"]).any() or d.shape[1]!=162: raise ValueError("Invalid output keys or schema")
            f=a.out/f"{month}_{site.site_id}.parquet";d.to_parquet(f,index=False,compression="zstd")
            report["rows"]+=len(d);report["shards"].append({"file":f.name,"rows":len(d),"sha256":hashlib.sha256(f.read_bytes()).hexdigest()})
        print(json.dumps({"month":month,"sites":len(sites),"rows_so_far":report["rows"]}),flush=True)
    expected=len(pd.date_range(start,end,freq="h"))*len(sites)
    if report["rows"]!=expected: raise ValueError("Incomplete grid")
    report["status"]="passed";(a.out/"manifest.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps({k:v for k,v in report.items() if k!="shards"}))
if __name__=="__main__": main()
