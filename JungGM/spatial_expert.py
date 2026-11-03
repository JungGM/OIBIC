"""Solar-normalized site expert with neighboring issued-forecast features."""
import argparse
from pathlib import Path
import pandas as pd
import solar_normalized as engine
import spatial_forecasts as spatial

BASE_INPUTS=engine.inputs
BASE_DESIGN=engine.design


def configure(data,production=False):
    spatial.prepare(data,'train')
    if production:spatial.prepare(data,'test')
    def inputs(path):
        meta,names,cats=BASE_INPUTS(path)
        return meta,names+spatial.NAMES,cats
    def design(d,sid,meta,categories):
        out=BASE_DESIGN(d,sid,meta,categories)
        split='test' if d.ts.min()>=engine.p.stamp('2025-06-01 01:00') else 'train'
        context=pd.read_parquet(spatial.RUN/'cache'/split/f'{sid}.parquet').set_index('ts')
        if not context.index.is_unique or not out.ts.isin(context.index).all():raise ValueError('Spatial feature timestamps missing or duplicated')
        context=context.reindex(out.ts)
        for c in spatial.NAMES:out[c]=context[c].to_numpy()
        return out
    engine.inputs=inputs;engine.design=design
    engine.RUN=spatial.RUN;engine.CANDIDATE='improved_v18';engine.NORMALIZE=True


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    configure(a.data,a.stage=='produce');getattr(engine,a.stage)(a.data)
