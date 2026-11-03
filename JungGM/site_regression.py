"""Same site-aware 163-feature regressor without target solar normalization."""
import argparse
from pathlib import Path
import solar_normalized as engine


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    engine.RUN=engine.ex.HERE/'_local/site_regression_v12'
    engine.NORMALIZE=False
    engine.CANDIDATE='improved_v12'
    getattr(engine,a.stage)(a.data)
