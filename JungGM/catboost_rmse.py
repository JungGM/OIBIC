"""163-feature site-aware CatBoost RMSE expert, GPU training and CPU inference."""
import argparse
from pathlib import Path
import solar_normalized as engine


def configure(normalized=False):
    engine.RUN=engine.ex.HERE/'_local'/('catboost_solar_v17' if normalized else 'catboost_rmse_v13')
    engine.NORMALIZE=normalized
    engine.KIND='catboost'
    engine.CANDIDATE='improved_v17' if normalized else 'improved_v13'
    engine.PARAMS=dict(iterations=1400,learning_rate=.045,depth=7,loss_function='RMSE',
        l2_leaf_reg=15,random_seed=20261103,thread_count=2,task_type='GPU',devices='0',
        gpu_ram_part=.65,border_count=64,max_ctr_complexity=1,one_hot_max_size=2,
        allow_writing_files=False,verbose=100)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate')
    ap.add_argument('--normalized',action='store_true');a=ap.parse_args()
    configure(a.normalized)
    getattr(engine,a.stage)(a.data)
