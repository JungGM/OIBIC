"""Correct NumPy scalar promotion during inference; preserve trained source bytes.

Validation predicts float32 capacity factors, then multiplies by float64
capacities. The original export used a weak Python float scalar and therefore
kept float32 through aggregation. This adapter restores float64 multiplication
without changing any trees, features, capacities, groups or calibration.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import build_submission as original

NativeRegressor=original.xgb.XGBRegressor
OUT=Path('outputs/OIBC_177_제출용_20261103_v2').resolve()

class Float64Inference:
    def __init__(self, **kwargs): self.native=NativeRegressor(**kwargs)
    def load_model(self, path): return self.native.load_model(path)
    def predict(self, features): return self.native.predict(features).astype(np.float64)

def main(stage):
    original.OUT=OUT
    if stage=='predict':
        original.xgb=SimpleNamespace(XGBRegressor=Float64Inference)
        original.predict()
        original.save(original.PROD/'float64_export_audit.json',{
            'trained_model_unchanged':True,'model_sha256':original.sha(original.PROD/'model.ubj'),
            'original_training_builder_sha256':original.sha(Path(original.__file__)),
            'export_adapter_sha256':original.sha(Path(__file__)),
            'change':'Cast capacity-factor output to float64 before multiplying by site capacity; matches historical validation dtype.',
            'groups':'original','calibration':[1.,1.],'tree_or_feature_changes':False})
    else:
        original.validate()
        import shutil
        shutil.copyfile(original.PROD/'float64_export_audit.json',OUT/'내보내기_정밀도_검증.json')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['predict','validate']);a=p.parse_args();main(a.stage)
