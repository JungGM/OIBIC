"""Power forecast with historical-weather forecast-bias teacher features."""
import argparse
from pathlib import Path
import solar_normalized as engine
import forecast_corrector as teacher
BASE_INPUTS=engine.inputs
BASE_DESIGN=engine.design


def configure(data):
    teacher.train(data)
    models=teacher.load()
    extras=[n for source,target in models for n in
            [f'corrected_{source}_{target}',f'corrected_q_{source}_{target}']]
    def inputs(path):
        meta,names,cats=BASE_INPUTS(path)
        return meta,names+extras,cats
    def design(d,sid,meta,categories):
        return teacher.augment(BASE_DESIGN(d,sid,meta,categories),models)
    engine.inputs=inputs;engine.design=design
    engine.RUN=engine.ex.HERE/'_local/weather_augmented_v14'
    engine.CANDIDATE='improved_v14'
    engine.NORMALIZE=True
    engine.RUN.mkdir(parents=True,exist_ok=True)
    engine.p.save_json(engine.RUN/'weather_provenance.json',dict(
        cutoff=str(teacher.CUTOFF),teacher_models={f'{s}_{t}':m['model_sha256'] for (s,t),(_,m) in models.items()},
        inference_inputs='provided previous-day forecasts and calendar only; no observed test weather'))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--stage',choices=['evaluate','produce'],default='evaluate');a=ap.parse_args()
    configure(a.data);getattr(engine,a.stage)(a.data)
