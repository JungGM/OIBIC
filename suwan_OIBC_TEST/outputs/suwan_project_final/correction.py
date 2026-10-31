"""Serializable regularized group median regression fitted on time-external errors."""
import numpy as np
from sklearn.linear_model import QuantileRegressor

WEATHER = ['solar','cloud_a','cloud_b','uv','humidity','cloud_sd','uv_sd',
           'provider_disagreement','revision_disagreement','hour_sin','hour_cos',
           'cloud_solar','uv_solar','weather_missing']
FEATURES = ['base_util'] + WEATHER

def design_fit(frame,names):
    x=frame[names].to_numpy(dtype=float)
    med=np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0 for c in x.T])
    x=np.where(np.isfinite(x),x,med)
    mean=x.mean(axis=0); scale=x.std(axis=0);scale[scale<1e-8]=1
    return (x-mean)/scale, dict(median=med.tolist(),mean=mean.tolist(),scale=scale.tolist())

def design(frame,model):
    x=frame[model['features']].to_numpy(dtype=float)
    x=np.where(np.isfinite(x),x,np.array(model['median']))
    return (x-np.array(model['mean']))/np.array(model['scale'])

def fit(frame,kind,alpha):
    models=[]
    for group in ['G1','G2']:
        d=frame[(frame.group==group)&(frame.solar>0)&(frame.weight>0)]
        assert len(d)>=100 and np.isfinite(d.actual).all()
        names=WEATHER if kind=='direct' else FEATURES
        x,prep=design_fit(d,names)
        y=(d.actual if kind=='direct' else d.base-d.actual).to_numpy()/d.capacity.to_numpy()
        w=d.weight.to_numpy();w=w/w.mean()
        m=QuantileRegressor(quantile=.5,alpha=alpha,solver='highs')
        m.fit(x,y,sample_weight=w)
        models.append(dict(group=group,kind=kind,alpha=alpha,features=names,
                      coef=m.coef_.tolist(),intercept=float(m.intercept_),**prep))
    return models

def predict(frame,recipe):
    base=frame.base.to_numpy(dtype=float)
    result=base.copy()
    if recipe['kind']=='none':return result
    for gi,g in enumerate(['G1','G2']):
        ix=(frame.group==g).to_numpy();d=frame.loc[ix]
        if recipe['kind']=='scale':
            result[ix]=base[ix]*(1+recipe['strength']*(recipe['factors'][gi]-1))
        else:
            model=recipe['models'][gi]
            value=(design(d,model)@np.array(model['coef'])+model['intercept'])*d.capacity.to_numpy()
            if recipe['kind']=='direct':
                result[ix]=(1-recipe['strength'])*base[ix]+recipe['strength']*np.maximum(value,0)
            else:
                # Bound extrapolation; do not create generation from a zero base.
                correction=np.clip(value,-.1*base[ix],.1*base[ix])
                result[ix]=base[ix]-recipe['strength']*correction
    result=np.maximum(result,0)
    result[frame.solar.to_numpy()<=0]=0
    assert np.isfinite(result).all()
    return result

def loss(frame,pred):
    return float(.1*np.sum(frame.weight.to_numpy()*abs(pred-frame.actual.to_numpy())))

def select(frame,allow_direct=False):
    fitmask=frame.ts.sub(__import__('pandas').Timedelta(hours=1)).dt.day<=20
    train,valid=frame.loc[fitmask],frame.loc[~fitmask]
    assert len(train) and len(valid)
    recipes=[{'kind':'none','strength':0.}]
    factors=[]
    for g in ['G1','G2']:
        d=train[(train.group==g)&(train.base>1)&(train.weight>0)]
        x=(d.actual/d.base).to_numpy();w=(d.weight*d.base).to_numpy()
        ix=np.argsort(x);val=x[ix][np.searchsorted(np.cumsum(w[ix]),w.sum()/2)]
        factors.append(float(np.clip(val,.9,1.1)))
    for strength in [.25,.5,1.]:
        recipes.append(dict(kind='scale',strength=strength,factors=factors))
    for alpha in [.0001,.001,.01]:
        models=fit(train,'residual',alpha)
        for strength in [.25,.5,1.]:
            recipes.append(dict(kind='residual',strength=strength,models=models))
    if allow_direct:
        models=fit(train,'direct',.001)
        for strength in [.25,.5,1.]:
            recipes.append(dict(kind='direct',strength=strength,models=models))
    scores=[loss(valid,predict(valid,r)) for r in recipes]
    best=int(np.argmin(scores))
    table=[{'kind':r['kind'],'strength':r['strength'],
            'alpha':r.get('models',[{}])[0].get('alpha'),
            'validation_penalty':s} for r,s in zip(recipes,scores)]
    return recipes[best],table
