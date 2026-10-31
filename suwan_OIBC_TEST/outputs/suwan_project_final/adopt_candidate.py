"""Explicit user adoption of the frozen candidate despite its failed stability gate."""
import argparse
import shutil
import run as r

def decision():
    path=r.HERE/'ADOPTION.json'
    if not path.exists():return None
    approval=r.read(path)
    assert approval['adopt_candidate'] is True
    assert r.p.sha256(r.OUT/'candidate_before_April.json')==approval['candidate_sha256'], 'Approval is for a different candidate'
    frozen=r.read(r.OUT/'frozen_before_May.json')
    return frozen['candidate']

def produce(sites,prices):
    candidate=decision();assert candidate is not None
    original_read=r.read
    original=original_read(r.OUT/'frozen_before_May.json')
    # Leave the original frozen experiment/gate evidence on disk unchanged.
    # The production routine consumes an effective, explicitly user-chosen recipe.
    def effective_read(path):
        value=original_read(path)
        if str(path)==str(r.OUT/'frozen_before_May.json'):
            return dict(value,selected=candidate,accepted=True)
        return value
    r.read=effective_read
    try:r.produce(sites,prices)
    finally:r.read=original_read
    record=original_read(r.OUT/'completion.json')
    record.update(adoption='explicit_user_override',passed_original_stability_gate=bool(original['accepted']),
                  adoption_record_sha256=r.p.sha256(r.HERE/'ADOPTION.json'))
    r.save(r.OUT/'completion.json',record)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');a=ap.parse_args()
    sites,prices=r.setup();candidate=decision();assert candidate is not None
    if a.build:
        backup=r.HERE/'submission_reference'
        if not backup.exists() and (r.HERE/'submission').exists():shutil.copytree(r.HERE/'submission',backup)
        if candidate['variant']=='feature':
            r.train_feature('production');r.feature_predictions('production')
    produce(sites,prices)

if __name__=='__main__':main()
