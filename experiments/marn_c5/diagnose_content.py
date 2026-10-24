"""Frozen full-val inference interventions; does not change source/checkpoints."""
import os
os.environ.update(OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',RF_HOME='/root/autodl-tmp/models/roboflow',HF_HOME='/root/autodl-tmp/cache/huggingface',TORCH_HOME='/root/autodl-tmp/cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/cache')
import sys,json,time,hashlib,copy,gc
from pathlib import Path
import torch
from pytorch_lightning import seed_everything
ROOT=Path('/root/autodl-tmp/project_marn-detr')
sys.path.insert(0,str(ROOT/'experiments/marn_c4'))
from query_only import load_experiment
from formal_tools import FullCOCOMetrics,atomic_json
from rfdetr.marn.data import MARNDataModule
from rfdetr.training.trainer import build_trainer
OUT=ROOT/'experiments/marn_c5/content_diagnostic'

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    started=time.time();torch.set_num_threads(4);seed_everything(42,workers=True)
    weights=ROOT/'runs/mechanism_pair_seed42_20261016_225833/marn/last_regular.pth'
    model=load_experiment(weights)
    mc=model.model_config.model_copy(update={'device':'cuda'})
    tc=model.train_config.model_copy(update={'accelerator':'gpu','devices':1,'use_ema':False,'eval_batch_size':8,'batch_size':16,'num_workers':8,'persistent_workers':False,'prefetch_factor':2,'compute_val_loss':False,'amp_dtype':'bf16','output_dir':str(OUT)})
    model.model_config=mc;model.train_config=tc
    dm=MARNDataModule(mc,tc);dm.setup('validate')
    assert len(dm._dataset_val)==5000
    original={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
    manifest={'scope':'same trained last regular model, inference interventions only; no training ablation or causal gain claim','weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'samples':5000,'max_dets':100,'seed':42,'modes':{},'source_sha256':{}}
    import rfdetr.marn.core as core,rfdetr.marn.integration as integration
    for path in (Path(core.__file__),Path(integration.__file__),Path(__file__)):
        manifest['source_sha256'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for mode in ('real','disabled','shuffled_features','shuffled_classes'):
        model.adapter.load_state_dict(original,strict=True);model.adapter.enabled=mode!='disabled';model.adapter.last_read={}
        if mode=='shuffled_features':
            gen=torch.Generator(device='cpu').manual_seed(42017)
            perm=torch.randperm(model.adapter.bank.features.shape[0],generator=gen)
            model.adapter.bank.features.copy_(model.adapter.bank.features[perm].clone())
        if mode=='shuffled_classes':
            model.adapter.bank.classes.copy_(model.adapter.bank.classes.roll(4))
        model.eval();folder=OUT/mode;folder.mkdir()
        frozen={k:v.detach().cpu().clone() for k,v in model.adapter.state_dict().items()}
        stats=[];reads=[]
        def residual_hook(module,inputs,output):
            q,mem,summary,accepted=inputs
            delta=(output.float()-q.float()).norm(dim=-1)
            rel=delta/q.float().norm(dim=-1).clamp_min(1e-6)
            mask=accepted
            stats.append(torch.stack((mask.sum(),delta.sum(),rel.sum(),delta.max(),rel.max())).detach())
        def read_hook(module,inputs,output):
            reads.append((module.last_read.get('selected',0),module.last_read.get('accepted',0)))
        handle=model.adapter.refiner.register_forward_hook(residual_hook)
        rh=model.adapter.register_forward_hook(read_hook)
        metric=FullCOCOMetrics(folder,max_dets=100,eval_base_model=True,log_per_class_metrics=False,eval_backend=tc.eval_backend)
        trainer=build_trainer(tc,mc,include_training_callbacks=False,enable_model_summary=False)
        trainer.callbacks=[c for c in trainer.callbacks if c.__class__.__name__!='COCOEvalCallback']+[metric]
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        trainer.validate(model,datamodule=dm,verbose=False)
        elapsed=time.monotonic()-t;handle.remove();rh.remove()
        for key,value in model.adapter.state_dict().items():
            torch.testing.assert_close(value.cpu(),frozen[key],rtol=0,atol=0)
        selected=sum(s for s,a in reads);accepted=sum(a for s,a in reads)
        row={'metrics':metric.captured['regular'],'seconds':elapsed,'peak_allocated_GiB':torch.cuda.max_memory_allocated()/1024**3,'selected':selected,'accepted':accepted,'frozen_state':True}
        if stats:
            tensors=torch.stack(stats).cpu();n=float(tensors[:,0].sum())
            row.update(mean_delta_norm_per_accepted=float(tensors[:,1].sum())/max(n,1),mean_relative_delta_per_accepted=float(tensors[:,2].sum())/max(n,1),max_delta_norm=float(tensors[:,3].max()),max_relative_delta=float(tensors[:,4].max()))
        manifest['modes'][mode]=row;manifest['wall_seconds']=time.time()-started
        atomic_json(OUT/'result.json',manifest)
        print('CONTENT_MODE',mode,json.dumps(row),flush=True)
        del trainer,metric;gc.collect();torch.cuda.empty_cache()
    model.adapter.load_state_dict(original,strict=True);model.adapter.enabled=True
    manifest['passed']=True;manifest['wall_seconds']=time.time()-started
    atomic_json(OUT/'result.json',manifest)
    print('CONTENT_DIAGNOSTIC_COMPLETED',json.dumps(manifest),flush=True)

if __name__=='__main__':main()
