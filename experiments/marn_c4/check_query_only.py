"""Bounded query-only control checks before any formal mechanism training."""
import os
os.environ.update(RF_HOME='/root/autodl-tmp/models/roboflow',HF_HOME='/root/autodl-tmp/cache/huggingface',
                  TORCH_HOME='/root/autodl-tmp/cache/torch',XDG_CACHE_HOME='/root/autodl-tmp/cache',
                  OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import copy
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import torch
from pytorch_lightning import seed_everything
from check_formal import config
from formal_tools import configure_trainer,atomic_json
from query_only import ExperimentModule,QueryOnlyAdapter,export_experiment,load_experiment
from rfdetr.marn.integration import MARNAdapter,MARNConfig
from rfdetr.marn.data import MARNDataModule
from rfdetr.training.trainer import build_trainer

HERE=Path(__file__).resolve().parent


def main():
    torch.set_num_threads(4);seed_everything(42,workers=True)
    folder=HERE/'query_only_check';folder.mkdir(exist_ok=True)
    report={'passed':False}
    for device,amp in [('cpu',False),('cuda',True)]:
        old=MARNAdapter(64,[1,3,7],300,MARNConfig(warmup_steps=0)).to(device)
        new=QueryOnlyAdapter(64,[1,3,7],300,MARNConfig(warmup_steps=0)).to(device)
        new.load_state_dict(old.state_dict(),strict=True)
        assert sum(p.numel() for p in old.parameters())==sum(p.numel() for p in new.parameters())
        optimizer=torch.optim.SGD(new.parameters(),lr=.1)
        for step in range(2):
            raw=torch.randn(2,600,64,device=device,requires_grad=True)
            with torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=amp):
                output=new(raw,None,None,None,None)
                if step==0:torch.testing.assert_close(output,raw,rtol=0,atol=0)
                loss=(output*torch.randn_like(output)).mean()
            loss.backward()
            assert new.refiner.out.weight.grad is not None and torch.isfinite(new.refiner.out.weight.grad).all()
            if step:
                assert new.reader.key.weight.grad is not None and new.reader.key.weight.grad.abs().sum()>0
            new.observe(None,None,None,None)
            optimizer.step();new.commit();optimizer.zero_grad(set_to_none=True)
        assert not new.bank.valid.any() and not new.pending and new.snapshot is None
        before={k:v.clone() for k,v in new.state_dict().items()}
        new.eval()
        with torch.no_grad():new(raw,None,None,None,None)
        for key,value in new.state_dict().items():torch.testing.assert_close(value,before[key],rtol=0,atol=0)
    mc,tc=config(folder,2)
    checkpoint_path=folder/'last.ckpt'
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False) if checkpoint_path.exists() else None
    if saved is None:
        dm=MARNDataModule(mc,tc);dm.setup('fit')
        ids=sorted(dm._dataset_train.coco.getCatIds())
    else:
        assert saved['marn_experiment_variant']=='query_only' and saved['global_step']==2
        ids=saved['marn_class_ids']
    model=ExperimentModule(copy.deepcopy(mc),copy.deepcopy(tc),variant='query_only',class_ids=ids,marn_config=MARNConfig(warmup_steps=1))
    reference=ExperimentModule(copy.deepcopy(mc),copy.deepcopy(tc),variant='marn',class_ids=ids,marn_config=MARNConfig(warmup_steps=1))
    assert sum(p.numel() for p in model.parameters())==sum(p.numel() for p in reference.parameters())
    report['total_parameters']=sum(p.numel() for p in model.parameters())
    report['adapter_parameters']=sum(p.numel() for p in model.adapter.parameters())
    del reference
    if saved is None:
        trainer=build_trainer(tc,mc,limit_train_batches=4,limit_val_batches=2,max_epochs=1,enable_model_summary=False)
        ema,metric=configure_trainer(trainer,tc,folder)
        trainer.fit(model,datamodule=dm)
        assert trainer.global_step==2
    else:
        model.load_state_dict(saved['state_dict'],strict=True)
        average=copy.deepcopy(model)
        state={k.removeprefix('module.'):v for k,v in saved['callbacks'][
            'MARNEMACallback:online_bank_exact_copy_v1']['average_model_state_dict'].items() if k.startswith('module.')}
        average.load_state_dict(state,strict=True)
        ema=SimpleNamespace(_average_model=SimpleNamespace(module=average))
    assert int(model.adapter.committed_step)==2
    assert not model.adapter.bank.valid.any() and not ema._average_model.module.adapter.bank.valid.any()
    assert model.adapter.refiner.out.weight.detach().abs().sum()>0
    for source,expected in [('regular',model),('ema',ema._average_model.module)]:
        export_experiment(expected,folder/(source+'.pth'),source)
        restored=load_experiment(folder/(source+'.pth'))
        assert restored.variant=='query_only' and isinstance(restored.adapter,QueryOnlyAdapter)
        for key,value in expected.model.state_dict().items():
            torch.testing.assert_close(value.cpu(),restored.model.state_dict()[key],rtol=0,atol=0)
        expected.to('cuda').eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            raw=torch.randn(1,3,512,512,device='cuda')
            output=expected.model(raw)
            restored.to('cuda')
            actual=restored.model(raw)
        for key in ('pred_logits','pred_boxes'):
            torch.testing.assert_close(output[key],actual[key],rtol=0,atol=0)
        del restored
    checkpoint=torch.load(folder/'last.ckpt',map_location='cpu',weights_only=False)
    assert checkpoint['marn_experiment_variant']=='query_only'
    report.update(passed=True,checks=['parameter_count_equal','zero_init_identity','key_gradient_after_second_update',
                  'no_bank_reads_or_writes','eval_read_only','real_data_accumulation_and_ema',
                  'variant_metadata','regular_ema_exact_state_and_predictions'])
    atomic_json(folder/'query_only_result.json',report)
    print('QUERY_ONLY_CHECK_COMPLETED',json.dumps(report),flush=True)


if __name__=='__main__':main()
