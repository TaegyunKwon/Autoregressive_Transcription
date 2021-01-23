To train:

```
$python -m transcription.train with model_name=ONF
```

To transcribe testset:

ONF model
```
$python -m transcription.inference_onf {model.pt path ex: runs/ONF_gt_nllloss_0.0006_210101-000000/model-1000.pt}
```

AR model
```
$python -m transcription.inference_onf {model.pt path ex: runs/ARModel_gt_nllloss_0.0006_210101-000000/model-1000.pt} --save-path=results_ARModel
```

Evaluation on test set (after transcribe):
```
$python -m transcription.cal_metrics {inference path ex: results/ARModel} (--rep_type=ONF)
```