These are small and weak neural net models for testing only.

See:
https://github.com/lightvector/KataGo/releases
or
https://d3dndmfyhecmj0.cloudfront.net/

for the actual latest neural nets.

### Constant policy networks
Commands used to generated constant policy networks used for testing:
```bash
python export_const_model.py \
  -saved-model-dir /nas/ucb/tony/go-attack/training/emcts1-v2/cp469-vis32/models/t0-s9851136-d2274300/saved_model \
  -export-dir ../cpp/tests/models \
  -model-name "const-policy-1" \
  -name-scope "swa_model" \
  -filename-prefix const-policy-1 \
  -for-cuda \
  -no-pass-lv 0 -pass-lv -3 -win-lv 1 -loss-lv 0 -nores-lv -5000
```

```bash
python export_const_model.py \
  -saved-model-dir /nas/ucb/tony/go-attack/training/emcts1-v2/cp469-vis32/models/t0-s9851136-d2274300/saved_model \
  -export-dir ../cpp/tests/models \
  -model-name "const-policy-2" \
  -name-scope "swa_model" \
  -filename-prefix const-policy-2 \
  -for-cuda \
  -no-pass-lv 0 -pass-lv 3 -win-lv -2 -loss-lv 0 -nores-lv -5000
```
