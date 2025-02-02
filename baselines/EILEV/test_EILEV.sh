# Run `python samples/eilev_generate_action_narration.py --help` for details
# By default, the demo uses `kpyu/eilev-blip2-opt-2.7b`, which requires about 16GB of VRAM.
python samples/test_eilev_action.py \
  --device cuda \
  --model pretrained_models/eilev-blip2-opt-2.7b