OPENPI_SYNC_CKPT=1 OPENPI_CKPT_FORMAT=zarr3 OPENPI_CKPT_SAVE_OPT_STATE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_clean_dish --exp-name=yihao_pi0_aloha_clean_dish --overwrite

OPENPI_SYNC_CKPT=1 OPENPI_CKPT_FORMAT=zarr3 OPENPI_CKPT_SAVE_PARAMS_ONLY=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_clean_dish --exp-name=yihao_pi0_aloha_clean_dish --overwrite


test on real aloha mobile robot: 
1. start the server (python 3.11)
uv run scripts/serve_policy.py --default-prompt "double-fold the shorts" --port 8000 policy:checkpoint --policy.config pi0_aloha_folding_shorts --policy.dir /home/allied/Disk2/Yihao/checkpoints/openpi/pi0_aloha_folding_shorts/yihao_pi0_aloha_folding_shorts/29999

# clean dish: 
/home/allied/Disk2/Yihao/checkpoints/openpi/pi0_aloha_clean_dish/yihao_pi0_aloha_clean_dish/29999

# folding_shorts: 
/home/allied/Disk2/Yihao/checkpoints/openpi/pi0_aloha_folding_shorts/yihao_pi0_aloha_folding_shorts/29999

# put sponge into pot
/home/allied/Disk2/Yihao/checkpoints/openpi/pi0_aloha_put_sponge_into_pot/yihao_pi0_aloha_put_sponge_into_pot/29999


2. start robot controller client (use python3.10)
install python packages according to the readme in aloha_real. 
install aloha system: https://docs.trossenrobotics.com/aloha_docs/2.0/getting_started.html

cd examples/aloha_real 
python3 robot_controller.py
