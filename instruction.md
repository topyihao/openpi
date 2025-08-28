OPENPI_SYNC_CKPT=1 OPENPI_CKPT_FORMAT=zarr3 OPENPI_CKPT_SAVE_OPT_STATE=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_clean_dish --exp-name=yihao_pi0_aloha_clean_dish --overwrite

OPENPI_SYNC_CKPT=1 OPENPI_CKPT_FORMAT=zarr3 OPENPI_CKPT_SAVE_PARAMS_ONLY=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_aloha_clean_dish --exp-name=yihao_pi0_aloha_clean_dish --overwrite


test on real aloha mobile robot: 
1. start the server (python 3.11)
uv run scripts/serve_policy.py --default-prompt "clean the dish" --port 8000 policy:checkpoint --policy.config pi0_aloha_clean_dish --policy.dir /home/allied/Disk2/Yihao/checkpoints/openpi/pi0_aloha_clean_dish/27000

2. start robot controller (use python3.10)
python robot_controller_ros2.py
