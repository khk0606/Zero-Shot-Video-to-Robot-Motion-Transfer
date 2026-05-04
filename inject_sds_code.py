"""
Deprecated helper script.

This repository now imports reward code directly from:
  dial-mpc/dial_mpc/envs/sds_reward_function.py

No source-file injection is performed anymore.
"""


def main() -> None:
    print("[DEPRECATED] inject_sds_code.py is disabled.")
    print("Reward is loaded directly from dial-mpc/dial_mpc/envs/sds_reward_function.py")
    print("Use gen_reward_code.py --output dial-mpc/dial_mpc/envs/sds_reward_function.py")


if __name__ == "__main__":
    main()
