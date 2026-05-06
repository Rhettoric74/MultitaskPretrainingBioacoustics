import torch

ckpt_path = "/scratch/Projects/CFP-04/CFP04-CF-029/checkpoints/jepa_audio/initial_run/last-v1.ckpt"  # change if needed

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

print("Top-level keys:")
print(list(ckpt.keys()))
print()

# Lightning checkpoints usually store weights under "state_dict"
state_dict = ckpt.get("state_dict", ckpt)

print(f"Number of parameters in state_dict: {len(state_dict)}")
print()

print("First 20 parameter keys:")
for i, k in enumerate(state_dict.keys()):
    print(k)
    #if i >= 19:
        #break