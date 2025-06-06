import h5py

# Replace this with the actual path to your .h5 file
h5_path = "connect4_features.h5"

with h5py.File(h5_path, "r") as f:
    # 1. List all top‐level names
    print("Top‐level keys:", list(f.keys()))
    # e.g. ['features', 'q_values', 'game_ids']

    # 2. Inspect 'features' dataset
    if "features" in f:
        ds = f["features"]
        print("\nfeatures.shape:", ds.shape)
        print("features.dtype:", ds.dtype)
        # Read the first 5 samples (rows)
        first_five = ds[:5]        # shape = (5, 138) if you used 138‐dim features
        print("features[0:5]:\n", first_five)

    # 3. Inspect 'q_values' dataset
    if "q_values" in f:
        ds = f["q_values"]
        print("\nq_values.shape:", ds.shape)
        print("q_values.dtype:", ds.dtype)
        # Read the first 10 q_values
        print("q_values[0:10]:", ds[:10])

    # 4. Inspect 'game_ids' dataset (if present)
    if "game_ids" in f:
        ds = f["game_ids"]
        print("\ngame_ids.shape:", ds.shape)
        print("game_ids.dtype:", ds.dtype)
        # Read the first 10 game IDs
        print("game_ids[0:10]:", ds[:10])

    # 5. Print file‐level attributes (e.g. total_samples, total_games)
    print("\nFile attributes:")
    for attr_name, attr_val in f.attrs.items():
        print(f"  {attr_name}: {attr_val}")
