import os
import json
import pandas as pd
import numpy as np

def export_difficulty_data():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load metadata (for cluster IDs)
    metadata_csv = os.path.join(project_root, "data", "bioscape", "biosoundscape_site_metadata.csv")
    if not os.path.exists(metadata_csv):
        print(f"[!] Metadata file not found at {metadata_csv}")
        return
    df_meta = pd.read_csv(metadata_csv)
    
    # Load richness / site coords from embeddings CSV
    richness_csv = os.path.join(project_root, "data", "bioscape", "western_cape_site_embeddings.csv")
    if not os.path.exists(richness_csv):
        print(f"[!] Richness file not found at {richness_csv}")
        return
    df_rich = pd.read_csv(richness_csv)
    
    # Load rankings CSV
    downloads_dir = os.path.expanduser("~/Downloads")
    stats_path = os.path.join(downloads_dir, "cluster_influence_stats(2).csv")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(downloads_dir, "cluster_influence_stats.csv")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(project_root, "reports", "cluster_influence_stats.csv")
        
    if not os.path.exists(stats_path):
        print(f"[!] Cluster stats CSV not found. Please verify the path.")
        return
        
    print(f"[*] Loading performance stats from: {stats_path}")
    df_stats = pd.read_csv(stats_path)
    
    # Join richness coordinates with metadata to get CLUSTER_ID
    # SiteID is in both. In df_rich it is "SiteID". In df_meta it is "SiteID".
    df_sites_merged = pd.merge(
        df_rich[['SiteID', 'Latitude', 'Longitude', 'richness']],
        df_meta[['SiteID', 'CLUSTER_ID']],
        on="SiteID",
        how="inner"
    )
    
    # Merge with cluster performance statistics
    df_merged = pd.merge(df_sites_merged, df_stats, left_on="CLUSTER_ID", right_on="cluster_id")
    
    # Filter noise clusters (-1)
    df_valid = df_merged[df_merged["CLUSTER_ID"] != -1].copy()
    
    # Prepare sites data
    sites = []
    for _, row in df_valid.iterrows():
        sites.append({
            "site_id": str(row["SiteID"]),
            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),
            "cluster_id": int(row["CLUSTER_ID"]),
            "richness": float(row["richness"]),
            "mean_r2_when_val": float(row["mean_r2_when_val"]) if not pd.isna(row["mean_r2_when_val"]) else None,
            "mean_r2_when_train": float(row["mean_r2_when_train"]) if not pd.isna(row["mean_r2_when_train"]) else None
        })
        
    # Prepare cluster rankings sorted from hardest to easiest
    df_clusters = df_stats.sort_values("mean_r2_when_val", ascending=True).copy()
    clusters = []
    for _, row in df_clusters.iterrows():
        c_id = int(row["cluster_id"])
        # Only include clusters that have sites in the dataset
        if c_id not in df_valid["CLUSTER_ID"].values:
            continue
        clusters.append({
            "cluster_id": c_id,
            "times_in_val": int(row["times_in_val"]),
            "times_in_train": int(row["times_in_train"]),
            "mean_r2_when_val": float(row["mean_r2_when_val"]) if not pd.isna(row["mean_r2_when_val"]) else None,
            "mean_r2_when_train": float(row["mean_r2_when_train"]) if not pd.isna(row["mean_r2_when_train"]) else None,
            "influence_diff": float(row["influence_diff"]) if not pd.isna(row["influence_diff"]) else None
        })
        
    output_path = os.path.join(project_root, "viz", "data", "difficulty_data.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump({
            "sites": sites,
            "clusters": clusters
        }, f, indent=2)
        
    print(f"[OK] Exported interactive map dataset to {output_path}")

if __name__ == "__main__":
    export_difficulty_data()
