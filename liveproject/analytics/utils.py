import pandas as pd

def build_recos(df, kpis, type_perf, hour_perf_json, top_posts):
    """
    Génère une liste de recommandations textuelles à partir des métriques.
    """
    recos = []

    # 1) Type le plus engageant
    best_type = type_perf[0]["status_type"]
    delta = type_perf[0]["num_reactions"] - type_perf[-1]["num_reactions"]
    recos.append(
        f"💡 Les {best_type}s génèrent en moyenne "
        f"{delta:+.0f} réactions de plus que le type le moins engageant."
    )

    # 2) Horaire optimal
    hp = pd.read_json(hour_perf_json)
    best_hours = hp.sort_values("num_reactions", ascending=False)\
                   .head(3)["hour"].tolist()
    heures = " – ".join(f"{h} h" for h in sorted(best_hours))
    recos.append(f"🕑 Pic d’activité vers {heures}. Programmez vos posts à ces heures.")

    # 3) Cluster top‑performer
    best_cluster = df.groupby("cluster_label")["num_reactions"]\
                     .mean().idxmax()
    recos.append(f"🎯 Le cluster {best_cluster} est le plus performant ; "
                 "sponsorisez ces contenus.")

    # 4) Posts à recycler
    ids = [str(p["status_id"]) for p in top_posts[:3]]
    recos.append("♻️ Recyclez ou épinglez les posts " + ", ".join(ids) + ".")

    return recos
