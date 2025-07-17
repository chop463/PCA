import io, json
import pandas as pd
from django.shortcuts import render, redirect
from django.urls import reverse
from django.http import HttpResponse
from django.template.loader import render_to_string
from django.utils import timezone
from weasyprint import HTML

from .forms import CsvUploadForm
from .services import predict_clusters, get_model
from .utils import build_recos                     # ← fonction recommandations


# ─────────────────────────── Upload ─────────────────────────── #
def upload_csv(request):
    if request.method == "POST":
        form = CsvUploadForm(request.POST, request.FILES)
        if form.is_valid():
            df = pd.read_csv(form.cleaned_data["csv_file"])
            df = predict_clusters(df)                       # ajoute cluster_label
            request.session["csv"] = df.to_json(orient="split")
            return redirect(reverse("analytics:dashboard"))
    else:
        form = CsvUploadForm()
    return render(request, "analytics/upload.html", {"form": form})


# ───────────────────────── Dashboard ───────────────────────── #
def dashboard(request):
    if "csv" not in request.session:
        return redirect("analytics:upload")

    df = pd.read_json(io.StringIO(request.session["csv"]), orient="split")

    # KPI globaux
    kpis = {
        "mean_likes":    int(df["num_likes"].mean()),
        "mean_comments": int(df["num_comments"].mean()),
        "mean_shares":   int(df["num_shares"].mean()),
    }

    # KPI par cluster
    cluster_kpis = (
        df.groupby("cluster_label")[["num_likes", "num_comments", "num_shares"]]
          .mean().round(1).reset_index().to_dict(orient="records")
    )

    # Type performant
    type_perf = (
        df.groupby("status_type")["num_reactions"]
          .mean().round(1).sort_values(ascending=False)
          .reset_index().to_dict(orient="records")
    )

    # Réactions / heure
    df["hour"] = pd.to_datetime(df["status_published"]).dt.hour
    hour_perf = (
        df.groupby("hour")["num_reactions"].mean().round(1)
          .reset_index().to_json(orient="records")
    )

    # Top‑posts (3 par cluster)
    bundle = get_model()
    from sklearn.metrics.pairwise import euclidean_distances
    X_scaled = bundle["scaler"].transform(df[bundle["columns"]])
    X_pca    = bundle["pca"].transform(X_scaled)
    dists = euclidean_distances(X_pca, bundle["kmeans"].cluster_centers_)
    df["dist_centroid"] = dists[range(len(df)), df["cluster_label"]]

    top_posts = (
        df.sort_values(["cluster_label", "dist_centroid"])
          .groupby("cluster_label").head(3)
          [["cluster_label", "status_id", "status_type",
            "num_reactions", "num_comments", "num_shares"]]
          .to_dict(orient="records")
    )

    # Recommandations automatiques
    recommendations = build_recos(df, kpis, type_perf, hour_perf, top_posts)

    # Scatter JSON
    scatter_json = df[["num_likes", "num_comments", "cluster_label"]].to_json(orient="values")
    preview = df.head(100).to_dict(orient="records")

    # (Option) mettre en cache pour le PDF
    request.session["kpis"]          = json.dumps(kpis)
    request.session["cluster_kpis"]  = json.dumps(cluster_kpis)
    request.session["top_posts_cache"] = json.dumps(top_posts)

    context = {
        "kpis": kpis,
        "cluster_kpis": cluster_kpis,
        "type_perf": type_perf,
        "hour_perf": hour_perf,
        "top_posts": top_posts,
        "recommendations": recommendations,
        "clusters": sorted(df.cluster_label.unique().tolist()),
        "preview": preview,
        "scatter_json": scatter_json,
    }
    return render(request, "analytics/dashboard.html", context)


# ───────────────────────── Download CSV ─────────────────────── #
def download_csv(request):
    if "csv" not in request.session:
        return redirect("analytics:upload")
    df = pd.read_json(io.StringIO(request.session["csv"]), orient="split")
    resp = HttpResponse(content_type="text/csv")
    resp["Content-Disposition"] = "attachment; filename=live_clustered.csv"
    df.to_csv(resp, index=False)
    return resp


# ───────────────────────── PDF report ───────────────────────── #
def generate_pdf(request):
    if "csv" not in request.session:
        return redirect("analytics:upload")

    df = pd.read_json(io.StringIO(request.session["csv"]), orient="split")
    kpis          = json.loads(request.session["kpis"])
    cluster_kpis  = json.loads(request.session["cluster_kpis"])
    type_perf     = (
        df.groupby("status_type")["num_reactions"]
          .mean().round(1).sort_values(ascending=False)
          .reset_index().to_dict(orient="records")
    )
    top_posts = json.loads(request.session["top_posts_cache"])

    # ── Recalcul heure + recommandations ──
    df["hour"] = pd.to_datetime(df["status_published"]).dt.hour      # ← Ajout
    hour_perf = (
        df.groupby("hour")["num_reactions"]
          .mean().round(1).reset_index().to_json(orient="records")   # ← OK
    )
    recommendations = build_recos(df, kpis, type_perf, hour_perf, top_posts)

    html_string = render_to_string(
        "analytics/report.html",
        {
            "generated_at": timezone.now().strftime("%d/%m/%Y %H:%M"),
            "kpis": kpis,
            "cluster_kpis": cluster_kpis,
            "type_perf": type_perf,
            "top_posts": top_posts,
            "recommendations": recommendations,                       # ← passe aux templates
        }
    )
    pdf = HTML(string=html_string, base_url=request.build_absolute_uri()).write_pdf()

    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = "attachment; filename=rapport_live_facebook.pdf"
    return response