"""
agents/ml_cluster_agent.py
ML Cluster Agent — automatic user/entity segmentation.

Pipeline:
  1. Select numeric columns (exclude ID-like)
  2. Scale features
  3. Auto-select K via silhouette score (K=2..8)
  4. K-Means clustering
  5. Profile each cluster: mean KPIs, dominant dimensions
  6. Name clusters using LLM (or rule-based fallback)
  7. UMAP 2D embedding for visualisation (optional, if installed)
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from agents.context import AnalysisContext, AgentResult
from core.logger import get_logger

logger = get_logger(__name__)


class MLClusterAgent(BaseAgent):
    name = "ml_cluster"
    description = "Auto-segments entities into meaningful groups via K-Means"

    def _run(self, context: AnalysisContext) -> AgentResult:
        df = context.df
        if df.empty:
            return self.skip("DataFrame is empty.")

        feature_cols = self._select_features(df)
        if len(feature_cols) < 2:
            return self.skip("Fewer than 2 numeric feature columns — clustering not meaningful.")

        n = len(df)
        if n < 20:
            return self.skip(f"Only {n} rows — need at least 20 for clustering.")

        X = df[feature_cols].dropna()
        valid_idx = X.index
        if len(X) < 20:
            return self.skip("Too many nulls in numeric columns.")

        # Scale
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.cluster import KMeans
            from sklearn.metrics import silhouette_score
        except ImportError:
            return self.skip("scikit-learn not installed. Run: pip install scikit-learn")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Auto-select K
        best_k, best_score, best_labels = 2, -1, None
        max_k = min(8, len(X) // 5)
        for k in range(2, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            score = silhouette_score(X_scaled, labels)
            if score > best_score:
                best_k, best_score, best_labels = k, score, labels

        # Attach cluster labels back to df
        df_clustered = df.loc[valid_idx].copy()
        df_clustered["cluster"] = best_labels

        # Profile clusters
        profiles = []
        for cluster_id in range(best_k):
            mask = df_clustered["cluster"] == cluster_id
            cluster_df = df_clustered[mask]
            profile = {"cluster_id": cluster_id, "size": int(mask.sum())}
            for col in feature_cols:
                profile[f"mean_{col}"] = round(float(cluster_df[col].mean()), 3)
            # Dominant categorical values
            cat_cols = df_clustered.select_dtypes(include="object").columns
            for col in list(cat_cols)[:3]:
                top = cluster_df[col].value_counts().index[0] if not cluster_df[col].empty else "—"
                profile[f"top_{col}"] = str(top)
            profiles.append(profile)

        profile_df = pd.DataFrame(profiles)

        # Name clusters
        cluster_names = self._name_clusters(profiles, feature_cols, context)

        # Optional UMAP
        umap_df = self._umap_embed(X_scaled, best_labels)

        quality = "strong" if best_score > 0.5 else "moderate" if best_score > 0.3 else "weak"
        summary = (
            f"K-Means clustering: {best_k} clusters found "
            f"(silhouette={best_score:.2f}, {quality} separation). "
            f"Cluster sizes: {', '.join(str(p['size']) for p in profiles)}. "
            f"Named segments: {', '.join(cluster_names)}."
        )

        return AgentResult(
            agent=self.name,
            status="success",
            summary=summary,
            data={
                "df_clustered": df_clustered,
                "profiles": profiles,
                "profile_df": profile_df,
                "cluster_names": cluster_names,
                "n_clusters": best_k,
                "silhouette_score": round(best_score, 3),
                "feature_cols": feature_cols,
                "umap_df": umap_df,
            },
        )

    def _select_features(self, df: pd.DataFrame) -> list[str]:
        numeric = df.select_dtypes(include=[np.number]).columns.tolist()
        return [
            c for c in numeric
            if not any(kw in c.lower() for kw in ["id", "index", "key", "seq", "row"])
            and df[c].nunique() > 1
        ]

    def _name_clusters(
        self, profiles: list[dict], feature_cols: list[str], context: AnalysisContext
    ) -> list[str]:
        from core.config import config
        if config.OPENAI_API_KEY or config.ANTHROPIC_API_KEY:
            try:
                from llm.client import LLMClient
                llm = LLMClient()
                biz_ctx = context.business_context
                prompt = (
                    f"Business context: {biz_ctx}\n"
                    f"Features used: {feature_cols}\n"
                    f"Cluster profiles:\n"
                )
                for p in profiles:
                    prompt += f"  Cluster {p['cluster_id']} (n={p['size']}): "
                    prompt += ", ".join(
                        f"{k}={v}" for k, v in p.items()
                        if k.startswith("mean_") or k.startswith("top_")
                    ) + "\n"
                prompt += "\nGive each cluster a short descriptive 2-3 word name. Return JSON list only."

                raw = llm.complete(
                    system="You are a data analyst naming customer segments. Return JSON list only.",
                    user=prompt,
                )
                raw = raw.strip().strip("```json").strip("```").strip()
                import json
                names = json.loads(raw)
                if isinstance(names, list) and len(names) == len(profiles):
                    return [str(n) for n in names]
            except Exception as e:
                logger.warning(f"LLM cluster naming failed: {e}")

        # Rule-based fallback
        names = []
        for p in profiles:
            means = {k.replace("mean_", ""): v for k, v in p.items() if k.startswith("mean_")}
            if means:
                top_feature = max(means, key=lambda k: abs(means[k]))
                val = means[top_feature]
                label = "high" if val > 0 else "low"
                names.append(f"{label}_{top_feature[:12]}")
            else:
                names.append(f"segment_{p['cluster_id']}")
        return names

    def _umap_embed(self, X_scaled, labels) -> pd.DataFrame | None:
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(X_scaled)
            return pd.DataFrame({
                "x": embedding[:, 0],
                "y": embedding[:, 1],
                "cluster": labels,
            })
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"UMAP failed: {e}")
        return None
