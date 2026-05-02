"""
graph_query.py
Connects Python to TigerGraph via pyTigerGraph and executes
the installed GSQL queries to retrieve diseases for given symptoms.
"""

import os
from typing import List, Dict, Any
import pyTigerGraph as tg


class GraphQuery:
    def __init__(
        self,
        host: str       = None,
        graph_name: str = "HealthcareGraph",
        username: str   = None,
        password: str   = None,
        secret: str     = None,
    ):
        self.host       = host       or os.environ.get("TGRAPH_HOST", "https://your-tg-host.i.tgcloud.io")
        self.graph_name = graph_name
        self.username   = username   or os.environ.get("TGRAPH_USERNAME", "tigergraph")
        self.password   = password   or os.environ.get("TGRAPH_PASSWORD", "")
        self.secret     = secret     or os.environ.get("TGRAPH_SECRET", "")

        self.conn = self._connect()

    def _connect(self) -> tg.TigerGraphConnection:
        conn = tg.TigerGraphConnection(
            host=self.host,
            graphname=self.graph_name,
            username=self.username,
            password=self.password,
        )
        # Obtain an auth token (required for TigerGraph Cloud)
        if self.secret:
            conn.getToken(self.secret)
        return conn

    # ── Query helpers ──────────────────────────────────────────────────────

    def get_diseases_by_symptoms(
        self, symptoms: List[str], top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Call the installed getDiseasesBySymptoms query.
        Returns a list of dicts: [{disease, matched_symptoms, score}, ...]
        """
        # Normalise symptom names to match graph vertex IDs
        symptom_set = [s.strip().lower().replace(" ", "_") for s in symptoms]

        results = self.conn.runInstalledQuery(
            "getDiseasesBySymptoms",
            params={"symptoms": symptom_set, "topK": top_k},
        )

        diseases = []
        if results and isinstance(results, list):
            for item in results[0].get("Diseases", []):
                diseases.append({
                    "disease": item.get("v_id", ""),
                    "name":    item.get("attributes", {}).get("name", ""),
                    "matched_symptoms": list(
                        item.get("attributes", {}).get("@matchedSymptoms", [])
                    ),
                })
        return diseases

    def get_symptoms_for_disease(self, disease_name: str) -> List[str]:
        """
        Return all symptoms associated with a specific disease.
        """
        results = self.conn.runInstalledQuery(
            "getSymptomsForDisease",
            params={"diseaseName": disease_name},
        )
        symptoms = []
        if results and isinstance(results, list):
            for item in results[0].get("Symptoms", []):
                symptoms.append(item.get("attributes", {}).get("name", ""))
        return symptoms

    # ── Formatted context for LLM ──────────────────────────────────────────

    def build_graph_context(self, symptoms: List[str], top_k: int = 5) -> str:
        """
        Retrieve diseases from the graph and format the result as
        a human-readable context string to inject into the LLM prompt.
        """
        diseases = self.get_diseases_by_symptoms(symptoms, top_k)

        if not diseases:
            return "No matching diseases found in the knowledge graph."

        lines = ["Diseases retrieved from the medical knowledge graph:\n"]
        for i, d in enumerate(diseases, 1):
            matched = ", ".join(d["matched_symptoms"]) or "N/A"
            lines.append(f"{i}. {d['name']}")
            lines.append(f"   Matching symptoms: {matched}\n")

        return "\n".join(lines)


# ── Quick smoke test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    gq = GraphQuery()
    ctx = gq.build_graph_context(["fever", "headache", "fatigue"])
    print(ctx)
