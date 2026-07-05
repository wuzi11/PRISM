#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline builder for LLM-derived PROGENy pathway direction priors.

For each unique drug-cell pair, query an OpenAI-compatible LLM and store a
14-dimensional prior vector in {-1, 0, 1}.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import scanpy as sc

from prism.model.prism_model import PROGENY_PATHWAY_NAMES
from prism.model.prism_utils import (
    DEFAULT_LLM_ENV,
    directions_to_prior_vector,
    drug_cell_key,
    load_env_file,
    parse_group,
    save_pathway_prior_store,
    unique_drug_cell_pairs,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def normalize_chat_url(url: str) -> str:
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def extract_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(text[start : end + 1])


def build_llm_prompt(cell_line: str, drug_name: str, pathway_names) -> tuple[str, dict]:
    allowed = ", ".join(pathway_names)
    system_prompt = (
        "You are a pharmacology expert for single-cell drug perturbation studies. "
        "Given a drug and cell line, predict the expected direction of change for "
        "each pathway after drug treatment relative to untreated control.\n\n"
        "Rules:\n"
        "Return JSON only.\n"
        f"Use only these pathway names: {allowed}.\n"
        "For each pathway, output 1 (expected up), -1 (expected down), or 0 "
        "(uncertain / no clear change)."
    )
    user_prompt = {
        "drug": drug_name,
        "cell_line": cell_line,
        "required_json_schema": {
            "pathway_directions": {name: "1, 0, or -1" for name in pathway_names},
        },
    }
    return system_prompt, user_prompt


def call_llm(
    cell_line: str,
    drug_name: str,
    pathway_names,
    args,
) -> dict:
    api_key = os.environ.get(args.llm_api_key_env, "")
    if not api_key:
        raise ValueError(
            f"Environment variable {args.llm_api_key_env} is required for LLM prior generation"
        )
    system_prompt, user_prompt = build_llm_prompt(cell_line, drug_name, pathway_names)
    request_body = {
        "model": args.llm_model,
        "temperature": args.llm_temperature,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt, indent=2)},
        ],
    }
    data = json.dumps(request_body).encode("utf-8")
    request = urllib.request.Request(
        normalize_chat_url(args.llm_api_url),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=args.llm_timeout) as response:
        raw = response.read()
    response_body = json.loads(raw.decode("utf-8"))
    content = response_body["choices"][0]["message"]["content"]
    return extract_json_object(content)


def parse_llm_prior_response(response: dict, pathway_names) -> list[int]:
    if "pathway_directions" in response:
        directions = response["pathway_directions"]
        prior = []
        for name in pathway_names:
            val = directions.get(name, 0)
            if isinstance(val, str):
                val = val.strip().lower()
                if val in {"up", "1", "positive"}:
                    prior.append(1)
                elif val in {"down", "-1", "negative"}:
                    prior.append(-1)
                else:
                    prior.append(0)
            else:
                iv = int(round(float(val)))
                prior.append(max(-1, min(1, iv)))
        return prior

    if "directions" in response:
        return directions_to_prior_vector(response["directions"], pathway_names)

    if "prior" in response:
        return [max(-1, min(1, int(round(float(v))))) for v in response["prior"]]

    raise ValueError("LLM response missing pathway_directions/directions/prior")


def fetch_llm_prior(
    cell_line: str,
    drug_name: str,
    key: str,
    pathway_names,
    args,
) -> tuple[str, list[int], str | None]:
    """Query LLM for one drug-cell pair; return (key, prior, error)."""
    last_error = None
    for attempt in range(max(1, args.llm_max_retries)):
        try:
            response = call_llm(cell_line, drug_name, pathway_names, args)
            prior = parse_llm_prior_response(response, pathway_names)
            return key, prior, None
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt + 1 < args.llm_max_retries:
                wait_s = args.llm_retry_delay * (2**attempt)
                time.sleep(wait_s)
    default = [int(args.default_prior)] * len(pathway_names)
    return key, default, str(last_error) if last_error else "unknown error"


def collect_groups(adata_paths: list[str]) -> list[str]:
    groups: list[str] = []
    for path in adata_paths:
        adata = sc.read_h5ad(path)
        groups.extend(adata.obs["Group"].astype(str).tolist())
    return groups


def main():
    parser = argparse.ArgumentParser(description="Build LLM pathway direction priors")
    parser.add_argument(
        "--env_file",
        default=os.environ.get("PRISM_ENV_FILE", DEFAULT_LLM_ENV),
        help="Env file with OPENAI_API_KEY and LLM settings",
    )
    parser.add_argument("--adata_path", required=True)
    parser.add_argument(
        "--extra_adata_paths",
        default="",
        help="Comma-separated extra AnnData paths (e.g. test set) to cover unseen pairs",
    )
    parser.add_argument(
        "--mode",
        choices=["llm", "zero", "random"],
        default="llm",
        help="llm: query LLM; zero: all-zero priors; random: random {-1,0,1} priors",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument(
        "--pathway_names",
        default=",".join(PROGENY_PATHWAY_NAMES),
        help="Comma-separated PROGENy pathway names",
    )
    parser.add_argument(
        "--llm_api_url",
        default=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com/v1"),
    )
    parser.add_argument(
        "--llm_model",
        default=os.environ.get("LLM_MODEL", "deepseek-v4-flash"),
    )
    parser.add_argument("--llm_api_key_env", default="OPENAI_API_KEY")
    parser.add_argument("--llm_temperature", type=float, default=0.0)
    parser.add_argument("--llm_timeout", type=int, default=60)
    parser.add_argument("--llm_max_retries", type=int, default=3)
    parser.add_argument("--llm_retry_delay", type=float, default=2.0)
    parser.add_argument("--resume", type=str2bool, default=True)
    parser.add_argument(
        "--replace_zero",
        type=str2bool,
        default=False,
        help="When resuming in llm mode, re-query priors that are all zeros",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of parallel LLM requests",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N completed queries",
    )
    parser.add_argument("--default_prior", type=int, default=0)
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Seed for random prior mode",
    )
    args = parser.parse_args()
    load_env_file(args.env_file)
    if args.env_file and os.path.isfile(args.env_file):
        print(f"Loaded env from {args.env_file}", flush=True)

    pathway_names = [name.strip() for name in args.pathway_names.split(",") if name.strip()]
    adata_paths = [args.adata_path]
    if args.extra_adata_paths.strip():
        adata_paths.extend(
            [p.strip() for p in args.extra_adata_paths.split(",") if p.strip()]
        )
    groups = collect_groups(adata_paths)
    pairs = unique_drug_cell_pairs(groups)

    existing: dict[str, list[int]] = {}
    output_path = Path(args.output_path)
    if args.resume and output_path.is_file():
        with open(output_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        existing = {str(k): list(v) for k, v in payload.get("priors", {}).items()}
        print(f"Resuming with {len(existing)} existing priors", flush=True)

    priors = dict(existing)
    if args.mode == "zero":
        for _, _, key in pairs:
            priors[key] = [0] * len(pathway_names)
        save_pathway_prior_store(
            output_path,
            priors,
            pathway_names=pathway_names,
            meta={"source": "zero", "n_pairs": len(priors)},
        )
        print(f"Saved {len(priors)} zero priors to {output_path}", flush=True)
        return

    if args.mode == "random":
        import random

        rng = random.Random(args.random_seed)
        choices = [-1, 0, 1]
        for _, _, key in pairs:
            priors[key] = [rng.choice(choices) for _ in pathway_names]
        save_pathway_prior_store(
            output_path,
            priors,
            pathway_names=pathway_names,
            meta={
                "source": "random",
                "random_seed": args.random_seed,
                "n_pairs": len(priors),
            },
        )
        print(
            f"Saved {len(priors)} random priors (seed={args.random_seed}) to {output_path}",
            flush=True,
        )
        return

    def needs_llm_query(key: str) -> bool:
        if key not in priors:
            return True
        if args.replace_zero and all(int(v) == 0 for v in priors[key]):
            return True
        return False

    pending = [(cell_line, drug, key) for cell_line, drug, key in pairs if needs_llm_query(key)]
    concurrency = max(1, int(args.concurrency))
    print(
        f"LLM prior build: {len(pairs)} total pairs, "
        f"{len(pending)} to query, {len(pairs) - len(pending)} skipped, "
        f"concurrency={concurrency}",
        flush=True,
    )

    if not pending:
        save_pathway_prior_store(
            output_path,
            priors,
            pathway_names=pathway_names,
            meta={"source": "llm", "llm_model": args.llm_model, "n_pairs": len(priors)},
        )
        print(f"Nothing to query. Saved {len(priors)} priors to {output_path}", flush=True)
        return

    save_lock = threading.Lock()
    completed = 0
    meta = {
        "source": "llm",
        "llm_model": args.llm_model,
        "n_pairs": len(priors),
        "concurrency": concurrency,
    }

    def persist_checkpoint(last_error: str | None = None):
        checkpoint_meta = dict(meta)
        checkpoint_meta["n_pairs"] = len(priors)
        if last_error:
            checkpoint_meta["last_error"] = last_error
        save_pathway_prior_store(
            output_path,
            priors,
            pathway_names=pathway_names,
            meta=checkpoint_meta,
        )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                fetch_llm_prior, cell_line, drug, key, pathway_names, args
            ): key
            for cell_line, drug, key in pending
        }
        for future in as_completed(futures):
            key, prior, error = future.result()
            with save_lock:
                priors[key] = prior
                completed += 1
                if error:
                    print(
                        f"[{completed}/{len(pending)}] failed {key}: {error}; using default",
                        flush=True,
                    )
                else:
                    print(f"[{completed}/{len(pending)}] LLM prior for {key}", flush=True)
                if completed % max(1, args.save_every) == 0 or completed == len(pending):
                    persist_checkpoint(error)

    save_pathway_prior_store(
        output_path,
        priors,
        pathway_names=pathway_names,
        meta={"source": "llm", "llm_model": args.llm_model, "n_pairs": len(priors), "concurrency": concurrency},
    )
    print(f"Saved {len(priors)} drug-cell priors to {output_path}", flush=True)


if __name__ == "__main__":
    main()
