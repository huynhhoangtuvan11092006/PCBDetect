"""Concurrent Load Testing utility for PCBDetect FastAPI Serving."""

import asyncio
import json
import logging
import time
from pathlib import Path

import httpx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pcbdetect.load_test")


async def send_single_request(
    client: httpx.AsyncClient,
    url: str,
    img_bytes: bytes,
    results: list[float],
):
    """Send a single multipart image prediction request and record latency."""
    t0 = time.perf_counter()
    files = {"file": ("test_pcb.jpg", img_bytes, "image/jpeg")}
    try:
        resp = await client.post(url, files=files, timeout=30.0)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        if resp.status_code == 200:
            results.append(lat_ms)
        else:
            logger.warning(f"Request failed: HTTP {resp.status_code}")
    except Exception as e:
        logger.error(f"Request error: {e}")


async def run_load_test(
    api_url: str = "http://localhost:8000",
    concurrency: int = 10,
    total_requests: int = 50,
    output_path: str = "docs/m7_load_test_metrics.json",
):
    """Execute concurrent load test against /predict endpoint."""
    test_dir = Path("data/processed/yolo/images/test")
    test_imgs = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
    if not test_imgs:
        raise FileNotFoundError(f"No test images found in {test_dir}")

    with open(test_imgs[0], "rb") as f:
        img_bytes = f.read()

    endpoint = f"{api_url}/predict"
    logger.info(f"Starting load test on {endpoint} with {concurrency} concurrent workers, {total_requests} requests...")

    results: list[float] = []
    t_start = time.perf_counter()

    transport = None
    try:
        async with httpx.AsyncClient() as check_client:
            await check_client.get(f"{api_url}/health", timeout=1.0)
    except Exception:
        logger.info("Live server not detected; using in-process ASGITransport with FastAPI app...")
        from pcbdetect.serving.app import app

        # Ensure lifespan is triggered
        from pcbdetect.serving.inference_service import InferenceService
        if not getattr(app.state, "service", None):
            app.state.service = InferenceService()
            app.state.startup_time = time.time()
        transport = httpx.ASGITransport(app=app)
        api_url = "http://testserver"
        endpoint = f"{api_url}/predict"

    async with httpx.AsyncClient(
        transport=transport,
        base_url=api_url if transport else None,
        limits=httpx.Limits(max_connections=concurrency * 2),
    ) as client:
        # Batch in chunks of concurrency
        queue = [img_bytes] * total_requests
        for i in range(0, total_requests, concurrency):
            batch = queue[i : i + concurrency]
            tasks = [send_single_request(client, endpoint, b, results) for b in batch]
            await asyncio.gather(*tasks)

    total_time = time.perf_counter() - t_start
    if not results:
        raise RuntimeError("All requests in load test failed!")

    success_count = len(results)
    throughput = success_count / total_time
    arr = np.array(results)

    metrics = {
        "endpoint": endpoint,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "successful_requests": success_count,
        "failed_requests": total_requests - success_count,
        "success_rate_pct": round((success_count / total_requests) * 100.0, 1),
        "total_elapsed_sec": round(total_time, 2),
        "throughput_req_per_sec": round(throughput, 2),
        "latency_mean_ms": round(float(np.mean(arr)), 2),
        "latency_p50_ms": round(float(np.percentile(arr, 50)), 2),
        "latency_p95_ms": round(float(np.percentile(arr, 95)), 2),
        "latency_min_ms": round(float(np.min(arr)), 2),
        "latency_max_ms": round(float(np.max(arr)), 2),
    }

    logger.info("=== Load Test Results ===")
    logger.info(f"Requests: {success_count}/{total_requests} (Success: {metrics['success_rate_pct']}%)")
    logger.info(f"Throughput: {metrics['throughput_req_per_sec']} req/s")
    logger.info(f"Latency: p50={metrics['latency_p50_ms']}ms, p95={metrics['latency_p95_ms']}ms, mean={metrics['latency_mean_ms']}ms")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved load test metrics to {output_path}")
    return metrics


if __name__ == "__main__":
    asyncio.run(run_load_test())
