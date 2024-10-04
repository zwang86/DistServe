import os
import sys
import argparse
import aiohttp
import asyncio
import json
import time
from datetime import datetime

from structs import RequestResult

start_time = time.perf_counter()

async def task(url: str, 
               emission_time_ms: float, 
               prompt: str, 
               prompt_length: int, 
               output_length: int = 256, 
               slo_ratio: float = 1.0,
               verbose: bool = False
               ) -> RequestResult:
    global start_time

    headers = {"User-Agent": "Benchmark Client"}
    payload = {
        "prompt": prompt,
        "n": 1,
        "best_of": 1,
        "use_beam_search": False,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": output_length,
        "ignore_eos": False,
        "stream": False,
    }

    elapsed_time_ms = (time.perf_counter() - start_time) * 1000
    wait_time_ms = emission_time_ms - elapsed_time_ms
    if wait_time_ms > 0:
        await asyncio.sleep(wait_time_ms / 1000)
    request_sent_time = time.perf_counter()

    timeout = aiohttp.ClientTimeout(total=3 * 3600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            async with session.post(url=url, headers=headers, json=payload) as response:
                chunks = []
                async for chunk, _ in response.content.iter_chunks():
                    chunks.append(chunk)
            output = b"".join(chunks).decode("utf-8")

            try: 
                output = json.loads(output)
            except:
                print(f"Error with prompt '{prompt}': {output}")
                print(output)
                continue

            if verbose:
                print(f"Prompt: {prompt}\n\nOutput: {output['text']}")

            if "error" not in output:
                request_output = output
                break
            else:
                print(f"Failed to process the request: {output['error']}")
                print(f"Resending the request: {payload}")

        request_end_time = time.perf_counter()
    

    print(f"Prompt: {prompt}\nSendTime: {request_sent_time}, EndTime: {request_end_time}\nOutput: {request_output}\n\n")


    return RequestResult(
        prompt_length,
        output_length,
        request_sent_time,
        request_end_time,
        slo_ratio,
        token_timestamps=request_output["timestamps"],
        lifetime_events=request_output.get("lifetime_events", None)
    )

def evaluate_request(req: RequestResult, target_ttft: float, target_tpot: float):
    request_slo_attained = 1

    if req.ftl > target_ttft * req.slo_ratio:
        return 0, 0

    for i in range(1, len(req.token_timestamps)):
        if req.token_timestamps[i] - req.token_timestamps[i-1] > target_tpot * req.slo_ratio:
            request_slo_attained = 0
            break

    return request_slo_attained, len(req.token_timestamps) if request_slo_attained else 0

async def main(args: argparse.Namespace):
    with open(args.dataset, 'r') as f:
        emission_data = json.load(f)

    url = f"http://{args.host}:{args.port}/generate"
    verbose = args.verbose

    tasks = []
    results = []

    if not os.path.exists('logs'):
        os.makedirs('logs')

    start_time = time.perf_counter()

    for req in emission_data:
        emission_time_ms = req['emission_time_ms']
        prompt = req['prompt']
        prompt_length = req['input_length']
        output_length = req['output_length']
        slo_ratio = req['slo_ratio']
        tasks.append(asyncio.create_task(task(url, emission_time_ms, prompt, prompt_length, output_length, slo_ratio, verbose)))

    request_results = await asyncio.gather(*tasks)
    total_time = time.perf_counter() - start_time
    total_requests = len(emission_data)

    print(f"Total time: {total_time:.2f} s")
    print(f"Throughput:")
    print(f"\t{total_requests / total_time:.2f} requests/s")

    with open(args.output, "w") as f:
        json.dump([req.__dict__ for req in request_results], f, indent=4)

    request_attainment = 0
    token_attainment = 0
    for req in request_results:
        req_attainment, tok_attainment = evaluate_request(req, args.base_ttft, args.base_tpot)
        request_attainment += req_attainment
        token_attainment += tok_attainment

    print(f"Request SLO Attainment: \n\t{request_attainment / total_requests * 100:.2f}%s")
    print(f"\t{request_attainment} / {total_requests}")
    print(f"Token SLO Goodput: \n\t{token_attainment / total_time:.2f} tokens/s")
    print(f"\t{token_attainment} / {sum(req.output_len for req in request_results)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="8000"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False
    )
    parser.add_argument(
        "--output",
        type=str,
        default="logs/results.json"
    )
    parser.add_argument(
        "--base-ttft",
        type=float,
        default=0.3
    )
    parser.add_argument(
        "--base-tpot",
        type=float,
        default=0.1
    )

    asyncio.run(main(parser.parse_args()))
