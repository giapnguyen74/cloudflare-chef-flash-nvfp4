"""HTTP benchmark of running clef servers, interleaved so drift hits every target equally.
usage: bench_http.py name=url [name=url ...]     e.g. host=http://127.0.0.1:8101 docker=http://127.0.0.1:8100"""
import asyncio, statistics, sys, time, httpx
targets = dict(a.split("=", 1) for a in sys.argv[1:]); ROUNDS = 3
Q = {"department": {"type": "choice", "instructions": "Which team should handle the message?", "criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "Pricing, demos"}}}
def body(n): return {"model": "clef-flash", "messages": [{"role": "user", "content": "Our checkout started returning errors. " + "Some filler context about the customer. " * n}], "questions": Q}
SIZES = [0, 100, 500, 1000]; lat = {t: {n: [] for n in SIZES} for t in targets}; toks = {}; thr = {t: {c: [] for c in (1, 8, 32)} for t in targets}; p95 = {t: {c: [] for c in (1, 8, 32)} for t in targets}
async def one(c, url, b):
    t = time.perf_counter(); r = await c.post(url + "/v1/chat/completions", json=b); r.raise_for_status(); return (time.perf_counter() - t) * 1000, r.json()["usage"]["prompt_tokens"]
async def main():
    async with httpx.AsyncClient(timeout=120) as c:
        for t, u in targets.items():
            for n in SIZES: await one(c, u, body(n))                       # warm
        for _ in range(ROUNDS):
            for t, u in targets.items():
                for n in SIZES:
                    for _ in range(10 if n < 500 else 4):
                        ms, k = await one(c, u, body(n)); lat[t][n].append(ms); toks[n] = k
                for conc in (1, 8, 32):
                    N = conc * 6; t0 = time.perf_counter(); ls = []
                    for k in range(0, N, conc): ls += [x[0] for x in await asyncio.gather(*[one(c, u, body(0)) for _ in range(conc)])]
                    thr[t][conc].append(N / (time.perf_counter() - t0)); p95[t][conc].append(sorted(ls)[int(len(ls) * .95) - 1])
    names = list(targets); w = 16
    print(f"{'single request (median)':34s}" + "".join(f"{n:>{w}s}" for n in names) + (f"{'difference':>14s}" if len(names) == 2 else ""))
    for n in SIZES:
        m = [statistics.median(lat[t][n]) for t in names]
        print(f"  {toks[n]:5d} input tokens: latency ms   " + "".join(f"{x:{w}.1f}" for x in m) + (f"{100*(m[1]/m[0]-1):+13.1f}%" if len(names) == 2 else ""))
        print(f"  {'':5s}               input tok/s  " + "".join(f"{toks[n]/x*1000:{w}.0f}" for x in m))
    print("short requests under load")
    for conc in (1, 8, 32):
        m = [statistics.median(thr[t][conc]) for t in names]
        print(f"  {conc:2d} clients: req/s                 " + "".join(f"{x:{w}.1f}" for x in m) + (f"{100*(m[1]/m[0]-1):+13.1f}%" if len(names) == 2 else ""))
        print(f"  {'':2s}          input tok/s           " + "".join(f"{x*toks[0]:{w}.0f}" for x in m))
        print(f"  {'':2s}          p95 latency ms        " + "".join(f"{statistics.median(p95[t][conc]):{w}.0f}" for t in names))
asyncio.run(main())
