"""OpenAI-compatible endpoint, exercised with the official openai SDK (plus raw HTTP for health and metrics)."""
import base64, io, json, os, sys, time
import httpx
from openai import OpenAI, BadRequestError
from PIL import Image, ImageDraw
BASE = os.environ.get("CLEF_URL", "http://127.0.0.1:8100")
c = OpenAI(base_url=BASE + "/v1", api_key="unused"); ok = []     # the SDK insists on a key value; the server ignores it; ok = []
def check(name, cond, detail=""):
    ok.append(bool(cond)); print(("PASS " if cond else "FAIL ") + name + (f"  {detail}" if detail else ""), flush=True)

m = c.models.list().data; check("models.list", m and m[0].id == "clef-flash", m[0].id)
SCHEMA = {"type": "object", "properties": {
    "department": {"description": "Which team should handle the message?", "enum": ["billing", "technical", "sales"],
                   "x-criteria": {"billing": "Payments or invoices", "technical": "Bugs or outages", "sales": "Pricing, demos"}},
    "outage": {"type": "boolean", "description": "Is a service down?"},
    "urgency": {"type": "integer", "minimum": 1, "maximum": 5, "description": "How urgent is this?", "x-levels": ["not urgent", "low", "medium", "high", "critical"]}},
    "required": ["department", "outage", "urgency"], "additionalProperties": False}
RF = {"type": "json_schema", "json_schema": {"name": "triage", "strict": True, "schema": SCHEMA}}
cases = [("Our checkout started returning errors and all orders are blocked!", "technical", True), ("I was charged twice on my invoice.", "billing", False), ("Can I get a demo and a quote for 50 seats?", "sales", False)]
for text, dept, outage in cases:
    t = time.time(); r = c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": text}], response_format=RF); dt = (time.time() - t) * 1000
    v = json.loads(r.choices[0].message.content)
    check(f"json_schema: {text[:32]!r}", v["department"] == dept and v["outage"] is outage and isinstance(v["urgency"], int) and 1 <= v["urgency"] <= 5, f"{v} {dt:.0f} ms, finish={r.choices[0].finish_reason}, prompt_tokens={r.usage.prompt_tokens}")
probs = r.model_extra["clef"]["answers"]["department"]["probabilities"]; check("extension: per-option probabilities", abs(sum(probs.values()) - 1) < 0.01, str(probs))
r = c.chat.completions.create(model="clef-flash", messages=[{"role": "system", "content": "You triage support tickets for a SaaS company."}, {"role": "user", "content": "Hi"}, {"role": "assistant", "content": "How can I help?"},
    {"role": "user", "content": "The dashboard has been down since 9am."}], response_format=RF)
v = json.loads(r.choices[0].message.content); check("multi-turn conversation", v["department"] == "technical" and v["outage"] is True, str(v))
s2 = {"type": "object", "properties": {"category": {"description": "News category", "oneOf": [{"const": "sports", "description": "Sports and games"}, {"const": "business", "description": "Companies, markets"}, {"const": "science", "description": "Science and technology"}]}}}
r = c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": "Shares of the chipmaker rose 8% after strong quarterly earnings."}], response_format={"type": "json_schema", "json_schema": {"name": "n", "schema": s2}})
check("oneOf/const choices", json.loads(r.choices[0].message.content)["category"] == "business", r.choices[0].message.content)
TOOL = {"type": "function", "function": {"name": "route_ticket", "description": "Route a ticket", "parameters": SCHEMA}}
r = c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": cases[1][0]}], tools=[TOOL], tool_choice={"type": "function", "function": {"name": "route_ticket"}})
tc = r.choices[0].message.tool_calls[0]; check("tool call", r.choices[0].finish_reason == "tool_calls" and tc.function.name == "route_ticket" and json.loads(tc.function.arguments)["department"] == "billing", tc.function.arguments)
text = ""; fin = None
for ch in c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": cases[2][0]}], response_format=RF, stream=True):
    if ch.choices: text += ch.choices[0].delta.content or ""; fin = ch.choices[0].finish_reason or fin
check("stream", json.loads(text)["department"] == "sales" and fin == "stop", text)
img = Image.new("RGB", (400, 200), "white"); ImageDraw.Draw(img).text((20, 20), "RECEIPT Coffee 4.50", fill="black"); b = io.BytesIO(); img.save(b, "PNG")
url = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
r = c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": [{"type": "text", "text": "What is in the picture?"}, {"type": "image_url", "image_url": {"url": url}}]}],
    response_format={"type": "json_schema", "json_schema": {"name": "i", "schema": {"type": "object", "properties": {"kind": {"enum": ["receipt", "nature photo"], "description": "What is this image?"}}}}})
check("image input", json.loads(r.choices[0].message.content)["kind"] == "receipt", r.choices[0].message.content)
r = c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": cases[0][0]}], extra_body={"questions": {"down": {"type": "noul", "instructions": "Is a service down?"}}})
v = json.loads(r.choices[0].message.content)["down"]; check("native questions via extra_body (value = P(true))", isinstance(v, float) and 0 <= v <= 1, r.choices[0].message.content)
try: c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": "Write a poem"}]); check("free text rejected", False)
except BadRequestError as e: check("free text rejected with 400", "classifier" in str(e), str(e)[:90])
try: c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": "x"}], response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": {"type": "object", "properties": {"note": {"type": "string"}}}}}); check("free string property rejected", False)
except BadRequestError as e: check("free string property rejected with 400", "unsupported property" in str(e), str(e)[:90])
try: c.chat.completions.create(model="clef-flash", messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "http://example.com/a.png"}}]}], response_format=RF); check("remote image rejected", False)
except BadRequestError as e: check("remote image URL rejected with 400", "data:" in str(e))
check("no native endpoint", httpx.post(BASE + "/v1/systemone", json={}).status_code == 404)
check("health open", httpx.get(BASE + "/health").json()["ok"] is True)
mt = httpx.get(BASE + "/metrics").text; check("metrics", "clef_requests_total" in mt, " ".join(mt.split("\n")[:4]))
print(f"\n{sum(ok)}/{len(ok)} passed"); sys.exit(0 if all(ok) else 1)
