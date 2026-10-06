"""OpenAI chat-completions <-> Clef. Clef does not generate text: it scores the options of typed questions. So a chat request must say what to
decide, in one of two standard ways (or in the model's own question format):

  response_format = {"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {...}}}}
      each property is one question; the reply's message.content is a JSON object with one value per property
  tools + tool_choice
      the chosen function's `parameters` schema is used the same way; the reply is a tool call with those arguments
  questions = {...}   (extra body field, the model's own format: {id: {"type": "noul"|"choice"|"score", "instructions": ..., "criteria": ...}})

Property schema -> question type:
  {"type": "boolean"}                                   -> noul   (true if P(true) >= 0.5)
  {"enum": [...]}  or  {"oneOf": [{"const", "description"}]}  -> choice (the label); optional "x-criteria": {label: description}
  {"type": "integer"|"number", "minimum": a, "maximum": b}    -> score  (levels a..b; integer: rounded expected level, number: expected level);
                                                                 optional "x-levels": [description per level]
  "description" is the question's instruction (falls back to the property name).

The conversation becomes the state: one user message is used as is, several messages are rendered as "role: text" lines. Images are accepted as
data: URLs in image_url parts. Every reply also carries a non-standard top-level "clef" object with the full per-option probabilities.
"""
import json, time, uuid
from dataclasses import dataclass, field


class Unsupported(ValueError):
    pass


@dataclass
class Plan:
    request: dict                                   # the request in the model's own format: model, state, questions, images
    kinds: dict = field(default_factory=dict)       # question id -> ("bool",) | ("choice", [original labels]) | ("int"|"num", minimum)
    tool: str | None = None                         # function name when answering as a tool call


def _question(name, sch):
    if not isinstance(sch, dict): raise Unsupported(f"{name}: property schema must be an object")
    instr = sch.get("description") or sch.get("title") or name
    t = sch.get("type")
    if isinstance(t, list): t = next((x for x in t if x != "null"), None)
    alts = sch.get("oneOf") or sch.get("anyOf")
    if alts and all(isinstance(a, dict) and "const" in a for a in alts):
        labels = [a["const"] for a in alts]; crit = {str(a["const"]): a.get("description") or str(a["const"]) for a in alts}
        return {"type": "choice", "instructions": instr, "criteria": crit}, ("choice", labels)
    if "enum" in sch:
        labels = [v for v in sch["enum"] if v is not None]
        if len(labels) < 2: raise Unsupported(f"{name}: enum needs at least two values")
        xc = sch.get("x-criteria") or {}
        return {"type": "choice", "instructions": instr, "criteria": {str(v): xc.get(str(v), str(v)) for v in labels}}, ("choice", labels)
    if t == "boolean":
        return {"type": "noul", "instructions": instr}, ("bool",)
    if t in ("integer", "number") and isinstance(sch.get("minimum"), int) and isinstance(sch.get("maximum"), int) and sch["maximum"] > sch["minimum"]:
        lo, hi = sch["minimum"], sch["maximum"]
        if hi - lo > 100: raise Unsupported(f"{name}: at most 101 score levels")
        levels = sch.get("x-levels") or [str(v) for v in range(lo, hi + 1)]
        if len(levels) != hi - lo + 1: raise Unsupported(f"{name}: x-levels needs {hi - lo + 1} entries")
        return {"type": "score", "instructions": instr, "criteria": list(levels)}, ("int" if t == "integer" else "num", lo)
    raise Unsupported(f"{name}: unsupported property schema; use boolean, enum / oneOf-const, or integer/number with integer minimum and maximum")


def _from_schema(schema):
    if not isinstance(schema, dict) or not isinstance(schema.get("properties"), dict) or not schema["properties"]:
        raise Unsupported("the JSON schema must be an object with at least one property")
    qs, kinds = {}, {}
    for name, sch in schema["properties"].items(): qs[name], kinds[name] = _question(name, sch)
    return qs, kinds


def _state(messages):
    if not isinstance(messages, list) or not messages: raise Unsupported("messages must be a non-empty list")
    texts, images = [], []
    for m in messages:
        c = m.get("content"); parts = []
        if isinstance(c, str): parts.append(c)
        elif isinstance(c, list):
            for part in c:
                if part.get("type") == "text": parts.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    url = (part.get("image_url") or {}).get("url", "")
                    if not url.startswith("data:"): raise Unsupported("images must be data: URLs (the server does not fetch remote URLs)")
                    images.append(url)
                else: raise Unsupported(f"unsupported content part type {part.get('type')!r}")
        elif c is not None: raise Unsupported("message content must be a string or a list of parts")
        if m.get("role") == "tool" or parts: texts.append((m.get("role", "user"), "\n".join(parts)))
    if not texts and not images: raise Unsupported("no text or image content in messages")
    state = texts[0][1] if len(texts) == 1 and texts[0][0] == "user" else "\n".join(f"{r}: {t}" for r, t in texts)
    return state, images


def to_native(body: dict) -> Plan:
    if (body.get("n") or 1) != 1: raise Unsupported("n must be 1")
    state, images = _state(body.get("messages"))
    rf = body.get("response_format") or {}; tools = body.get("tools") or []; tool = None
    if isinstance(body.get("questions"), dict) and body["questions"]:
        qs = body["questions"]; kinds = {k: ("native",) for k in qs}
    elif rf.get("type") == "json_schema":
        qs, kinds = _from_schema((rf.get("json_schema") or {}).get("schema"))
    elif tools:
        fns = {t["function"]["name"]: t["function"] for t in tools if t.get("type") == "function"}
        tc = body.get("tool_choice")
        if isinstance(tc, dict): tool = (tc.get("function") or {}).get("name")
        elif len(fns) == 1 and tc != "none": tool = next(iter(fns))
        if tool not in fns: raise Unsupported("with several tools, name one in tool_choice: this model fills in a function's arguments, it does not pick the function")
        qs, kinds = _from_schema(fns[tool].get("parameters"))
    else:
        raise Unsupported("clef-flash is a classifier and cannot write free text. Send response_format={'type':'json_schema',...} with boolean / enum / "
                          "bounded-integer properties, or one tool with such parameters, or a native 'questions' object.")
    req = {"model": body.get("model") or "clef-flash", "state": state, "questions": qs}
    if images: req["images"] = images
    return Plan(req, kinds, tool)


def _value(kind, ans):
    if kind[0] == "bool": return ans["noul"] >= 0.5
    if kind[0] == "choice": return next(v for v in kind[1] if str(v) == ans["choice"])     # the caller's original label (keeps its JSON type)
    if kind[0] == "int": return kind[1] + int(round(ans["score"]))
    if kind[0] == "num": return round(kind[1] + ans["score"], 4)
    return ans.get("choice", ans.get("noul", ans.get("score")))                              # native questions: label / P(true) / expected level


def to_openai(plan: Plan, native: dict, model: str) -> dict:
    values = {q: _value(plan.kinds[q], a) for q, a in native["answers"].items()}
    text = json.dumps(values, ensure_ascii=False)
    if plan.tool:
        msg = {"role": "assistant", "content": None, "tool_calls": [{"id": "call_" + uuid.uuid4().hex[:24], "type": "function", "function": {"name": plan.tool, "arguments": text}}]}
        finish = "tool_calls"
    else:
        msg = {"role": "assistant", "content": text}; finish = "stop"
    n = native["usage"]["input_tokens"]
    return {"id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": n, "completion_tokens": 0, "total_tokens": n}, "clef": {"answers": native["answers"]}}


def sse(resp: dict, include_usage=False):
    """The whole answer is known at once: one content chunk, one finish chunk, [DONE]."""
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"]}
    msg = resp["choices"][0]["message"]; delta = {"role": "assistant"}
    if msg.get("tool_calls"): delta["tool_calls"] = [{"index": 0, **tc} for tc in msg["tool_calls"]]
    else: delta["content"] = msg["content"]
    chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
              {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0]["finish_reason"]}], "clef": resp["clef"]}]
    if include_usage: chunks.append({**base, "choices": [], "usage": resp["usage"]})
    for c in chunks: yield f"data: {json.dumps(c, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"
