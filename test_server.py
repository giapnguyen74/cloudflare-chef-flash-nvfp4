import asyncio, httpx, time, base64, io, statistics
from PIL import Image, ImageDraw
import os
U=os.environ.get("CLEF_URL","http://127.0.0.1:8100")+"/v1/chat/completions"
dept={"department":{"type":"choice","instructions":"Which team should handle the message?","criteria":{"billing":"Payments or invoices","technical":"Bugs or outages","sales":"Pricing, demos"}}}
cases=[("Checkout returns errors and orders are blocked.","technical"),("I was charged twice on my invoice.","billing"),("Can I get a demo and quote for 50 seats?","sales")]
def body(content,qs): return {"model":"clef-flash","messages":[{"role":"user","content":content}],"questions":qs}
async def one(c,msg,exp):
    t=time.time(); r=await c.post(U,json=body(msg,dept)); dt=(time.time()-t)*1000
    return r.json()["clef"]["answers"]["department"]["choice"]==exp, dt
async def main():
    async with httpx.AsyncClient(timeout=60) as c:
        print("sequential:", [await one(c,*x) for x in cases])
        for conc in (1,8,32):
            reqs=[cases[i%3] for i in range(conc*4)]; t=time.time()
            res=[]
            for k in range(0,len(reqs),conc): res+=await asyncio.gather(*[one(c,*x) for x in reqs[k:k+conc]])
            wall=time.time()-t; lat=sorted(d for _,d in res)
            print(f"conc={conc:>2}: acc {sum(o for o,_ in res)}/{len(res)} throughput {len(res)/wall:.1f} req/s median {statistics.median(lat):.0f}ms p95 {lat[int(len(lat)*.95)-1]:.0f}ms")
        img=Image.new("RGB",(400,200),"white"); ImageDraw.Draw(img).text((20,20),"RECEIPT Coffee",fill="black"); b=io.BytesIO(); img.save(b,"PNG")
        url="data:image/png;base64,"+base64.b64encode(b.getvalue()).decode()
        r=await c.post(U,json=body([{"type":"text","text":"review"},{"type":"image_url","image_url":{"url":url}}],{"c":{"type":"choice","instructions":"What is this?","criteria":{"receipt":"A receipt","photo":"nature photo"}}})); print("image:",r.json()["clef"]["answers"])
        r=await c.post(U,json=body("x",{"q":{"type":"choice"}})); print("bad req ->",r.status_code,r.json())
asyncio.run(main())
