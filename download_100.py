import requests, os, time

ids = [
  345,174,209,159,175,43,1952,5765,10007,33562,
  215,910,1408,78,139,2226,164,1268,62,3089,
  45,541,284,2641,2891,242,24,160,532,4514,
  35,36,1013,5230,1264,12750,18857,11870,32,129,
  16,17396,289,1450,3536,157,1874,78120,1448,42091,
  1661,2852,155,1947,18081,1390,1695,1872,1063,6133,
  5116,408,1998,4363,2500,852,26659,36111,25015,15725,
  140,1097,2376,1325,3608,78131,14960,78124,5711,43081,
  1321,1081,3825,3007,3503,5720,2814,1254,2229,29002,
  78128,45368,754,5001,10660,1228,2300,201,78118,753, 1346,5199
]

os.makedirs("data", exist_ok=True)
failed = []

for i, gid in enumerate(ids):
    path = f"data/pg{gid}.epub"
    if os.path.exists(path):
        print(f"  [{i+1}/100] Already exists: pg{gid}.epub")
        continue
    for url in [
        f"https://www.gutenberg.org/ebooks/{gid}.epub.images",
        f"https://www.gutenberg.org/ebooks/{gid}.epub.noimages",
    ]:
        try:
            r = requests.get(url, timeout=30)
            if r.status_code == 200:
                open(path, "wb").write(r.content)
                print(f"  [{i+1}/100] Downloaded: pg{gid}.epub")
                break
        except: pass
    else:
        print(f"  [{i+1}/100] FAILED: pg{gid}")
        failed.append(gid)
    time.sleep(1)

print(f"\nDone. Failed: {failed if failed else 'none'}")