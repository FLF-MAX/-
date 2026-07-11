import requests

queries = [
    "我遇到一个很复杂的bug，快帮我看看",
    "我想听你用开心的声音说话",
    "我们今天一起探索一个新项目吧",
]

for q in queries:
    r = requests.post("http://localhost:11546/v1/express", json={"input": q})
    d = r.json()
    print(f"Input: {q}")
    print(f"  need={d['dominant_need']}, voice={d['tts']['voice']}, expr={d['live2d']['expression']}, motion={d['live2d']['motion']}")
