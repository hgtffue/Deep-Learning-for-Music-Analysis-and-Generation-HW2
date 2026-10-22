"""Real-data reference restricted to the first 32 bars of every song (same length as generations)."""
import json
d = json.load(open("../data/basic_sequences.json"))
out = []
for s in d:
    t, n, res = s["tokens"], 0, []
    for tok in t:
        if tok == "Bar":
            n += 1
            if n > 32:
                break
        res.append(tok)
    out.append({"song_name": s["song_name"], "tokens": res})
json.dump(out, open("../data/basic_sequences_32bars.json", "w"))
