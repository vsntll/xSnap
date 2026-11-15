# xSnap

An NFL predictor built on [xValue](https://github.com/vsntll/xValue)'s architecture: one shared data layer
(nflverse, via [nflreadpy](https://pypi.org/project/nflreadpy/)) feeding an outcome model, a player value
model and a per-game player rating, scored against the closing line.

Status: outcome v1 live (team Elo + recent EPA form + starting QB, ridge on margin), scored against the closing line; value model next.

See [run_pipeline.md](run_pipeline.md) to run it.
