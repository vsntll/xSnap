# xSnap

An NFL predictor built on [xValue](https://github.com/vsntll/xValue)'s architecture: one shared data layer
(nflverse, via [nflreadpy](https://pypi.org/project/nflreadpy/)) feeding an outcome model, a player value
model and a per-game player rating, scored against the closing line.

Status: skeleton. The pipeline pulls schedules and builds the unified game table; models come next.

See [run_pipeline.md](run_pipeline.md) to run it.
