# Examples

Real prompts and their rendered output, so a prompt can be judged against what
it actually produced instead of taken on faith. Nothing here is a benchmark —
it's a reference for what worked.

## Adding one

Each example is its own folder, named after the prompt (short, kebab-case):

```
Examples/
  neon-alley-rain/
    prompt.txt       # the exact prompt, verbatim. Negative prompt too, if used.
    settings.md       # resolution, frame count/seconds, seed, cfg/stg/upscale
    screenshot.png     # one representative frame, for the thumbnail below
    output.mp4
```

1. Generate a clip you're happy to show.
2. Create the folder and drop the four files in (the `.mp4` is small enough at
   these lengths/resolutions that committing it directly is fine).
3. Add a row to the index below: the screenshot as a thumbnail, linking to the
   folder; the video embedded with `<video>` so it plays inline on the GitHub
   page rather than just downloading.

Multiple renders of the same prompt (e.g. base vs. 2-stage upscale) can share
one folder — suffix each file instead (`output-base.mp4`/`output-upscaled.mp4`,
etc.) and give each its own index row, sharing the one `prompt.txt`.

## Index

<!-- Copy this row per example:
| <a href="neon-alley-rain/"><img src="neon-alley-rain/screenshot.png" width="200"></a> | [prompt](neon-alley-rain/prompt.txt) · [settings](neon-alley-rain/settings.md) | <video src="neon-alley-rain/output.mp4" controls width="200"></video> |
-->

| preview | prompt / settings | clip |
|---|---|---|
| <a href="singer-in-a-pub/"><img src="singer-in-a-pub/screenshot-base.png" width="200"></a> | [prompt](singer-in-a-pub/prompt.txt) · [settings](singer-in-a-pub/settings.md) · base, 768x512 | <video src="singer-in-a-pub/output-base.mp4" controls width="200"></video> |
| <a href="singer-in-a-pub/"><img src="singer-in-a-pub/settings-upscaled.png" width="200"></a> | [prompt](singer-in-a-pub/prompt.txt) · [settings](singer-in-a-pub/settings.md) · 2-stage upscale, 1536x1024 | <video src="singer-in-a-pub/output-upscaled.mp4" controls width="200"></video> |
| <a href="reporter-and-robots/"><img src="reporter-and-robots/screenshot.png" width="200"></a> | [prompt](reporter-and-robots/prompt.txt) · [settings](reporter-and-robots/settings.md) | <video src="reporter-and-robots/output.mp4" controls width="200"></video> |

See [`video-to-video/`](video-to-video/) for source-clip → output pairs.
