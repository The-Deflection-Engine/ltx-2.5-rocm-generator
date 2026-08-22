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

## Index

<!-- Copy this row per example:
| <a href="neon-alley-rain/"><img src="neon-alley-rain/screenshot.png" width="200"></a> | [prompt](neon-alley-rain/prompt.txt) · [settings](neon-alley-rain/settings.md) | <video src="neon-alley-rain/output.mp4" controls width="200"></video> |
-->

| preview | prompt / settings | clip |
|---|---|---|
| <a href="singer-in-a-pub/"><img src="singer-in-a-pub/screenshot.png" width="200"></a> | [prompt](singer-in-a-pub/prompt.txt) · [settings](singer-in-a-pub/settings.md) | <video src="singer-in-a-pub/output.mp4" controls width="200"></video> |
