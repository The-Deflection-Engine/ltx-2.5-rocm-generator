# Video-to-video examples

Same idea as the main [Examples](../) index, but each entry needs a *source*
clip alongside the output — video-to-video only makes sense shown next to
what it started from.

## Adding one

Each example is its own folder, named after the prompt (short, kebab-case):

```
Examples/video-to-video/
  film-noir-bar/
    source.mp4         # the input clip
    prompt.txt          # the exact prompt, verbatim
    settings.md          # strength, resolution, frame count, seed
    screenshot.png        # one representative frame of the OUTPUT
    output.mp4
```

**`strength` is the one setting worth double-checking before you save it** —
it's easy to get backwards. `1.0` keeps the source frames fully clean (little
to no change); `0.0` is fully noised, giving the prompt maximum room to
diverge. That's the *opposite* direction from img2img denoise strength. For
a style change (e.g. "make it film noir"), a low number — `0.2`-`0.4` — is
what actually does something; the project default is `0.4`.

## Index

<!-- Copy this row per example:
| <a href="film-noir-bar/"><img src="film-noir-bar/screenshot.png" width="200"></a> | [prompt](film-noir-bar/prompt.txt) · [settings](film-noir-bar/settings.md) | <video src="film-noir-bar/source.mp4" controls width="160"></video> | <video src="film-noir-bar/output.mp4" controls width="160"></video> |
-->

| preview | prompt / settings | source | output |
|---|---|---|---|
| _(add yours)_ | | | |
