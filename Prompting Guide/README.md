# Writing prompts for LTX-2.5

Condensed from [LTX's official prompting guide](https://ltx.io/blog/prompting-guide-for-ltx-2),
with notes on what this project already automates. Read the source for the
full video examples — this is the reference version for quick lookup while
you're writing.

## The six things a good prompt covers

Write it as **one flowing paragraph**, not a list — the model reads it as a
scene description, not a checklist. Aim for 4-8 sentences that cover:

1. **The shot.** Cinematography terms for the style you want (wide/medium/
   close-up, genre-specific framing).
2. **The scene.** Lighting, color palette, surface textures, atmosphere.
3. **The action.** The core action as a natural sequence, start to finish,
   in present tense ("she turns and walks", not "she turned").
4. **The character(s).** Age, hair, clothing, distinguishing details —
   express emotion through physical cues (posture, gesture, expression),
   never a label like "she is sad."
5. **Camera movement.** State when the view shifts and how, and describe
   where the subject ends up after the move — that gives the model an
   endpoint to resolve the motion toward, not just a direction.
6. **Audio.** Ambient sound, music, dialogue. Quoted dialogue in `"..."`;
   name the language/accent if it matters.

Match your level of detail to shot scale — a close-up needs more precision
than a wide shot, since there's more of the frame for the model to get wrong.

## Vocabulary that reliably lands

Pulled from the guide's term lists — these are just genre/technical
vocabulary, not prescriptive, but they're words the model responds to
consistently:

- **Categories:** stop-motion, claymation, hand-drawn, comic book, cyberpunk,
  8-bit pixel, painterly, period drama, film noir, epic space opera,
  documentary.
- **Lighting/atmosphere:** flickering candles, neon glow, dramatic shadows,
  golden hour, fog, mist, rain, dust, smoke.
- **Camera language:** tracks, pans across, circles around, tilts upward,
  pushes in, pulls back, dolly in, handheld tracking, over-the-shoulder,
  wide establishing shot, static frame.
- **Pacing:** slow motion, time-lapse, rapid cuts, lingering shot,
  freeze-frame, seamless transition.
- **Sound/voice:** ambient descriptions (rain and wind, coffeeshop noise),
  delivery style (resonant, distorted radio-style, childlike curiosity),
  volume (whisper, mutters, shouts).

## What LTX-2.5 is good at

Cinematic single-subject compositions, emotive human moments (facial nuance,
subtle gesture), atmospheric weather/lighting effects, clean camera language
("slow dolly in" reads more consistently than vague mood words), stylized
aesthetics named early in the prompt, and dialogue/singing across languages.

## What to avoid

- **Emotional labels without a visual cue.** "Sad" alone does less than
  describing the posture and expression that reads as sad.
- **Readable text or logos.** Not currently reliable — avoid signage, brand
  names, printed material.
- **Chaotic or non-linear motion.** Jumping, juggling, fast twisting tend to
  glitch. Dancing generally works fine.
- **Too much at once.** Many characters/actions/objects in one prompt lowers
  the odds any single one lands correctly — start simple, layer on detail
  once the basics work.
- **Conflicting light sources.** "Warm sunset with cold fluorescent glow"
  confuses the lighting model unless the scene actually motivates both.

Iterating is expected, not a fallback — the guide's own framing is that fast
experimentation is part of the intended workflow.

## What this project already automates

- **Rewriting a short idea into the trained caption style** (the "one flowing
  paragraph, 4-8 sentences" structure above) is exactly what the **✨ Enhance
  Now** button does — it runs `google/gemma-4-E2B-it`, the enhancer the
  LTX-2.5 checkpoints were trained alongside, so it already knows this
  guide's structure. Write a rough idea, click Enhance Now, then edit the
  result — you don't need a second model or script for that part. GUI-only
  for now; the CLI has no enhance path.
- **STG** (`STG: fix anatomy / floating objects` in the GUI) targets the
  "chaotic motion" and duplicated-limb failure modes structurally, without
  needing a negative prompt or heavier CFG mode — see the main README for
  how it works.
- **Negative prompts only do anything under CFG quality mode.** The
  distilled 8-step schedule (the default) never evaluates the negative
  branch, so "avoid X" phrasing belongs in the positive prompt as a
  description of what you *do* want instead, unless CFG is on.

See [`Examples/`](Examples/) for prompts paired with what they actually produced.
