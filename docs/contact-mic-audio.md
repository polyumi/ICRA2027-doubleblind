# Contact-mic audio: why `mic_0` goes dead, and what was measured

The finger piezo channel stops converting partway through a capture and returns a frozen constant.
This is the investigation record: what was measured, what was ruled out, and what remains open. It
exists because the failure is invisible from every check the pipeline had — the codec keeps
clocking, ALSA reports `state: RUNNING`, `arecord` exits 0, and the recorded audio looks like a
plausible quiet signal.

## The symptom

The left channel of the finger audio converts for roughly 10–20 s, then latches to a constant
(observed values 120, sometimes 32767) with `std == 0` for every subsequent sample. The right
channel, on the same ADC and the same stream, runs indefinitely.

**The latch survives PCM close and reopen.** A fresh `arecord` on a wedged codec is dead within a
second. Only re-initialising the codec — `systemctl restart wm8960-soundcard` — clears it.

## Why it stayed hidden

Recording episodes are **8–14 s** (median 8.3 s over the 110 lightbulb episodes), so most end
before the failure onset. Continuous inference runs for minutes and hits it every time. That is the
whole reason "audio has worked before" and "the mic is dead" are both true.

It is nonetheless already in the training data. In `light_bulb_v1_polyumi.zarr.zip`:

- **33 of 110 episodes** contain dead `mic_0` rows
- **7172 of 28767 rows (24.9%)** are dead
- the first dead row falls at **median 0% through** its episode — i.e. those episodes began on a
  codec that was already wedged from the previous capture

The visualisation bags show the same: `gears_episode_58.mcap` is **88% dead** while
`outside_gears_episode_3`, `outside_picknplace_episode_2`, `picknplace_episode_26` and
`water_bottle_empty` are 0% dead, all at 8–10.5 s.

## Ruled out

Each of these was tested and changed nothing:

| variable | result |
|---|---|
| the contact mic | identical with the jack **empty** |
| the HAT | identical on **two** boards; the swap was unnecessary |
| several different mics | already known before this investigation |
| capture gain | tested at +28.5 dB and −2.25 dB |
| ADC high-pass filter | on and off |
| PortAudio | reproduces with plain `arecord` |
| CPU load / overflow | pure capture, no video, no ZMQ, **zero** overflows |
| sample rate | dies at 48 kHz too (faster: ~1 s) |
| settle delay after codec reset | 0/5/10/20 s → 22.8/7.0/8.5/11.5 s, no pattern |

The boot-time errors are a **red herring**:

```
wm8960 1-001a: failed to configure clock
wm8960 1-001a: ASoC: error at snd_soc_component_set_bias_level: -22
wm8960 1-001a: ASoC: Failed to prepare bias: -22
```

They appear identically on a working board, and originate in `wm8960_configure_clocking()` when no
divider combination matches the requested LRCLK — expected at probe before a stream sets a rate.

## The likely cause: the preset reads a pin the device tree never routes

`wm8960-soundcard.dts` routes the Mic jack to four pins, and **LINPUT2 is not one of them**:

```
simple-audio-card,routing =
        "LINPUT1", "Mic Jack",
        "LINPUT3", "Mic Jack",
        "RINPUT1", "Mic Jack",
        "RINPUT2", "Mic Jack";
```

`Line In` is declared as a widget but has **no routing entry at all**. On this HAT the physical
ports are MIC, LINE-IN and headphones; LINPUT2 is the Line-In jack, which nothing is plugged into.

`pi/alsa_preset` took the **left** channel from **LINPUT2**. DAPM powers down a widget with no
route, and an unpowered input stage settles to a constant — which is what the channel does.

That accounts for every observation without a hardware fault: two HATs behaving alike, the empty
jack changing nothing, gain and filtering being irrelevant, and a codec re-init reviving it briefly
by re-powering the stage before it settles again.

## What routing left to LINPUT1 measured

- **150 s soak clean** (rms 240–320 throughout) where LINPUT2 died at 20 s in every test
- taps register at **5800–8200 peak** against a ~1300 floor
- the sync chirp is still detected on **both** channels at the same instant (t=1.90 s for a chirp
  played 2.0 s into an 8 s capture; peak/mean 1273 and 2088)
- survives a reboot when installed as `/etc/wm8960-soundcard/wm8960_asound.state`

## Open questions

**The two channels are not independent sensors.** L/R correlation runs **0.87–0.98** with an
amplitude ratio near 1.0, on every tap. The overlay routes the Mic jack to LINPUT1 *and* RINPUT1,
so one microphone reaches both. The documented `L=piezo, R=air` contract does not describe this
hardware as configured. Only channel 0 becomes `mic_0`, so this does not break inference, but it
does mean `finger_air` is not an independent air mic.

**The left channel needs a gain decision.** At left −17.25 dB / right −2.25 dB the piezo sits at
**2–3% of full scale** (rms 245–346) while the air channel runs at **33–62%**. The attenuation was
set to stop a railing input back when the routing was wrong and is now just discarded resolution.

**There is an analog feedback path.** `Left Output Mixer Boost Bypass` ships **on at 100%**, wiring
the left input boost straight to the left output. Harmless while the left input was the dead
LINPUT2 pin; with LINPUT1 live it feeds the mic to the speaker and howls. It must be switched off
alongside any routing change.

## Constraints on any fix

- **Do not change the right channel.** `preproc/time_sync.py` cross-correlates `finger/finger_air`
  — channel 1, RINPUT1 — against the GoPro audio to find the sync chirp. Its gain and routing are
  load-bearing for data collection.
- **The codec shares one clock between capture and playback**, so the chirp must be played at the
  capture sample rate. `audio_streamer.py` already does this; passing `AUDIO_OUTPUT_SAMPLE_RATE`
  instead fails with `Invalid sample rate`.
- **Mono capture is not supported.** The device advertises `CHANNELS: 2` only, and a mono request
  froze capture in 1.8 s — a separate bug, fixed by defaulting `stream()` to `channels=2`.

## Verifying, in future

The provisioning check in [pi-provisioning.md](pi-provisioning.md) step 7 **cannot detect this**.
It records 5 s at 48 kHz and treats "no errors" as success; it passed with the contact-mic channel
completely flat. Any real check must inspect the samples, per channel, for longer than the failure
onset:

```bash
arecord -D hw:0,0 -f S16_LE -r 16000 -c 2 -d 60 /tmp/check.wav
```

then confirm neither channel has `std == 0` over any 10 s window. Ingest should apply the same test
at export time — `gears_episode_58` reached a training set at 88% dead with nothing complaining.
