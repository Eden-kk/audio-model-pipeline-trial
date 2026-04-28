// AudioWorkletProcessor: convert Float32 mic frames → Int16 PCM bytes and
// post them to the main thread. Runs on the audio rendering thread.
//
// Browser feeds us 128-sample frames in [-1, 1] floats. We assume the
// AudioContext was created at the target sample rate (the main thread
// does AudioContext({sampleRate: 16000})), so no resampling is needed
// in this worklet — the browser's resampler does it for us.
//
// Output: Int16Array buffers posted to port. Each buffer = one render
// quantum (~128 samples = ~8 ms at 16 kHz).

class PcmWorklet extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0]
    if (!input || input.length === 0) return true
    const channel = input[0]   // first channel only
    if (!channel || channel.length === 0) return true

    const out = new Int16Array(channel.length)
    for (let i = 0; i < channel.length; i++) {
      // Clamp to [-1, 1] then scale to int16 range
      const s = Math.max(-1, Math.min(1, channel[i]))
      out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF
    }
    this.port.postMessage(out.buffer, [out.buffer])
    return true
  }
}

registerProcessor('pcm-worklet', PcmWorklet)
