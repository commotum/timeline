# Jukebox: A Generative Model for Music (Not specified in the paper)
Source: Jukebox- A Generative Model for Music.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Music generation (unconditional) | None (unconditional) | Not specified in the paper. | Not specified in the paper. | Static (inferred) | Constructed (inferred) | raw audio music (waveform) | 1D (t) | Open (inferred) |
| Music generation (artist/genre/timing conditioned) | artist labels; genre labels; timing signal (duration/start time/fraction elapsed) | 0D (inferred) | Fixed (inferred) | Static (inferred) | Constructed (inferred) | raw audio music (waveform) | 1D (t) | Open (inferred) |
| Lyrics-to-singing music generation | lyrics text (characters) | 1D (t) (inferred) | Capped (inferred) | Static (inferred) | Constructed (inferred) | raw audio music with singing | 1D (t) | Open (inferred) |
| Music generation (MIDI conditioned) | MIDI (symbolic composition) | 1D (t) (inferred) | Not specified in the paper. | Static (inferred) | Constructed (inferred) | raw audio music (waveform) | 1D (t) | Open (inferred) |
| Music continuation/completion (audio-primed) | existing audio segment (raw audio) | 1D (t) | Capped (inferred) | Static (inferred) | Constructed (inferred) | continuation raw audio | 1D (t) | Open (inferred) |
| Audio autoencoding/reconstruction | raw audio waveform | 1D (t) | Open (inferred) | Static (inferred) | Constructed (inferred) | reconstructed audio waveform | 1D (t) | Open (inferred) |

## Summary
Jukebox is presented as a raw-audio music generator that can produce songs with singing and extend samples beyond a fixed context via continuation sampling. The paper also describes conditional generation using artist/genre/timing metadata, lyrics-to-singing conditioning, MIDI conditioning, and audio-primed completion, alongside a VQ-VAE that reconstructs audio from discrete codes. Inputs cover 0D metadata and 1D temporal sequences (lyrics, MIDI, audio), and outputs are 1D raw-audio waveforms that can be continued beyond a single context window.

## Evidence
### Task: Music generation (unconditional)
- "We introduce Jukebox, a model that generates music with singing in the raw audio domain." (Abstract)
- "We consider music in the raw audio domain represented as a continuous waveform  $\mathbf{x} \in [-1,1]^T$" (Section 2. Background)
- "To generate music longer than the model's context length (12 in this figure), we repeatedly sample continuations" (Figure 2b)
- Inference: Out Dynamics marked Open based on continuation sampling, and Attention/State marked Static/Constructed because the model uses fixed-length Transformer contexts and discrete codes ("At each level, we use Transformers over the same context length of discrete codes"; "The code  $z_t$  is a discrete representation of the audio"). (Section 4; Figure 1 caption)

### Task: Music generation (artist/genre/timing conditioned)
- "We can condition on artist and genre to steer the musical and vocal style" (Abstract)
- "Additionally, we attach a timing signal for each segment at training time." (Section 4.1)
- "We have introduced Jukebox, a model that generates raw audio music imitating many different styles and artists." (Section 8. Conclusion)
- Inference: In Dimension/ Dynamics marked 0D/Fixed because inputs are labels and timing scalars; Attention/State marked Static/Constructed from fixed-context Transformers and discrete codes; Out Dynamics marked Open from continuation sampling ("At each level, we use Transformers over the same context length of discrete codes"; "The code  $z_t$  is a discrete representation of the audio"; "To generate music longer than the model's context length (12 in this figure), we repeatedly sample continuations"). (Section 4; Figure 1 caption; Figure 2b)

### Task: Lyrics-to-singing music generation
- "conditioning the model on the lyrics corresponding to each audio segment" (Section 4.2)
- "Lyrics-to-singing (LTS) task: The conditioning signal only includes the text of the lyrics" (Section 4.2)
- "pass a fixed-side window of characters centered around the current segment during training." (Section 4.2)
- "allowing the model to produce singing simultaneosly with the music." (Section 4.2)
- Inference: In Dimension marked 1D and In Dynamics marked Capped based on the character window; Attention/State marked Static/Constructed from fixed-context Transformers and discrete codes; Out Dynamics marked Open from continuation sampling ("pass a fixed-side window of characters centered around the current segment during training."; "At each level, we use Transformers over the same context length of discrete codes"; "The code  $z_t$  is a discrete representation of the audio"; "To generate music longer than the model's context length (12 in this figure), we repeatedly sample continuations"). (Section 4.2; Section 4; Figure 1 caption; Figure 2b)

### Task: Music generation (MIDI conditioned)
- "we can condition on lyrics to tell the singer what to sing, or on midi to control the composition." (Section 1. Introduction)
- "While we are able to steer our current model somewhat through lyric and midi conditioning" (Section 7. Future work)
- "We have introduced Jukebox, a model that generates raw audio music imitating many different styles and artists." (Section 8. Conclusion)
- Inference: In Dimension marked 1D for MIDI sequences; Attention/State marked Static/Constructed from fixed-context Transformers and discrete codes; Out Dynamics marked Open from continuation sampling ("At each level, we use Transformers over the same context length of discrete codes"; "The code  $z_t$  is a discrete representation of the audio"; "To generate music longer than the model's context length (12 in this figure), we repeatedly sample continuations"). (Section 4; Figure 1 caption; Figure 2b)

### Task: Music continuation/completion (audio-primed)
- "We can also generate novel completions of existing songs." (Section 1. Introduction)
- "The model can generate continuations of an existing audio signal" (Figure 2c)
- "We prime the model with 12 seconds of existing songs and ask it to complete them" (Section 5.3)
- Inference: In Dynamics marked Capped because priming uses a bounded segment; Attention/State marked Static/Constructed from fixed-context Transformers and discrete codes; Out Dynamics marked Open from continuation sampling ("We prime the model with 12 seconds of existing songs and ask it to complete them"; "At each level, we use Transformers over the same context length of discrete codes"; "The code  $z_t$  is a discrete representation of the audio"; "To generate music longer than the model's context length (12 in this figure), we repeatedly sample continuations"). (Section 5.3; Section 4; Figure 1 caption; Figure 2b)

### Task: Audio autoencoding/reconstruction
- "A one-dimensional VQ-VAE learns to encode an input sequence  $\mathbf{x}$" (Section 2.1)
- "a decoder  $D(\mathbf{e})$  that decodes the embedding vectors back to the input space." (Section 2.1)
- "The decoder takes the sequence of codebook vectors and reconstructs the audio." (Figure 1 caption)
- "we can use the same VQ-VAE to produce codes for arbitrary lengths of audio." (Section 4)
- Inference: In/Out Dynamics marked Open from arbitrary-length processing; Attention marked Static and State marked Constructed based on fixed-context processing and explicit discrete codes ("we can use the same VQ-VAE to produce codes for arbitrary lengths of audio."; "The code  $z_t$  is a discrete representation of the audio"). (Section 4; Figure 1 caption)
