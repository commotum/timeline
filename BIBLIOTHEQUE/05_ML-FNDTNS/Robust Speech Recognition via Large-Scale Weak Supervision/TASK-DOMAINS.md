# Robust Speech Recognition via Large-Scale Weak Supervision (2022)
Source: Robust Speech Recognition via Large-Scale Weak Supervision.md

## Task Table
| Task | Input | In Dimension | In Dynamics | Attention Dynamic | State Dynamic | Output | Out Dimension | Out Dynamics |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Speech transcription | 30-second audio segments | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Transcript text tokens | 1D (t) (inferred) | Capped (inferred) |
| Speech translation (X→en) | 30-second audio segments | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | English text tokens | 1D (t) (inferred) | Capped (inferred) |
| Spoken language identification | 30-second audio segments | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Language token (99-language classification target) | 0D (inferred) | Fixed (inferred) |
| Voice activity detection | 30-second audio segments (including no-speech segments) | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | `< | nospeech | >` token (speech/no-speech label) | 0D (inferred) | Fixed (inferred) |
| Timestamp alignment prediction | 30-second audio segments; optional previous transcript context tokens | 1D (t) (inferred) | Fixed | Static (inferred) | Direct (inferred) | Timestamp tokens (start/end times) interleaved with caption tokens | 1D (t) (inferred) | Capped (inferred) |

## Summary
The paper describes a single multitask speech-processing model that handles transcription, speech translation, spoken language identification, voice activity detection, and timestamp-based alignment. Across tasks, inputs are fixed 30-second audio chunks, supporting a 1D (t) temporal input interpretation. Outputs include sequence generation tasks (transcript, translation, and timestamp-token alignment) and single-token classification-style outputs (language ID and no-speech detection), spanning inferred 1D (t) and 0D output dimensions. Attention and state dynamics are not explicitly labeled in the paper and are mapped as Static/Direct only by inference from the described seq2seq interface.

## Evidence
### Task: Speech transcription
- "In contrast to a lot of work on speech recognition, we train Whisper models to predict the raw text of transcripts without any significant standardization, relying on the expressiveness of sequence-to-sequence models to learn to map between utterances and their transcribed form." (Section 2.1)
- "The next token specifies the task (either transcription or translation) with an < | transcribe | > or < | translate | > token." (Section 2.3)
- Inference: `In Dimension`, `Attention Dynamic`, `State Dynamic`, `Out Dimension`, and `Out Dynamics` are inferred from temporal audio input and token-sequence generation over fixed windows; support includes "We break audio files into 30-second segments paired with the subset of the transcript that occurs within that time segment." (Section 2.1) and the seq2seq token-prediction setup in Section 2.3.

### Task: Speech translation (X→en)
- "We make an exception if the transcript language is English and add these pairs to the dataset as X→en speech translation training examples instead." (Section 2.1)
- "We study the translation capabilities of Whisper models by measuring their performance on the X→en subset of CoVoST2 (Wang et al., 2020b)." (Section 3.5)
- Inference: `In Dimension`, `Attention Dynamic`, `State Dynamic`, `Out Dimension`, and `Out Dynamics` are inferred by the same temporal-audio-to-token-sequence interface described in Sections 2.1 and 2.3, with fixed 30-second input segmentation.

### Task: Spoken language identification
- "First, we predict the language being spoken which is represented by a unique token for each language in our training set (99 total)." (Section 2.3)
- "To evaluate language identification, we use the Fleurs dataset (Conneau et al., 2022)." (Section 3.6)
- Inference: `In Dimension`, `Attention Dynamic`, `State Dynamic`, `Out Dimension`, and `Out Dynamics` are inferred as temporal audio input to a single-token classification target from the Section 2.3 token format.

### Task: Voice activity detection
- "We train on all audio, including segments where there is no speech (though with sub-sampled probability) and use these segments as training data for voice activity detection." (Section 2.1)
- "In the case where there is no speech in an audio segment, the model is trained to predict a < | nospeech | > token indicating this." (Section 2.3)
- Inference: `In Dimension`, `Attention Dynamic`, `State Dynamic`, `Out Dimension`, and `Out Dynamics` are inferred as temporal audio input to a single-token detection output (`< | nospeech | >`) under the multitask token format.

### Task: Timestamp alignment prediction
- "There are many different tasks that can be performed on the same input audio signal: transcription, translation, voice activity detection, alignment, and language identification are some examples." (Section 2.3)
- "For timestamp prediction, we predict time relative to the current audio segment, quantizing all times to the nearest 20 milliseconds which matches the native time resolution of Whisper models, and add additional tokens to our vocabulary for each of these." (Section 2.3)
- "Transcribing long-form audio using Whisper relies on accurate prediction of the timestamp tokens to determine the amount to shift the model's 30-second audio context window by, and inaccurate transcription in one window may negatively impact transcription in the subsequent windows." (Section 4.5)
- Inference: `In Dimension`, `Attention Dynamic`, `State Dynamic`, `Out Dimension`, and `Out Dynamics` are inferred from fixed-window temporal audio plus token-level timestamp outputs interleaved with caption tokens in Section 2.3.
