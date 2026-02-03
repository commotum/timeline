1. Number of distinct tasks evaluated: 4.
"We indeed see relative attention drastically improve negative log-likelihood (NLL) over baseline Transformer (Table 2)." (Section 4.1 J.S. BACH CHORALES)
"Table 3 shows that Transformer-based architectures fits this dataset better than LSTM-based models." (Section 4.2 PIANO-E-COMPETITION)
"When primed with an initial motif (Chopin's Étude Op. 10, No. 5) shown in the top left corner of Figure 4, we see the models perform qualitatively differently." (Section 4.2.1 QUALITATIVE PRIMING EXPERIMENTS)
"To explore the sequence-to-sequence setup of Transformers, we experimented with a conditioned generation task where the encoder takes in a given melody and the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 HARMONIZATION: CONDITIONING ON MELODY)

2. Number of trained model instances required to cover all tasks: 3.
"The JSB Chorale dataset consists of four-part scored choral music, which can be represented as a matrix where rows correspond to voices and columns to time discretized to sixteenth notes." (Section 3.1 Data representation)
"Compared to JSB Chorale, the piano performance data in the Piano-e-Competition dataset includes expressive timing information at much finer granularity and more voices. For the Piano-e-Competition we therefore use the performance encoding proposed by Oore et al. (2018) which consists of a vocabulary of 128 NOTE_ON events, 128 NOTE_OFFs, 100 TIME_SHIFTs allowing for expressive timing at 10ms and 32 VELOCITY bins for expressive dynamics (see A.2 for more details)." (Section 3.1 Data representation)
"To explore the sequence-to-sequence setup of Transformers, we experimented with a conditioned generation task where the encoder takes in a given melody and the decoder has to realize the entire performance, i.e. melody plus accompaniment." (Section 4.2.2 HARMONIZATION: CONDITIONING ON MELODY)

$$
\boxed{
\frac{4\ \text{tasks}}{3\ \text{models}} = 1.33
}
$$
