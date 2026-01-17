# Thermodynamics for a network of neurons: Signatures of criticality

Gašper Tkačik,<sup>a</sup> Thierry Mora,<sup>b</sup> Olivier Marre,<sup>c</sup> Dario Amodei,<sup>d,e</sup> Michael J. Berry II,<sup>e,f</sup> and William Bialek<sup>d,g,h</sup>

<sup>a</sup> Institute of Science and Technology Austria, Am Campus 1, A-3400 Klosterneuburg, Austria,

<sup>b</sup> Laboratoire de Physique Statistique, CNRS, UPMC and l'École Normale Supérieure,

24 rue Lhomond, 75231 Paris Cedex 05, France,

<sup>c</sup> Institut de la Vision, UMRS 968 UPMC, INSERM,

CNRS U7210, CHNO Quinze-Vingts, F-75012 Paris, France,

<sup>d</sup> Joseph Henry Laboratories of Physics, <sup>e</sup> Princeton Neuroscience Institute,

<sup>f</sup> Department of Molecular Biology, and <sup>g</sup> Lewis-Sigler Institute for Integrative Genomics,

Princeton University, Princeton, New Jersey 08544 USA,

<sup>h</sup> Initiative for the Theoretical Sciences, The Graduate Center,

City University of New York, 365 Fifth Ave., New York, New York 10016 USA

(Dated: July 23, 2014)

The activity of a neural network is defined by patterns of spiking and silence from the individual neurons. Because spikes are (relatively) sparse, patterns of activity with increasing numbers of spikes are less probable, but with more spikes the number of possible patterns increases. This tradeoff between probability and numerosity is mathematically equivalent to the relationship between entropy and energy in statistical physics. We construct this relationship for populations of up to N=160 neurons in a small patch of the vertebrate retina, using a combination of direct and model—based analyses of experiments on the response of this network to naturalistic movies. We see signs of a thermodynamic limit, where the entropy per neuron approaches a smooth function of the energy per neuron as N increases. The form of this function corresponds to the distribution of activity being poised near an unusual kind of critical point. Networks with more or less correlation among neurons would not reach this critical state. We suggest further tests of criticality, and give a brief discussion of its functional significance.

#### I. INTRODUCTION

Our perception of the world seems a coherent whole, yet it is built out of the activities of thousands or even millions of neurons, and similarly for our memories, thoughts, and actions. It seems difficult to understand the emergence of behavioral and phenomenal coherence unless the underlying neural activity also is coherent. Put simply, the activity of a brain—or even a small region of a brain devoted to a particular task—cannot be just the summed activity of many independent neurons. But if neurons are not independent, how do we describe their collective activity?

Statistical mechanics provides a language for connecting the interactions among many microscopic degrees of freedom to the macroscopic behavior of matter. Thus, we use statistical mechanics to give a quantitative theory of how a rigid solid emerges from the interactions between atoms, how a magnet emerges from the interactions between electron spins, and so on [1, 2]. Importantly, these are all collective phenomena: there is no sense in which a single molecule, or even a small cluster, can be described as solid or liquid; rather, solid and liquid are statements about the joint behaviors of many, many molecules.

At the core of equilibrium statistical mechanics is the Boltzmann distribution, which describes the probability of finding a system in any one of its possible microscopic states. As we consider systems with larger and larger numbers of degrees of freedom, this probabilistic description converges onto a deterministic, thermodynamic description. In the emergence of thermodynamics from statistical mechanics, many microscopic details are lost,

and many systems that differ in their microscopic constituents nonetheless exhibit quantitatively similar thermodynamic behavior. Perhaps the oldest example of this idea is the "law of corresponding states" [3].

The power of statistical mechanics to describe collective, emergent phenomena in the inanimate world led many people to hope that it might also provide a natural language for describing networks of neurons [4–7]. But if one takes the language of statistical mechanics seriously, then as we consider networks with larger and larger numbers of neurons, we should see the emergence of something like thermodynamics. If we can find this "thermodynamic limit" for real neurons, we can hope to find simpler universal behaviors for the network as a whole, independent of many microscopic details.

# II. THEORY

At first sight, the notion of a thermodynamics for neural networks seems hopeless. Thermodynamics is about temperature and heat, both of which are irrelevant to the dynamics of these complex, non–equilibrium systems. But all of the thermodynamic variables that we can measure experimentally in an equilibrium system can be calculated from the Boltzmann distribution, and hence statements about thermodynamics are equivalent to statements about this underlying probability distribution. It is then only a small jump to realize that all probability distributions over N variables can have an associated thermodynamics in the  $N \to \infty$  limit. This link between probability and thermodynamics is well stud-

ied by mathematical physicists [8], and has been a useful guide to the analysis of experiments on dynamical systems [9, 10], although perhaps still is not as widely appreciated as it might be.

To be concrete, we consider a system with N elements, and each element is described by a state  $\sigma_i$ ; the state of the entire system is given by  $\sigma \equiv \{\sigma_1, \sigma_2, \dots, \sigma_N\}$ . We are interested in the probability  $P(\sigma)$  that we will find the system in any one of its possible states. There is one state  $\sigma_0$  that is the most likely state, and we can measure all probabilities relative to the probability of this state. In particular, we can define

$$E(\sigma) = -\ln\left[\frac{P(\sigma)}{P(\sigma_0)}\right]. \tag{1}$$

In an equilibrium system, this is precisely the energy of each state (in units of  $k_BT$ ), but we can define this "energy" for any probability distribution. As discussed in detail in Appendix A, all of thermodynamics can be derived from the distribution of these energies. Specifically, what matters is how many states have  $E(\sigma)$  close to a particular value E. We can count this number of states, n(E), and define a local (microcanonical) entropy  $S(E) = \ln n(E)$ . If we can imagine a family of systems in which the number of degrees of freedom N varies, then a thermodynamic limit will exist provided that both the entropy and the energy are proportional to N at large N.

In most systems, including the networks that we study here, there are relatively few states that have high probability, and many more states with low probability; mathematically, n(E) is an increasing function. At large N, this competition between decreasing probability and increasing numerosity picks out a special value of  $E=E^*$ , which is the energy of the "typical" states that we actually see;  $E^*$  is the solution to the equation

$$\frac{dS(E)}{dE} = 1. (2)$$

For most systems, the energy  $E(\sigma)$  has only small fluctuations around  $E^*$ ,  $\langle (\delta E)^2 \rangle / (E^*)^2 \sim 1/N$ , and in this sense most of the states that we see have the *same* value of log probability per degree of freedom. But hidden in the function S(E) are all the parameters describing the interactions among the N degrees of freedom in the system. At special values of these parameters,  $[d^2S(E)/dE^2]_{E=E^*} \to 0$ , and the variance of E diverges as E0 becomes large. This is a critical point, and is mathematically equivalent to the divergence of the specific heat in an equilibrium system [11].

These observations focus our attention on the "density of states" n(E). Rather than asking how often we will see specific combinations of neurons spiking while the others remain silent, we ask how many states there are with a particular probability. If we can estimate n(E) for patterns of neural activity, then we can construct a thermodynamics for the network.

#### III. AN EXPERIMENTAL EXAMPLE

The vertebrate retina offers a unique system in which the activity of most of the neurons comprising a local circuit can be monitored simultaneously using multielectrode array recordings. As described more fully in Ref [12], we stimulated salamander retina with naturalistic gravscale movies of fish swimming in a tank (Fig. 1A), while recording from 100–200 retinal ganglion cells (RGCs), the output cells of the retina that in an intact animal project to the central brain; additional experiments used artificial stimulus ensembles, as described in Appendix B. Sorting the raw data [13], we could reliably identify single spikes from 160 neurons whose activity passed our quality checks and was stable for the whole  $\sim$ 2 hour duration of the experiment; a small segment of the data is shown in Fig 1B. Importantly, our experiments monitored a substantial fraction of the RGCs in the area of the retina from which we record, capturing the behavior of an almost complete local population responsible for encoding a small patch of the visual world. The experiment collected a total of  $\sim 2 \times 10^6$  spikes, and time was discretized in bins of duration  $\Delta \tau = 20 \,\mathrm{ms}$ ; for

![](_page_1_Figure_11.jpeg)

FIG. 1: Counting states in the response of retinal ganglion cells. (A) A single frame from the naturalistic movie, red ellipse indicates the approximate extent of a receptive field center for a typical retinal ganglion cell. (B) Responses of N=160 neurons to a 19.2s naturalistic movie clip; dots indicate the times of action potentials from each neuron. In subsequent analyses these events are discretized into binary (spike/silence) variables in time slices of  $\Delta \tau = 20\,\mathrm{ms}$ . (C) 1000 most common binary patterns of activity across N=160 neurons, in order of their frequency. (D) Number of occurrences of each pattern, with labels for the total number of spikes in each pattern.

each neuron i,  $\sigma_i = 1$  in a bin denotes that the neuron emitted at least one spike, and  $\sigma_i = 0$  that it was silent.

#### IV. COUNTING STATES

Conceptually, estimating the function n(E) and hence the entropy vs energy is easy: we count how often each state occurs, thus estimating its probability, and then count how many states have (log) probabilities in a given range. In Fig 1C and D we show the first steps in this process. We identify the unique patterns of activity—combinations of spiking and silence across all 160 neurons—that occur in the experiment, and then count how many times each of these patterns occurs.

Even without trying to compute S(E), the results of Fig 1D are surprising. With N neurons that can either spike or remain silent, there are  $2^N$  possible states. We know that not all these states can be visited equally often, since spikes are less common than silences, but even taking account of this bias, and trying to capture the correlations among neurons, our best estimate of the entropy for the patterns of activity we observe is  $S \sim 0.15 \, \text{bits/neuron}$  (see below). With  $N = 160 \, \text{cells}$ , this is a total entropy of S = 24 bits, which means that the patterns of activity are spread over  $2^S \sim 1.67 \times 10^7$ possibilities. This is one hundred times larger than the number of samples that we collect during our experiment. Indeed, most of the states that we see in the full population occur only once. But roughly one thousand states occur with sufficient frequency that we can make a reasonable estimate of their probability just by counting across the  $\sim 2\,\mathrm{hrs}$  of the experiment. Thus, the probability distribution  $P(\sigma)$  is extremely inhomogeneous.

To probe more deeply into the long tail of low probability events, we can construct models of the distribution of states. As described in detail elsewhere [12], we have done this using the maximum entropy construction [14]: we take from experiment certain average behaviors of the network, and then search for models that match these data but otherwise have as little structure as possible. This approach works if we can show that matching a relatively small number of features produces a model which predicts many other aspects of the data.

The maximum entropy approach to describing activity in networks of neurons has been explored, in several different systems, for nearly a decade [15–24], and there have been parallel efforts to use this approach in other biological systems [25–37]. Recently we have used the maximum entropy method to build models for the activity of up to N=120 neurons in the experiments described above [12]. As summarized in Fig 2, we take from experiment the mean probability of each neuron generating a spike  $(\langle \sigma_i \rangle)$ , the correlations between spiking in pairs of neurons  $(\langle \sigma_i \sigma_j \rangle)$ , and the probability that K out of the N neurons spike in the same small window of time [P(K)]. Mathematically, the maximum entropy models

consistent with these data have the form

$$P(\lbrace \sigma_{\rm i} \rbrace) = \frac{1}{Z} \exp\left[-E(\lbrace \sigma_{\rm i} \rbrace)\right] \tag{3}$$

$$E(\{\sigma_{i}\}) = -\sum_{i=1}^{N} h_{i}\sigma_{i} - \frac{1}{2} \sum_{i,j=1}^{N} J_{ij}\sigma_{i}\sigma_{j} - V(K), (4)$$

where  $K = \sum_{i=1}^{N} \sigma_i$  counts the number of neurons that spike simultaneously, and Z is set to insure normalization. All of the parameters  $\{h_i, J_{ij}, V(K)\}$  are determined by the measured averages  $\{\langle \sigma_i \rangle, \langle \sigma_i \sigma_j \rangle, P(K)\}$ .

This model accurately predicts the correlations among triplets of neurons (Fig 7 in [12]), and the probability of spiking in individual neurons depends on the pattern of activity in the rest of the population as the model predicts (Fig 9 in [12]). One can even predict the time dependent response of single cells from the behavior of the population, without reference to the visual stimulus

![](_page_2_Figure_12.jpeg)

FIG. 2: Maximum entropy models for retinal activity in response to natural movies [12]. (A) The correlation coefficients between pairs of neurons (red = positive, blue = negative) for a 120-neurons subnetwork. Inset shows the distribution of the correlation coefficients over the population. (B) The pairwise coupling matrix of the inferred model,  $J_{ij}$  from Eq (4). Inset shows the distribution of these pairwise couplings across all pairs ij. (C) The average probability of spiking per time bin for all neurons (sorted). (D) The corresponding bias terms  $h_i$  in Eq. (4). (E) The probability P(K) that K out of the N neurons spike in the same time bin. (F) The corresponding global potential V(K) in Eq. (4). Notice that panels A, C, and E describe the statistical properties observed for these neurons, while panels B, D, and F describe parameters of the maximum entropy model that reproduces these data within experimental errors.

![](_page_3_Figure_1.jpeg)

FIG. 3: Entropy vs. energy. (A) Computed directly from the data using Eqs (A6, A10). Different colors show results for different numbers of neurons; in each case we choose 1000 groups of size N at random, and points are means with standard deviations over groups. Inset shows extrapolations of the energy per neuron at fixed energy per neuron, summarized as black points with error bars in the main figure. Dashed line is the best linear fit to the extrapolated points,  $S/N = (0.974 \pm 0.021)(E/N) + (-0.005 \pm 0.003)$ . (B) Computed from the maximum entropy models. We choose  $N = 20, 40, \dots, 120$  neurons out of the population from which we record, and for each of these subnetworks we construct maximum entropy models as in Eq (4); for details of the entropy calculation see Appendix C. Inset shows results for many subnetworks, and the main figure shows means and standard deviations across the different groups of N neurons. Black points with error bars are an extrapolation to infinite N, as in (A), and the dashed line is S = E.

(Fig 15 in [12]). Most important for our present discussion, we have checked that the distribution of the energy  $E(\{\sigma_i\})$  across the patterns of activity that occur in the data agrees with the distribution predicted by the model, deep into the tail of patterns that occur only once in the two hour long experiment (Fig 8 in [12]). This distribution is very closely related to the plot of entropy vs energy that we would like to construct, and so the agreement with experiment gives us confidence.

The direct counting of states (Fig 1) and the maximum entropy models (Fig 2) give us two complementary ways of estimating the function n(E) and hence the entropy vs energy in the same data set. Results are shown in Fig 3, with some technical issues discussed in Appendix C.

As emphasized above and in Appendix A, the plot of entropy vs energy contains all of the thermodynamic behavior of a system, and this has a meaning for any probability distribution, even if we are not considering a system at thermal equilibrium. Thus, Fig 3 is as close as we can get to constructing the thermodynamics of this network. With the direct counting of states we see less and less of the plot at larger N, but the part we can see is approaching a limit as  $N \to \infty$ , and this is confirmed by the results from the maximum entropy models. This by itself is a significant result. If we write down a model like Eq (4), then in a purely theoretical discussion we can scale the couplings between neurons  $J_{ij}$  with N to guarantee the existence of a thermodynamic limit [5], but with  $J_{ii}$ constructed from real data, as we do here, we can't impose this scaling ourselves—either it emerges from the data or it doesn't. We can make the emergence of the

thermodynamic limit more precise by noting that, at a fixed value of S/N, the value of E/N extrapolates to a well defined limit in a plot vs. 1/N, as in the inset to Fig 3A. The results of this extrapolation from the real data are strikingly simple: the entropy is equal to the energy, within (small) error bars.

## V. INTERPRETING S(E)

What thermodynamic behavior is predicted by Fig 3? If the plot of entropy vs energy is a straight line with unit slope, then Eq (2) is solved not by one single value of E but by a whole range. Not only do we have  $d^2S/dE^2=0$ , as at an ordinary critical point, but all higher order derivatives also are zero. Thus, the results in Fig 3 suggest that the joint distribution of activity across neurons in this network is poised at a very unusual critical point.

Figure 3 is telling us that the tradeoff between the probability and numerosity of states in the network takes a very special form. We expect that states of lower probability (e.g., those in which more cells spike) are more numerous (because there are more ways to arrange K spikes among N cells as K increases from very low values). But the usual result is that this tradeoff—which is precisely the tradeoff between energy and entropy in thermodynamics—selects "typical" states that all have roughly the same probability. The statement that S(E) = E, as suggested in Fig 3, is the statement that states which are ten times less probable are exactly ten times more numerous, and so there is no typical value

of the probability. This balancing of probability and numerosity extends over a range of at least 0.2 along the E/N axis, corresponding to a factor of  $\sim e^{(0.2)N} \sim 10^{10}$  in probability for networks of N=120 neurons.

#### VI. HEAT CAPACITIES

The divergence of the specific heat is one of the classical signs of criticality in equilibrium thermodynamic systems. Although the neurons obviously are not an equilibrium system, the model probability distribution in Eqs (3, 4) is mathematically identical to the Boltzmann distribution for a system in equilibrium at temperature  $k_BT = 1$ . Thus we can take this model seriously as a statistical mechanics problem, and compute the specific heat in the usual way; for details see Appendix C. Furthermore, we can change the effective temperature by considering a one parameter family of models,

$$P(\lbrace \sigma_{\rm i} \rbrace; T) = \frac{1}{Z(T)} \exp \left[ -\frac{1}{T} E(\lbrace \sigma_{\rm i} \rbrace) \right], \tag{5}$$

with  $E(\{\sigma_i\})$  as before in Eq (4). The goal is to see whether there is anything special about the value T=1 that describes the real system.

Results for the heat capacity of our model vs T are shown in Fig 4. There is a dramatic peak in this plot, and as we look at larger and larger groups of neurons, the peak grows and moves closer to T=1, which is the model of the actual network. Importantly, the heat capacity grows even when we normalize by N, so that the specific heat, or heat capacity per neuron, is growing with N, as expected at a critical point.

Temperature and energy may seem like foreign concepts in the context of real neurons, so the results of Fig 4 require some interpretation. We have studied one particular network, that is described by some set of parameters. Are the parameters that describe the real network in any sense special? Changing "temperature" allows us to probe along one axis in parameter space, and the fact that we observe a peak in the specific heat means that the real network in poised in parameter space very close to a maximum in the variance of log(probability), which we can think of as the dynamic range of surprise that can be represented by the network.

The temperature is only one axis in parameter space, and along this direction there are variations in both the correlations among neurons and their mean spike rates. As an alternative, we can consider a family of models in which the strength of correlations changes but spike rates are fixed. Such a family is given by

$$P(\{\sigma_{i}\}; \alpha) = \frac{1}{Z(\alpha)} \exp\left[-E_{\alpha}(\{\sigma_{i}\})\right]$$

$$E_{\alpha}(\{\sigma_{i}\}) = -\sum_{i=1}^{N} h'_{i}(\alpha)\sigma_{i}$$
(6)

![](_page_4_Figure_10.jpeg)

FIG. 4: Heat capacity in maximum entropy models of neurons responding to naturalistic stimuli. (A) Heat capacity, C(T), computed for subnetworks of  $N=20,\,40,\,80,\,120$ , neurons. Points are averages, and error bars are standard deviations, across 30 choices of subnetwork at each N. For details of the computations see Appendix C. (B) Zoom-in of the peak of C(T), shown as an intensive quantity, C(T)/N. The peak moves towards T=1 with increasing N, and the growth of C(T) near T=1 is faster than linear with N. Green arrow points to results for a model of independent neurons whose spike probabilities (firing rates) match the data exactly; the heat capacity is exactly extensive.

$$-\alpha \left[ \frac{1}{2} \sum_{i,j=1}^{N} J_{ij} \sigma_{i} \sigma_{j} + V\left(K\right) \right], \quad (7)$$

where changing  $\alpha$  changes the strength of correlations, and we adjust all the  $h'_i(\alpha)$  to hold mean spike rates fixed at their observed values. The behavior of this family of models is explored in Fig 5.

At  $\alpha = 0$ , the model describes a population of independent neurons, so that the distribution of correlation coefficients across the populations consists of a singular peak at C = 0 (Fig 5A, left panel). As  $\alpha$  is increased beyond the strength of correlations present in the data ( $\alpha = 1$ , Fig 5A, middle panel), the distribution of correlations broadens such that at  $\alpha = 2$  some pairs are very strongly correlated (Fig 5A, right panel). This is reflected in a distribution of frequent joint activity states that seem to cluster around a small number of prototypical patterns, much as in the Hopfield model of associative memories [4, 6]. The entropy vs energy plot for the corresponding network, shown in Fig 5B, singles out the ensemble at  $\alpha = 1$  as the one whose density of states aligns most closely with the diagonal: going towards independence (smaller  $\alpha$ ) gives rise to a concave bump at low energies, while  $\alpha > 1$  ensembles deviate away from the equality line more at high energies. Interestingly, the drop in entropy from independence to  $\alpha = 1$  is modest, consistent with our previous reports [12], and the entropy decreases substantially only at high  $\alpha$  and large N, as shown in Fig 5C. Corresponding to the variations in S(E) shown in Fig 5B, we see in Fig 5D that there is a peak in the specific heat of the model ensemble near  $\alpha = 1$ . As we look at larger and larger networks, this peak rises and moves toward  $\alpha = 1$ , which describes the real system.

## VII. COULDN'T IT JUST BE ... ?

In equilibrium thermodynamics, poising a system at a critical point involves careful adjustment of temperature, pressure, and other parameters. Finding that the retina seems to have poised itself near criticality should thus be treated with some skepticism. Here we consider some of the pitfalls that might mislead us into thinking that the system is critical when it is not; see also Appendix D.

Part of our analysis is based on the use of maximum entropy models, and one could worry that the inference of these models is unreliable for finite data sets [40, 41]. Expanding on the discussion in Ref [12], we can show that signatures of criticality, such as the peak in specific heat, are not the result of over fitting to a limited data set. As an example, we see a clear peak in the specific heat when we learn models for N=100 neurons from even one—tenth of our data, and the variance across fractions of the data is only a few percent (Appendix D 1, Fig 6).

While the inference of maximum entropy models might be accurate, some less interesting models might mimic

![](_page_5_Figure_6.jpeg)

FIG. 5: Changing correlations at fixed spike rates. (A) Three maximum entropy models for a 120-neuron network, where correlations have been eliminated (left,  $\alpha = 0$ ), left at the strength found in data (middle,  $\alpha = 1$ ), or scaled up (right,  $\alpha = 2$ ). The first row shows 10000 most frequent patterns (black=spike, white=silence) in each model. The second row shows the distribution of pairwise correlation coefficients. (B) Entropy vs. energy for the networks in A. (C) Entropy per neuron as a function of  $\alpha$ , for different subnetwork sizes N. (D) Heat capacity per neuron exhibits a peak close to  $\alpha = 1$ . Error bars are standard deviations over 10 subnetworks for each N and  $\alpha$ .

the signatures of criticality [42]. In particular, it has been suggested that independent neurons with a broad distribution of spike rates could generate a distribution of N-neuron activity patterns  $\{\sigma_i\}$  that mimics some aspects of critical behavior [43]. But in an independent model built from the actual spike rates of the neurons, the probability of seeing the same state twice would be less than one part in a billion, dramatically inconsistent with the measured  $P_c \sim 0.04$ . Such independent models also cannot account for the faster than linear growth of the heat capacity with N (Fig 4), which is an essential feature of the data and its support for criticality.

In maximum entropy models, the probability distribution over patterns of neural activity is described in terms of "interactions" between neurons, such as the terms  $J_{ii}$ in Eq (4); an alternative view is that the correlations result from the response of the neurons to fluctuating external signals. Testing this idea has a difficulty that has nothing to do with neurons. Going back to equilibrium statistical mechanics, many models in which spins (or other degrees of freedom) interact with one another are mathematically equivalent to a collection of spins responding independently to a fluctuating field (Appendix D 3). Thus, correlations are almost always interpretable as independent responses to unmeasured fluctuations, and for neurons there are many possibilities, including internal states of neurons, that would not fall into the usual classification of network effects vs common input.

We can exclude the visual stimulus alone as the explanation of what we see, since similar behavior—e.g., the growth of a peak in specific heat with increasing N appears in response to random checkerboard movies or even full field flicker (Appendix D 2, Fig 7). Our analysis focused on the total distribution over activity states, not distinguishing between the correlations due to a common stimulus input and those due to intrinsic circuit interactions. While a network of conditionally independent neurons might give rise to signatures of criticality as observed here, such models are incompatible with our data. We can construct conditionally independent neurons by shuffling the repeated presentations of the same movie, and in this surrogate data the probability of the whole network remaining silent changes slightly, but this change is ten times larger than the standard error in the measurement. Conditional independence thus is excluded, based just on this one statistic, with a likelihood ratio of more than ten thousand (Appendix D 2, Fig 8). Apart from these arguments, experiments show that electrically evoked spikes in retinal ganglion cells feed back into the retinal circuitry via electrical synapses and modulate the response of other ganglion cells to visual input [44]. This is a direct demonstration that retinal ganglion cells are not conditionally independent encoders of their visual input, as are the more common measurements of "noise correlations" between ganglion cells [45–47].

In interacting models that are equivalent to a collection of independent spins responding to a fluctuating field, criticality usually requires that the distribution of fluctuations be very special, e.g. with the variance tuned to a particular value. In this sense, saying that correlations result from fluctuating inputs wouldn't really "explain" our observations. Recently, it has been suggested that sufficiently broad distributions of fluctuations might lead more generically to critical phenomenology [48], but this argument requires that the number of neurons be much larger than the number of independently fluctuating fields. If the relevant fields are generated by retinal neurons that provide input to the ganglion cells, then changing the effective dimensionality of the stimulus—from full field flicker to natural movies to random checkerboards—would change the number of effective fields, and at one extreme this number is comparable to or larger than the number of neurons from which we record. Nonetheless we see near-critical behavior in all these cases. It is also not clear how such models with fluctuating fields would explain the fact that networks with slightly stronger or weaker correlations among neurons both deviate from criticality (Fig 5).

The evidence for criticality that we find in the analysis of 160 neurons is consistent with extrapolations from the analysis of smaller populations under similar conditions [16, 19]. The basis for those predictions was the assumption that the smaller populations were typical of the larger one, i.e. that the spike probabilities and pairwise correlations that we could observe were drawn from the same distribution as in the full system, and that these distributions (rather than the detailed structure of the correlation matrix, for example) were sufficient to determine the thermodynamic behavior [49]. Hints of criticality also are observable in vastly simpler models, which capture the distribution of summed activity in the network but ignore the identity of individual neurons [23].

## VIII. DISCUSSION

The traditional formulation of the neural coding problem makes an analogy to a dictionary, asking for the "meaning" of each neural response in terms of events in the outside world [50]. But before we can build a dictionary, we need to know the lexicon, and for large populations of neurons this already is a difficult problem: what is the set of responses that the network actually uses? With 160 neurons, the number of possible responses is larger than the number of words in the vocabulary of a well educated English speaker, and is more comparable to the number of possible short phrases or sentences. In passing from letters to words to sentences, we encounter many features of language that leave an imprint on the probability distribution: the joint distribution of letters in words embodies spelling rules [30], while the joint distribution of words in sentences encodes aspects of grammar [51] and semantic categories [52]. By analogy, the joint distribution of activity among neurons should reveal structures that have biological significance.

The small patch of the retina that we consider has

several types of ganglion cells, and indeed no two cells have truly identical input/output characteristics, even in response to spatial uniform stimuli [53]. Nonetheless, groups of twenty or more cells have collective properties that vary very little across different groups of neurons. If we ask not about the probability of each detailed pattern of spiking and silence, but rather count how many combinations have a given probability, this relationship is highly reproducible from group to group, and simplifies as we look at larger and larger groups.

The relationship between probability and numerosity of states is mathematically identical to the relationship between energy and entropy in statistical physics. The fact that this relationship simplifies as we look at larger groups of neurons suggests that we are seeing signs of a thermodynamic limit. As emphasized above, the existence of such a limit in a real network is not guaranteed.

If we can identify the thermodynamic limit, we can try to place the network in a phase diagram of possible networks. Crucial to our analysis is that the critical surfaces which separate different phases have signatures that are detectable even if we don't know the nature of the phases themselves. In particular, criticality is associated with a finely balanced tradeoff between probability and numerosity: states that are a factor F times less probable are also a factor F times more numerous. At conventional critical points, this balancing occurs only in a small neighborhood of the typical probability, but in the network of retinal ganglion cells we see nearly perfect balancing across a wide range of probabilities (Fig 3). If we construct model networks that have slightly stronger or weaker correlations among pairs of neurons, then in both cases we see a breakdown of this balance (Fig 5).

The strength of correlations depends on the structure of visual inputs, on the connectivity of the neural circuit, and on the state of adaptation in the system. The fact that we see signatures of criticality in response to very different visual inputs (Appendix D 2), but that these signatures break down in model networks with stronger or weaker correlations, strongly suggests that adaptation is tuning the system toward a critical state. This predicts that a sudden change of visual input statistics will drive the network to a non–critical state, and that during the course of adaptation the distribution of activity will relax back to the critical surface. This can be tested directly.

Does criticality have functional consequences? The crucial signature of criticality that we have seen in the data is the extreme inhomogeneity of the probability distribution over states. This allows the system to construct an instantaneously readable code for events in the world that have a large dynamic range of likelihoods or surprise, and this may be well suited to the challenges of the natural environment; it is not, however, an efficient code in the usual sense. Systems near critical points are maximally responsive to certain external signals, and this sensitivity may be functionally useful. Most of the systems that exhibit criticality in the thermodynamic sense also exhibit a wide range of time scales in their dynam-

ics, so that criticality may provide a general strategy for neural systems to bridge the gap between the microscopic time scale of spikes and the macroscopic time scales of behavior. Critical states are extremal in all these different senses, and more; in the examples we understand, these different features also are not separable, so it may be difficult to decide which is relevant for the organism.

In addition to the the network of neurons studied here, closely related signatures of criticality have been detected in ensembles of amino acid sequences for protein families [31], in flocks of birds [36] and swarms of insects [54], and in the network of genes controlling morphogenesis in the early fly embryo [55]; there is also evidence that cell membranes have lipid compositions tuned to a true thermodynamic critical point [56]. Different, dynamical notions of criticality have been explored in neural [57, 58] and genetic [59, 60] networks, as well as in the active mechanics of the inner ear [61–63]. This convergence may hint at a general principle, but there is considerable room for skepticism [38, 64]. A new generation of experiments on larger populations of neurons should allow for decisive tests of our ideas about collective behavior in these networks, including the possibility of criticality.

## Acknowledgments

We thank A Cavagna, I Giardina, M Ioffe, SCF van Opheusden, SE Palmer, E Schneidman, DJ Schwab, and AM Walczak for helpful discussions. Research supported in part by NSF Grants PHY-1305525 and CCF-0939370, by NIH Grant R01 EY14196, and by Austrian Science Foundation Grant FWF P25651. Additional support provided by the Fannie and John Hertz Foundation, by the Swartz Foundation, and by the WM Keck Foundation.

# Appendix A: Thermodynamics and probability distributions

The fundamental variables of thermodynamics are energy, temperature, and entropy. For the states taken on by a network of neurons, energy and temperature are meaningless, so it is difficult to see how we can construct a thermodynamics for these systems. But in statistical mechanics, all thermodynamic quantities are derivable from the Boltzmann distribution, the probability that the system will be found in any particular state. Thus, all thermodynamic statements can be seen as statements about this underlying probability distribution, and in this sense we should be able to construct thermodynamics for any probability distribution that describes a large number of variables.

The idea that all probability distributions over N variables have an associated thermodynamics in the  $N \to \infty$  limit is powerful, but perhaps not so widely used. This connection is well studied by mathematical physicists [8],

and has been a guide to the analysis of experiments on dynamical systems [9, 10]. We have used these ideas to construct a thermodynamics of natural images [65], and have emphasized the connection of thermodynamic criticality to Zipf's law [38]. Here we give a somewhat pedagogical discussion, in the hope of making the results accessible to a broader audience.

We start by recalling that, for a system in thermal equilibrium at temperature T, the probability of finding the system in state s is given by

$$P_s = \frac{1}{Z} e^{-E_s/k_B T},\tag{A1}$$

where  $E_s$  is the energy of the state, and Boltzmann's constant  $k_B$  converts between conventional units of temperature and energy. The partition function Z serves to normalize the distribution, which requires

$$Z = \sum_{s} e^{-E_s/k_B T}, \tag{A2}$$

but in fact this normalization constant encodes many physical quantities. The logarithm of the partition function is proportional to the free energy of the system, the derivative of the free energy with respect to the volume occupied by the system is the pressure, the derivative with respect to the strength of an applied magnetic field is the magnetization, and so on.

The "state" of a system is defined by the joint configuration of all its parts. Thus in a classical gas or liquid, s is defined by the positions and velocities of all the constituent atoms. Different gases or liquids differ not because these variable are different, but because the energy  $E_s$  is a different function of these N underlying variables. But thermodynamics doesn't make reference to all these details. Which aspects of the underlying microscopic rules actually matter for predicting the free energy and its derivatives?

We can write the sum over all states as a sum first over states that have the same energy, and then a sum over energies. We do this by introducing an integral over a delta function into the sum:

$$Z = \sum_{s} e^{-E_s/k_B T}$$

$$= \sum_{s} \left[ \int dE \, \delta(E - E_s) \right] e^{-E_s/k_B T} \qquad (A3)$$

$$= \int dE \sum_{s} \delta(E - E_s) e^{-E_s/k_B T} \qquad (A4)$$

$$= \int dE \, e^{-E/k_B T} \left[ \sum_{s} \delta(E - E_s) \right]. \qquad (A5)$$

We see that the way in which the energy depends on each state appears only in the brackets, a function n(E) that counts how many states have a particular energy.

Looking ahead to the analysis of real data, it will be convenient to rearrange Eq (A5) slightly. Instead of counting the number of states that have energy E, we can count the number of states with energy less than E:

$$\mathcal{N}(E) = \sum_{s} \Theta(E - E_s), \qquad (A6)$$

where the step function is defined by

$$\Theta\left(x > 0\right) = 1 \tag{A7}$$

$$\Theta\left(x<0\right) = 0. \tag{A8}$$

But the step function is the integral of the delta function, which means that we can integrate by parts in Eq (A5) to give

$$Z = \frac{1}{k_B T} \int dE \, e^{-E/k_B T} \mathcal{N}(E). \tag{A9}$$

If we think about N variables, each of which can take on only two states, the total number of states is  $2^N$ . More generally, we expect that the number of possible states in a system with N variables is exponentially large, so it is natural to think not about the number of states  $\mathcal{N}(E)$  but about its logarithm,

$$S(E) = \ln \mathcal{N}(E),\tag{A10}$$

which is called the entropy.

As a technical aside, we can define the entropy either in terms of the number of states with energy close to E, what we have called n(E) in the main text, or we can use the number of states with energy less than E, what we have called  $\mathcal{N}(E)$ . In the limit that the number of degrees of freedom in the system become large, there is no difference in the resulting estimate of the entropy per degree of freedom, because the number of states is growing exponentially fast with the energy, so that the vast majority of states with energy less than E also have energy very close to E. When it comes time to analyze experimental data, however, using  $\mathcal{N}(E)$  allows us to avoid making bins along the E axis.

Substituting from Eq (A10) into Eq (A9), the partition function can be written as an integral determined only

the function S(E), entropy vs energy:

$$Z = \frac{1}{k_B T} \int dE \, \exp\left[-\frac{E}{k_B T} + S(E)\right]. \tag{A11}$$

One of the key ideas in thermodynamics is that certain variables are "extensive," that is proportional to the number of particles or variables in the system, while other variables are "intensive," independent of the system size. Temperature is an intensive variable, energy and entropy are extensive variables. It is then natural to think about the energy per particle  $\epsilon = E/N$ , and the entropy per particle,  $S(E)/N = s(\epsilon)$ . In the limit of large N, we expect  $s(\epsilon)$  to become a smooth function. Substituting into Eq (A11), the partition function can be written as

$$Z = \frac{N}{k_B T} \int d\epsilon \, e^{-Nf(\epsilon)/k_B T}$$
 (A12)

$$f(\epsilon) = \epsilon - k_B T s(\epsilon). \tag{A13}$$

We note that  $f(\epsilon)$  is the difference between energy and entropy, scaled by the temperature, and is called the free energy.

Whenever we have an integral of the form in Eq (A12), at large N we expect that it will be dominated by values of  $\epsilon$  close to the minimum of  $f(\epsilon)$ . This minimum  $\epsilon_*$  is the solution to the equation

$$\frac{df(\epsilon)}{d\epsilon} = 0 \Rightarrow \frac{1}{k_B T} = \frac{ds(\epsilon)}{d\epsilon},$$
 (A14)

which we can also think of as defining the temperature. Notice that T being positive requires that the system have  $ds(\epsilon)/d\epsilon > 0$ , which means there are more states with higher energies.

If we expand  $f(\epsilon)$  in the neighborhood of  $\epsilon_*$ , we have

$$f(\epsilon) = f(\epsilon_*) - \frac{k_B T}{2} \frac{d^2 s(\epsilon)}{d\epsilon^2} \bigg|_{\epsilon_*} (\epsilon - \epsilon_*)^2 + \cdots, \quad (A15)$$

which gives

$$Z = \frac{N}{k_B T} e^{-f(\epsilon_*)/k_B T} \int d\epsilon \exp\left(-\frac{N}{2} \left[ -\frac{d^2 s(\epsilon)}{d\epsilon^2} \Big|_{\epsilon_*} \right] (\epsilon - \epsilon_*)^2 + \cdots \right). \tag{A16}$$

This looks as if the energy per particle is drawn from a Gaussian distribution, with mean  $\epsilon_*$  and variance

$$\langle (\delta \epsilon)^2 \rangle = \frac{1}{N} \left[ -\frac{d^2 s(\epsilon)}{d\epsilon^2} \Big|_{\epsilon_*} \right]^{-1},$$
 (A17)

and indeed this can be shown more directly from the

Boltzmann distribution.

With the interpretation of  $\epsilon_*$  as the mean energy per particle, we can use Eq (A14) to calculate how this energy changes when we change the temperature, and we find

$$\frac{d\epsilon_*}{dT} = \frac{1}{k_B T^2} \left[ -\frac{d^2 s(\epsilon)}{d\epsilon^2} \Big|_{\epsilon_*} \right]^{-1}.$$
 (A18)

The change in energy with temperature is called the heat capacity C, and when we normalize per particle it is referred to as the specific heat. Combining Eqs (A17) and (A18), we see that the specific heat C/N is connected to the variance in energies,

$$\langle (\delta \epsilon)^2 \rangle = k_B T^2 \frac{C}{N}.$$
 (A19)

This relationship also can be proven without resorting to the approximation in Eq (A15).

Our discussion thus far assumes that the second derivative of the entropy with respect to the energy is not zero. If we take all our results at face value, then when  $d^2s/d\epsilon^2 \to 0$ , the specific heat will become infinite [Eq (A18)], as will the variance of the energy per particle [Eq (A17)]. This is a critical point.

There is much more to be said about the analysis of critical points using the entropy vs energy. But our concern here is how these ideas connect to systems that are not in thermal equilibrium, so that temperature and energy are not relevant concepts. What we would like to show is that many of the thermodynamic quantities nonetheless serve to characterize the behavior of *any* probability distribution for a very large number of variables.

Rather than trying to compute the partition function, we can ask, for any distribution, how the normalization condition is satisfied. We still imagine that there are states s, built of of N different variables, as with the patterns of spiking and silence in a network of neurons. Each state s has a probability  $P_s$ , and we must have

$$1 = \sum_{s} P_s. \tag{A20}$$

We can now follow the same strategy that we used above for the partition function: we do the sum first by summing over all the states that have the same value of the (log) probability, and then we sum over this value. We start by defining

$$E_s = -\ln\left(\frac{P_s}{P_0}\right),\tag{A21}$$

where  $P_0$  is the probability of the most likely state, as in Eq (1). Then we have

$$\sum_{s} P_{s} = \sum_{s} \int dE \, \delta(E - E_{s}) P_{s}. \tag{A22}$$

But since  $P_s = P_0 e^{-E_s}$ , we can rewrite this as

$$\sum_{s} P_s = P_0 \int dE \, e^{-E} \sum_{s} \delta(E - E_s). \tag{A23}$$

Integrating by parts, we obtain

$$\sum P_s = P_0 \int dE \, e^{-E} \mathcal{N}(E), \tag{A24}$$

where  $\mathcal{N}(E)$  is a cumulative density of states, as in Eq (A6),

$$\mathcal{N}(E) = \sum_{s} \Theta(E - E_s). \tag{A25}$$

Again, this is a number of states, so the logarithm of this number is an entropy, exactly as in Eq (A10). Thus the statement that the probability distribution is normalized becomes

$$\sum_{s} P_s = P_0 \int dE \, \exp\left[-E + S(E)\right]. \tag{A26}$$

If we have system in which the state s is built out of N variables, then we expect that, for large N, both  $\log$ probabilities (E) and entropies (S) are proportional to N. A standard example is in information theory, where s could label a message built out of N symbols, and the proportionality  $E \propto N$  is central to proofs of the classic coding theorems [66]. In the case of interest to us here, we can look at the states taken on by groups of N neurons, and we can vary N over some range. The function  $\mathcal{N}(E)$ , and hence the entropy S(E), is a property of a single system with a particular value of N, and to remind us of this fact we can write  $S_N(E)$ . It is an experimental question what happens as N become large. But, in many of the examples we understand—from statistical physics, from information theory, and indeed from more general examples in probability theory—we find that there is a well defined limiting behavior at large N, which means that there is a function

$$s(\epsilon) = \lim_{N \to \infty} \frac{1}{N} S_N(E = N\epsilon).$$
 (A27)

If this limit exists, then the normalization condition on the probability distribution in Eq (A26) becomes

$$\sum_{s} P_s \to NP_0 \int d\epsilon \, e^{-Nf(\epsilon)}, \tag{A28}$$

$$f(\epsilon) = \epsilon - s(\epsilon).$$
 (A29)

Now we can see the correspondence with the description of an equilibrium thermodynamic system, which leads up to the expressions for the partition function in Eqs (A12) and (A13):

- 1. We can assign an "energy" to every state of the system, which is just the negative log probability. It is convenient to normalize so that the most likely state has zero energy. The "effective temperature" of the system is  $k_BT=1$ .
- 2. We can count the number of states below a given energy, and the log of this number is an entropy.
- 3. If there are N elements (e.g., neurons) in our system, it is natural to ask about the entropy per element as a function of the energy per element. If this function has a smooth limit as N becomes large,  $s(\epsilon)$ , then we can define a thermodynamics for the system.

- 4. When we sum over states, the sum is dominated by states that minimize the free energy,  $f(\epsilon) = \epsilon s(\epsilon)$ , just as in ordinary thermodynamics, provided that the curvature of the free energy at this minimum is nonzero.
- 5. The dominance of states near at the minimum of the free energy enforces the notion of "typicality" [66], so that at large N most of the states we actually see have essentially the same value of log probability.
- 6. If the curvature at the minimum of the free energy vanishes, then the usual ideas of typicality break down, and we will see large fluctuations in the log probability of states, even if we normalize this log probability by N.
- 7. The large variance in log probability is mathematically equivalent to a diverging specific heat in the thermodynamic case. This is a signature of a critical point.

Before leaving this discussion, we should note that there are other signatures of criticality, and even different notions of criticality. In equilibrium systems with interactions that extend only over short distances, correlations typically extend over some longer but finite distance  $\xi$ ; at the critical point this correlation diverges, so that there is no characteristic length scale—all scales between the size of the constituent particles and the size of the system as a whole are relevant [67]. Not only does the specific heat diverge at the critical point, but so does the susceptibility to external fields. All of these diverging quantities have a power-law dependence on the difference between the actual temperature and the critical temperature, and the exponents of these power-laws are quantitatively universal: many different systems, with different microscopic constituents, exhibit precisely the same exponents, and in a certain precise sense these exponents give a complete description of the system in the neighborhood of the critical point [2, 68]. In the study of complex, non-equilibrium systems, scale invariance and power-law behaviors often are taken as signs of criticality, but seldom is it possible to exhibit these behaviors over the wide range of scales that are the standard in studies of equilibrium critical phenomena, so one must be cautious.

In almost all equilibrium systems, the approach to criticality also is associated with the emergence of long time scales in the dynamics: as with the divergence of the correlation length  $\xi$ , the divergence of the correlation time in the dynamics typically means that there is a form of temporal scale invariance at criticality. Deterministic dvnamical systems also exhibit critical phenomena, often called bifurcations, where the system's behavior changes qualitatively in response to an infinitesimal change in parameters [69]. These phenomena are easiest to understand when the number of degrees of freedom N is small, but then the sharp bifurcations are rounded if there is noise in the system; the example of equilibrium statistical mechanics shows how noisy dynamical systems can recover sharp transitions in the limit of large N. In general it is not clear how dynamical and statistical notions of criticality are related to one another in systems with many degrees of freedom.

#### Appendix B: Experimental methods

Much of the analysis in this paper is based on the same data set as in Ref [12]. For completeness we review our experimental methods here.

Experiments were performed on the larval tiger salamander, Ambystoma tigrinum tigrinum, in accordance with institutional animal care standards. Retinae were isolated from the eye in darkness, and the retina was pressed against a custom fabricated array of 252 electrodes. The retina was superfused with oxygenated Ringer's medium (95% O<sub>2</sub>, 5% CO<sub>2</sub>) at room temperature. Electrode voltage signals were acquired and digitized at 10 kHz by a 252 channel preamplifier (Multi-Channel Systems, Germany). The sorting of these signals into action potentials from individual neurons was done offline using the methods of Ref [13].

The repeated natural movie was a movie of a fish tank captured at 30 Hz with a standard camera; it lasted 20 s, and was repeated 297 times. As noted in the text, this experiment allowed us to resolve 160 neurons across the recording array. The random checkerboard consisted of square pixels, 69  $\mu$ m on a side, each chosen independently black or white 30 times per second, creating a 30 second random movie that was repeated 69 times; this experiment yielded 120 stable, resolved cells. For the spatially uniform flicker, the luminance of the entire screen was chosen randomly from a Gaussian distribution 60 times per second, creating a ten second long random sequence that was repeated 98 times; we separated the signals from 111 neurons.

# Appendix C: Density of states and heat capacity in maximum entropy models

We can take our maximum entropy model seriously as a statistical mechanics problem and use Monte Carlo simulation to generates samples of the states  $\{\sigma_i\}$  drawn from our model distribution. Heat capacity curves were estimated by running a Metropolis Monte Carlo sampler independently at every T. Since the model assigns an energy  $E = \mathcal{H}(\{\sigma_i\})$  to each state, we can compute the mean and variance of E from a single long Monte Carlo run, and thus estimate the heat capacity through the thermodynamic identity in Eq (A19). Samples of the energy were collected at every sweep (roughly N spin flips);  $2 \times 10^6$  sweeps were performed for every T.

To estimate the function n(E) in the maximum entropy models, including the  $\alpha$ -ensembles of Fig 5, we used Wang-Landau sampling [39]. In detail, the complete energy range was divided into  $2 \times 10^4$  equidistant energy bins  $(6 \times 10^3$  for the  $\alpha$ -ensembles), the histogram flatness criterion was 0.9, and the final multiplicative update  $1+10^{-5}$ . These measurements, as well as the specific heat curves, can both be used to give an estimate of the entropy of the distribution, and these agree to within better than one percent [12], providing a check on our

sampling procedures.

For more on these matters, see the methods section "Computing the entropy ..." of Ref [12].

#### Appendix D: More about alternatives

In this section we expand on alternative interpretations of the data, arguing that the signatures of criticality are unlikely to be explained away as spurious consequences of less interesting models.

# 1. Impact of limited data

We have tested in detail the reliability with which maximum entropy models can be inferred from the available data. As explained in Ref [12], we can learn these models from 90% of the data, and then compare the quality of the model against both the training set and the held out test set. Even with N=120 neurons, the model predicts that the log-likelihood of the test data is the same as that of the training data, within error bars, and these errors are less than 1% (Fig 4 of Ref [12]). Still, one could worry that small errors associated with the finiteness of the data set could have a disproportionate impact on the putative signatures of criticality. To test for this, we have learned models for N = 100 neurons from fractions of the data ranging down to just 10%; results for the heat capacity vs temperature (as in Fig 4) are shown in Fig 6. We see that the sharp peak in C(T) is essentially independent of the sample size across this wide range, and that the variations in C(T) across different small fractions of the data are only a few percent. Thus, this behavior is not a result of over fitting, nor is it linked in any way to the size of our data set.

It is important that, in Fig 6, we are always looking at the same 100 neurons, else variability across subsets could be confused with sampling errors. When we change the size of the data we are choosing at random some fraction of the experiment, and for each fraction we examine 10 such random choices. For each choice we make a completely independent reconstruction of the maximum entropy model, which means that variability includes not just the effects of finite data but also any errors in parameter estimation or in the Monte Carlo estimate of the specific heat. Evidently all of these errors are quite small.

# 2. Are correlations inherited from the visual stimulus?

As discussed in the main text, one possible interpretation of our observations is that correlations among neurons simply reflect correlations in the visual stimulus. In this case, any interesting features in the joint distribution of activity among many neurons would be entirely traceable to the structure of the sensory inputs.

![](_page_11_Figure_10.jpeg)

FIG. 6: Heat capacity in maximum entropy models learned from limited data. At left, C(T) for models learned from different fractions f of the data. At right, we zoom in on the peak near T=1; error bars show standard deviations across different randomly chosen fractions of the data.

The idea that correlations among neurons should be decomposed into contributions from their inputs and contributions intrinsic to the circuit is very old [70], dating back to a time when it was hoped that measurement of correlations would allow a direct inference of connectivity in the circuit. Before discussing the origin of correlations, it is important to emphasize that the distinction between "stimulus induced" and "intrinsic" correlations is not a distinction that the brain can make. Experimentally, we make this distinction by providing exact repetitions of the stimulus, but this never happens in the natural world, and the only knowledge that that brain has of its visual inputs is the set of signals provided by the population of ganglion cells itself, so there is no way to search for correlations with some other reference signal.

While the dissection of the correlations is irrelevant for brain function, it is interesting to ask, mechanistically, how these correlations arise. If they arise solely from the visual inputs, then changing the statistical structure of these inputs should produce a dramatic effect. We have replaced the natural movies with random flickering checkerboards (an approximation to spatiotemporal white noise) and spatially uniform but temporally random flicker (Appendix B). In each case we have constructed maximum entropy models [Eqs (3) and (4)] and searched for a peak in the specific heat vs temperature, as in Fig 4; results are shown in Fig 7.

Although there are quantitative differences among the responses to the different stimulus ensembles, we see that there are signatures of criticality in each case. As with the natural movies, there is a peak in the specific heat, the height of the peak grows with the number of neurons, and the location of the peak moves toward T=1 at larger N. It thus seems unlikely that these signatures of criticality in the specific heat are merely a reflection of input statistics. Indeed, we should remember that the decomposition of correlations into intrinsic and stimulus—induced is incomplete, because the retina adapts to the distribution of its inputs, on many time scales. It would appear that some combination of anatomical connectivity and adaptation poises the population of retinal ganglion

![](_page_12_Figure_1.jpeg)

FIG. 7: Specific heat in different stimulus ensembles. (A) Random checkerboard stimuli, averaged over five subnetworks at each N. (B) Full field flicker stimuli, averaged over 30 subnetworks at each N. In both panels, arrows indicate the peak of heat capacity for a matched network of independent neurons; error bars not shown for clarity. (C) The approach of the peak of specific heat,  $T^*$ , to T=1 (model reconstructions), for fish movie (black, from Fig 4), full field flicker (brown), and checkerboard (violet). (D) The growth of specific heat with N for the same stimulus ensembles.

cells near a peak in the specific heat. This points toward future experiments that should probe more directly the invariance of thermodynamic behavior across adaptation states [71].

We can test the picture of stimulus-driven correlations more directly by creating surrogate data in which the neurons respond independently to the visual stimulus. Since the visual stimuli consist of many repetitions of a short movie clip, we can do this by permuting the labels on the repetitions independently for each neuron. We can compute many statistical properties of these surrogate data, but our previous analysis [12] emphasized that the probability of observing complete silence in the population is a very sensitive test of our models. As shown in Fig 8, the probability of silence in the surrogate data differs from that in the real data by a small amount, but this difference is more than ten times the standard errors in our estimates:  $P_{\rm silence} = 0.2101 \pm 0.0009$  for the real data vs  $P_{\rm silence} = 0.2000 \pm 0.0009$  for the surrogate data. To make this more precise, we look at the distribution of our estimates over many randomly chosen halves of the data, and find that even if choose  $10^5$  times we never find the probability of silence in the surrogate data to be as large as in the real data. Thus we can reject the conditionally independent model with high confidence.

### 3. More general hidden variable models

The idea that correlations among neurons might be inherited from the visual stimulus is one possibility among many. More generally we might ask if the pattern of correlations could be understood as the independent response of neurons to *some* signal that is effectively external to the network, or at least hidden from an observer that sees only the patterns of spikes and silence. To assess this possibility it is useful to step back and think about well known models in statistical mechanics.

Consider the mean–field Ising ferromagnet, in which spins  $\sigma_i = \pm 1$  experience an effective magnetic field that is proportional to the average over all the other spins in the system, so that

$$E(\{\sigma_{i}\}) = -\frac{J}{2N} \sum_{i \neq j} \sigma_{i} \sigma_{j}.$$
 (D1)

Note that the sum is over all pairs, and the factor of N insures that the energy of the system is proportional to N. The sum over all distinct pairs is missing the term i=j, but since  $\sigma_i^2=1$  we have

$$E(\lbrace \sigma_{i} \rbrace) = -\frac{J}{2N} \sum_{i,j} \sigma_{i} \sigma_{j} + \frac{J}{2}$$
 (D2)

![](_page_12_Figure_11.jpeg)

FIG. 8: The probability of silence. We compute the probability of the silent state in the full population of N=160 neurons, using randomly chosen halves of the data. Mean and standard deviation as point with error bars, blue for the real data and green for surrogate data in which trials are shuffled to give conditionally independent neurons. Blue (green) curve shows the probability of finding, in half of the data, a fraction of silent bins less than (more than) a given value for the real data (conditionally independent neurons). The two distributions are non-overlapping, to better than  $p=10^{-4}$ .

$$= -\frac{J}{2N} \left( \sum_{i} \sigma_{i} \right)^{2} + \frac{J}{2}.$$
 (D3)

Thus when we compute the partition function, we have (choosing units where  $k_BT=1$ )

$$Z \equiv \sum_{\{\sigma_i\}} e^{-E(\{\sigma_i\})/k_B T} \tag{D4}$$

$$= \sum_{\{\sigma_{i}\}} \exp \left[ \frac{J}{2N} \left( \sum_{i} \sigma_{i} \right)^{2} - \frac{J}{2} \right]$$
 (D5)

But we can always write

$$\exp\left[\frac{A}{2}x^2\right] = \int \frac{dh}{\sqrt{2\pi A}} \exp\left[-\frac{1}{2A}h^2 + hx\right]. \quad (D6)$$

Applying this identity to Eq (D5), we have

$$Z = e^{-J/2} \int \frac{dh}{\sqrt{2\pi N/J}} \sum_{\{\sigma_i\}} \exp\left[-\frac{N}{2J}h^2 + h\sum_i \sigma_i\right].$$
(D7)

We can think of this, more suggestively, as

$$Z = e^{-J/2} \int dh P(h) \sum_{\{\sigma_i\}} \exp \left[ h \sum_i \sigma_i \right]$$
 (D8)

$$= e^{-J/2} \int dh \, P(h) \sum_{\{\sigma_i\}} e^{-E(\{\sigma_i\};h)}, \quad (D9)$$

where P(h) is a Gaussian probability distribution for h, with zero mean and variance J/N. But now the sum over states involves a model in which each spin  $\sigma_i$  experiences a magnetic field h, so that the total energy

$$E(\{\sigma_{i}\};h) = -h\sum_{i}\sigma_{i}.$$
 (D10)

Thus a model in which all spins interact with one another, equally, is *mathematically identical* to a model in which each spin responds independently to a magnetic field chosen at random from a Gaussian distribution.

We can find essentially the same equivalence in a broader class of models, which includes the Hopfield model of associative memories [4]. Consider a collection of spins that interact through some matrix  $J_{ij}$ , so that the energy

$$E(\{\sigma_{i}\}) = \frac{1}{2} \sum_{i,j} J_{ij} \sigma_{i} \sigma_{j}. \tag{D11}$$

The Hopfield model corresponds to the choice

$$J_{ij} = -\frac{J}{N} \sum_{\mu=1}^{K} \xi_i^{\mu} \xi_j^{\mu},$$
 (D12)

where there are K "stored memories"

$$\xi^{\mu} \equiv \{\xi_1^{\mu}, \, \xi_2^{\mu}, \, \cdots, \, \xi_N^{\mu}\}.$$
 (D13)

In this case, the same arguments that lead to Eqs (D9) and (D10) now give

$$Z \propto \left\langle \sum_{\{\sigma_i\}} e^{-E(\{\sigma_i\};\{h_i\})} \right\rangle,$$
 (D14)

where the energy corresponds to each spin responding independently to a magnetic field,

$$E(\lbrace \sigma_{i} \rbrace; \lbrace h_{i} \rbrace) = -\sum_{i} h_{i} \sigma_{i}, \qquad (D15)$$

and the magnetic fields are

$$h_{\rm i} = \sum_{\mu=1}^{K} \phi_{\mu} \xi_{\rm i}^{\mu},$$
 (D16)

where each  $\phi_{\mu}$  is a Gaussian random variable with zero mean and variance  $\langle \phi_{\mu}^2 \rangle = J/N$ ; the average in Eq (D14) is an average over these fluctuating fields.

In fact this construction is yet more general. If the energy has the form of Eq (D11), and the matrix  $-J_{ij}$  has all positive eigenvalues, then we can rewrite the partition function as an average over a model of independent spins in magnetic fields, as in Eq (D15), where the fields are Gaussian random variable with zero mean and a covariance matrix  $\langle h_i h_i \rangle = (-J^{-1})_{ij}$ .

The conclusion from these arguments, which are well known, is that a large class of models for interacting spins are mathematically equivalent to models of spins that respond independently to fluctuating magnetic fields. As applied to models for the activity of neurons, this means that large classes of models for correlated activity are *identical* to models of conditionally independent neurons responding to fluctuating inputs.

How do these identities relate to the signatures of criticality? The mean–field model defined by Eq (D1) has a critical point at J=1, which means that in Eq (D9) the variance of the field h must be tuned to exactly 1/N to achieve a critical state. In this sense, saying that the system is equivalent to independent spins experiencing a random field doesn't "explain" anything, it just moves the problem from the distribution of states being poised at a special point to the distribution of fields being poised at some special point.

Recently, Schwab et al [48] have suggested a different view. If we think of the effective magnetic field as arising from a truly external source, then instead of being generated internally, then it is more natural to write

$$P(\lbrace \sigma_{i} \rbrace) = \int dh P(h) P(\lbrace \sigma_{i} \rbrace | h)$$
 (D17)

$$= \int dh P(h)e^{F(h)}e^{-E(\{\sigma_{i}\};h)}, \quad (D18)$$

where the free energy  $F(h) = -N \ln [2 \cosh(h)]$ , and the energy  $E(\{\sigma_i\}; h)$  as in Eq (D10). Note the difference between this equation and Eq (D9): because we take the

field as being truly external, there is an extra factor of the (exponential of) the free energy in the integral, and this is a huge effect, because the free energy is proportional to N. When the fields are generated internally, as a consequence of the interactions, then in the simple case of Eq (D9) we find that the variance of the field is fixed to be J/N, and hence the fields are small. In contrast, in Eq (D18), we are free to imagine any distribution that we might like.

In particular, if the distribution of fields P(h) is sufficiently broad, then the structure of this distribution plays relatively little role in the overall distribution of states. Mathematically, if P(h) is broad then the integral over hcan be approximated for large N as

$$P(\{\sigma_i\}) \propto \frac{P(h = h_*(m))}{\sqrt{N(1 - m^2)}} e^{-Ns(m)}$$
 (D19)  
 $s(m) = -\frac{1 + m}{2} \ln\left(\frac{1 + m}{2}\right)$   
 $-\frac{1 - m}{2} \ln\left(\frac{1 - m}{2}\right)$ , (D20)

where the magnetization  $m = (1/N) \sum_{i} \sigma_{i}$  and  $h_{*}(m)$  is defined by

$$m = \tanh[h_*(m)]. \tag{D21}$$

At large N, Eq (D19) tells us that the energy per neuron associated with any state  $\{\sigma_i\}$  is just  $\epsilon = s(m)$ , depending only on the magnetization m. But then we can count the number of states with energy near  $\epsilon$  by counting the number of states with magnetization m, and this shows that s(m) is also the entropy per neuron, so that  $s(\epsilon) = \epsilon$ , a signature of criticality. Could this, or some generalization, provide an explanation for what we see in the data?

Passing from Eq (D18) to Eq (D19) hinges on the fact that we are integrating a function of one variable that has a factor of N in the exponential. We could generalize to have K different fluctuating fields, as in Eq (D16), but for the large N approximation to be valid we must have  $N \gg K$ . In particular, in the general case where we try to simulate an arbitrary matrix of pairwise interactions. we need N distinct fields, and now large N no longer helps us do the integral. The argument of Ref [48] thus requires that the number of effective fields be small compared to the number of neurons. If we imagine that the unobserved fields are generated by inputs to the retinal ganglion cells from other cells in the retina, then as we change the complexity of the visual stimulus we expect that the effective dimensionality of these inputs also will change. Nonetheless, as described in Appendix D 2, we see signatures of criticality for stimulus ensembles including full field flicker, natural movies, and spatiotemporal white noise.

In models where the neurons do not really interact, but instead respond to a fluctuating field with a large dynamic range, the correlations between neurons have a strength that is set by this dynamic range. But the argument of Ref [48] is that criticality emerges once the dynamic range is large enough, independent of its precise value. The appearance of a critical relation between entropy and energy in this class of models thus is decoupled from the strength of correlations in the network. But Fig 5 shows that, in an accurate model for the distribution of activity patterns, if we change one parameter to modulate the strength of correlations while holding the mean spike probabilities fixed, the entropy/energy relation moves away from its critical form whether we increase or decrease the strength of correlations.

PW Anderson, More is different. Science 177, 393–396 (1972).

<sup>[2]</sup> JP Sethna, Statistical Mechanics: Entropy, Order Parameters, and Complexity (Oxford University Press, Oxford, 2006).

<sup>[3]</sup> EA Guggenheim, The principle of corresponding states. J Chem Phys 13, 253–261 (1945).

<sup>[4]</sup> JJ Hopfield, Neural networks and physical systems with emergent collective computational abilities. *Proc Natl Acad Sci (USA)* **79**, 2554–2558 (1982).

<sup>[5]</sup> DJ Amit, H Gutfreund, and H Sompolinsky, Statistical mechanics of neural networks near saturation. *Ann Phys* **173**, 30–67 (1987).

<sup>[6]</sup> DJ Amit, Modeling Brain Function: The World of Attractor Neural Networks (Cambridge University Press, Cambridge, 1989).

<sup>[7]</sup> J Hertz, A Krogh, and RG Palmer, Introduction to the Theory of Neural Computation (Addison Wesley, Redwood City CA, 1991).

<sup>[8]</sup> D Ruelle, Thermodynamic Formalism: The Mathematical Structures of Classical Equilibrium Statistical Me-

chanics (Addison-Wesley, Reading MA, 1978).

<sup>[9]</sup> TC Halsey, MH Jensen, LP Kadanoff, I Procaccia, and BI Shraiman, Fractal measures and their singularities: The characterization of strange sets. *Phys Rev A* 33, 1141– 1151 (1986).

<sup>[10]</sup> MJ Feigenbaum, MH Jensen, and I Procaccia, Time ordering and the thermodynamics of strange sets: Theory and experimental tests. *Phys Rev Lett* 57, 1503–1506 (1986).

<sup>[11]</sup> S Schnabel, DT Seaton, DP Landau, and M Bachmann, Microcanonical entropy inflection points: Key to systematic understanding of transitions in finite systems. *Phys Rev E* 84, 011127 (2011).

<sup>[12]</sup> G Tkačik, O Marre, D Amodei, E Schneidman, W Bialek, and MJ Berry II, Searching for collective behavior in a large network of sensory neurons. *PLoS Comput Biol* 10, e1003408 (2014).

<sup>[13]</sup> O Marre, D Amodei, K Sadeghi, F Soo, TE Holy, and MJ Berry II, Recording from a complete population in the retina. J Neurosci 32, 14859–14873 (2012).

<sup>[14]</sup> ET Jaynes, Information theory and statistical mechanics.

- Phys Rev 106, 620-630 (1957).
- [15] E Schneidman, MJ Berry II, R Segev, and W Bialek, Weak pairwise correlations imply strongly correlated network states in a neural population. *Nature* 440, 1007– 1012 (2006).
- [16] G Tkačik, E Schneidman, MJ Berry II, and W Bialek, Ising models for networks of real neurons. arXiv.org:qbio/0611072 (2006).
- [17] J Shlens, GD Field, JL Gaulthier, MI Grivich, D Petrusca, A Sher, AM Litke, and EJ Chichilnisky, The structure of multi-neuron firing patterns in primate retina. J. Neurosci 26, 8254–8266 (2006).
- [18] A Tang, D Jackson, J Hobbs, W Chen, JL Smith, H Patel, A Prieto, D Petruscam MI Grivich, A Sher, P Hottowy, W Dabrowski, AM Litke, and JM Beggs, A maximum entropy model applied to spatial and temporal correlations from cortical networks in vitro. J Neurosci 28, 505-518 (2008).
- [19] G Tkačik, E Schneidman, MJ Berry II, and W Bialek, Spin-glass models for a network of real neurons. arXiv.org:0912.5409 (2009).
- [20] J Shlens, GD Field, JL Gaulthier, M Greschner, A Sher, AM Litke, and EJ Chichilnisky, The structure of large– scale synchronized firing in primate retina. *J Neurosci* 29, 5022–5031 (2009).
- [21] IE Ohiorhenuan, F Mechler, KP Purpura, AM Schmid, Q Hu, and JD Victor, Sparse coding and higher-order correlations in fine-scale cortical networks. *Nature* 466, 617–621 (2010).
- [22] E Ganmor, R Segev, and E Schniedman, Sparse low-order interaction network underlies a highly correlated and learnable neural population code, *Proc Natl Acad Sci (USA)* 108, 9679–9684 (2011).
- [23] G Tkačik, O Marre, T Mora, D Amodei, MJ Berry II, and W Bialek, The simplest maximum entropy model for collective behavior in a neural network. J Stat Mech P03011 (2013).
- [24] E Granot-Atedgi, G Tkačik, R Segev, and E Schneidman Stimulus-dependent maximum entropy models of neural population codes. PLoS Comput Biol 9, e1002922 (2013).
- [25] TR Lezon, JR Banavar, M Cieplak, A Maritan, and NV Federoff, Using the principle of entropy maximization to infer genetic interaction networks from gene expression patterns. Proc Natl Acad Sci (USA) 103, 19033–19038 (2006).
- [26] G Tkačik, Information Flow in Biological Networks (Dissertation, Princeton University, 2007).
- [27] W Bialek and R Ranganathan, Rediscovering the power of pairwise interactions. arXiv.org:0712.4397 (2007).
- [28] F Seno, A Trovato, JR Banavar, and A Maritan, Maximum entropy approach for deducing amino acid interactions in proteins. *Phys Rev Lett* 100, 078102 (2008).
- [29] M Weigt, RA White, H Szurmant, JA Hoch, and T Hwa, Identification of direct residue contacts in protein– protein interaction by message passing. *Proc Natl Acad Sci (USA)* 106, 67–72 (2009).
- [30] GJ Stephens and W Bialek, Statistical mechanics of letters in words. Phys Rev E 81, 066119 (2010).
- [31] T Mora, AM Walczak, W Bialek, and CG Callan, Maximum entropy models for antibody diversity. *Proc Natl Acad Sci (USA)* **107**, 5405–5410 (2010).
- [32] DS Marks, LJ Colwell, R Sheridan, TA Hopf, A Pagnani, R Zecchina, and C Sander, Protein 3D structure computed from evolutionary sequence variation. PLoS One

- 6, e28766 (2011).
- [33] JI Sulkowska, F Morocos, M Weigt, T Hwa, and JN Onuchic, Genomics-aided structure prediction. *Proc Natl Acad Sci (USA)* 109, 10340–10345 (2012).
- [34] W Bialek, A Cavagna, I Giardina, T Mora, E Silvestri, M Viale, and A Walczak, Statistical mechanics for natural flocks of birds. *Proc Natl Acad Sci (USA)* 109, 4786–4791 (2012).
- [35] S Mukherjee, S-C Seok, VJ Vieland, and J Das, Cell responses only partially shape cell-to-cell variations in protein abundances in *Escherichia coli* chemotaxis. *Proc* Natl Acad Sci (USA) 110, 18531–18536 (2013).
- [36] W Bialek, A Cavagna, I Giardina, T Mora, O Pohl, E Silvestri, M Viale, and A Walczak, Social interactions dominate speed control in poising natural flocks near criticality. Proc Natl Acad Sci (USA) 111, 7212–7217 (2014).
- [37] M Santolini, T Mora, and V Hakim, A general pairwise interaction model provides an accurate description of in vivo transcription factor binding sites. PLoS One 9, e99015 (2014).
- [38] T Mora and W Bialek, Are biological systems poised at criticality? J Stat Phys 144, 268–302 (2011).
- [39] F Wang and DP Landau, Efficient, multiple-range random walk algorithm to calculate the density of states. *Phys Rev Lett* 86, 2050–2053 (2001).
- [40] I Mastromatteo and M Marsili, On the criticality of inferred models. J Stat Mech P10012 (2011).
- [41] M Marsili, I Mastromatteo, and Y Roudi, On sampling and modeling complex systems. J Stat Mech P09003 (2013).
- [42] A related example is well known in condensed matter physics. Many systems exhibit 1/f noise, and it is tempting to search for deep and universal explanations, linking the temporal scale invariance of the noise to some sort of critical phenomenon. An alternative is that the observed noise is the sum of many terms, each of which has a characteristic time scale, but the distribution of these time scales is broad enough to generate a close approximation to scale invariance. This seems to be the answer for 1/f current noise in metals. P Dutta and PM Horn, Low-frequency fluctuations in solids: 1/f noise. Revs Mod Phys 53, 497–516 (1981).
- [43] SCF van Opheusden, Critical States in Retinal Population Codes (Masters thesis, Universiteit Leiden, 2013).
- [44] H Asari and M Meister, Feedback from retinal ganglion cells to the inner retina. Computational and Systems Neuroscience 2012; see http://cosyne.org.
- [45] DN Mastronarde, Interactions between ganglion cells in cat retina. J Neurophysiol 49, 350–365 (1983).
- [46] IH Brivanlou, DK Warland, and M Meister, Mechanisms of concerted firing among retinal ganglion cells, *Neuron* 20, 527–539 (1998).
- [47] GD Field and EJ Chichilnisky, Information processing in the primate retina: circuitry and coding. *Annu Rev Neurosci* **30**, 1–30 (2007).
- [48] DJ Schwab, I Nemenman, and P Mehta, Zipf's law and criticality in multivariate data without fine—tuning. arXiv:1310.0448v2 [q-bio.NC] (2013).
- [49] M Castellana and W Bialek, The inverse spin glass and related maximum entropy problems. arXiv.org:1312.0886 (2013).
- [50] F Rieke, D Warland, R de Ruyter van Steveninck, and W Bialek Spikes: Exploring the Neural Code. (MIT Press, Cambridge, 1997).

- [51] F Pereira, Formal grammar and information theory: Together again? *Phil Trans R Soc Lond A* **358**, 1239–1253 (2000).
- [52] FC Pereira, N Tishby, and L Lee, Distributional clustering of English words. In 31st Annual Meeting of the Association for Computational Linguistics, LK Schubert, ed., pp 183–190 (1993).
- [53] E Schneidman, W Bialek, and MJ Berry II, An information theoretic approach to the functional classification of neurons. In Advances in Neural Information Processing 15, S Becker, S Thrun, and K Obermayer, eds, pp 197–204 (MIT Press, Cambridge, 2003)
- [54] A Attanasi, A Cavagna, L Del Castello, I Giardina, S Melillo, L Parisi, O Pohl, B Rossaro, E Shen, E Silvestri, and M Viale, Wild swarms of midges linger at the edge of an ordering phase transition. arXiv:1307.5631 (2013)
- [55] D Krotov, JO Dubuis, T Gregor, and W Bialek, Morphogenesis at criticality. Proc Natl Acad Sci (USA) 111, 3683–3688 (2014).
- [56] AR Honerkamp-Smith, SL Veatch, and SL Keller, An introduction to critical points for biophysicists: observations of compositional heterogeneity in lipid membranes. *Biochim Biophys Acta* 1788, 53–63 (2009).
- [57] JM Beggs and D Plenz, Neuronal avalanches in neocortical circuits. J Neurosci 23, 11167–11177 (2003).
- [58] N Friedman, S Ito, BAW Brinkman, M Shimono, REL DeVille, KA Dahmen, JM Beggs, and TC Butler, Universal critical dynamics in high resolution neuronal avalanche data. *Phys Rev Lett* 108, 20812 (2012).
- [59] M Nykter, ND Price, M Aldana, SA Ramsey, SA Kauffman, LE Hood, O Yli-Harja, and I Shmulevich, Gene expression dynamics in the macrophage exhibit criticality. *Proc Natl Acad Sci (USA)* 105, 1897–1900 (2008).
- [60] E Balleza, ER Alvarez-Buylla, A Chaos, S Kauffman,

- I Shmulevich, and M Aldana, Critical dynamics in genetic regulatory networks: Examples from four kingdoms. *PLoS One* **3**, e2456 (2008).
- [61] VM Eguíluz, M Ospeck, Y Choe, AJ Hudspeth, and MO Magnasco, Essential nonlinearities in hearing. Phys Rev Lett 84, 5232–5235 (2000).
- [62] S Calamet, T Duke, F Jülicher, and J Prost, Auditory sensitivity provided by self-tuned critical oscillations of hair cells. *Proc Natl Acad Sci (USA)* 97, 3183–3187 (2000).
- [63] M Ospeck, VM Equíluz, and MO Magnasco, Evidence of a Hopf bifurcation in frog hair cells. *Biophys J* 80, 25972607 (2001).
- [64] JM Beggs and N Timme, Being critical of criticality in the brain. Front Physiol 3, 163 (2012).
- [65] GJ Stephens, T Mora, G Tkačik, and W Bialek, Statistical thermodynamics of natural images. Phys Rev Lett 110, 018701 (2013).
- [66] TM Cover and JA Thomas, Elements of Information Theory (Wiley, New York, 1991).
- [67] KG Wilson, Problems in physics with many scales of length. Sci Am 241, 158–179 (1979).
- [68] G Parisi, Statistical Field Theory (Addison-Wesley, Redwood City CA, 1988).
- [69] J Guckenheimer and P Holmes, Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields (Springer-Verlag, New York, 1983).
- [70] DH Perkel and TH Bullock, Neural coding. Neurosci Res Prog Sum 3, 221–348 (1968).
- [71] M Ioffe, G Tkačik, W Bialek, and MJ Berry II, Robustness of criticality under different stimulus conditions. *Computational and Systems Neuroscience 2014*; see http://cosyne.org.