# Physical Principles for Scalable Neural Recording

<sup>△</sup>Adam H. Marblestone,<sup>1,2</sup> <sup>△</sup>Bradley M. Zamft,<sup>3</sup> Yael G. Maguire,<sup>3,4</sup> Mikhail G. Shapiro,<sup>5</sup> Thaddeus R. Cybulski,<sup>6</sup> Joshua I. Glaser,<sup>6</sup> Dario Amodei,<sup>7</sup> P. Benjamin Stranges,<sup>3</sup> Reza Kalhor,<sup>3</sup> David A. Dalrymple,<sup>1,8,9</sup> Dongjin Seo, <sup>10</sup> Elad Alon,<sup>10</sup> Michel M. Maharbiz,<sup>10</sup> Jose M. Carmena,<sup>10,11</sup> Jan M. Rabaey,<sup>10</sup> Edward S. Boyden,<sup>>9,12</sup> George M. Church,<sup>>1,2,3</sup> and Konrad P. Kording,<sup>>13,14</sup>

□ Joint first authors □ Joint last authors

Correspondence to: adam.h.marblestone (at) gmail.com

(Draft produced September 16, 2013)

To understand in depth what is going on in a brain, we need tools that can fit inside or between neurons and transmit reports of neural events to receivers outside. We need observing instruments that are local, nondestructive and noninvasive, with rapid response, high band-width and high spatial resolution... There is no law of physics that declares such an observational tool to be impossible.

Freeman Dyson, *Imagined Worlds*, 1997

Abstract

Simultaneously measuring the activities of all neurons in a mammalian brain at millisecond resolution is a challenge beyond the limits of existing techniques in neuroscience. Entirely new approaches may be required, motivating an analysis of the fundamental physical constraints on the problem. We outline the physical principles governing brain activity mapping using optical, electrical, magnetic resonance, and molecular modalities of neural recording. Focusing on the mouse brain, we analyze the scalability of each method, concentrating on the limitations imposed by spatiotemporal resolution, energy dissipation, and volume displacement. Based on this analysis, all existing approaches require orders of magnitude improvement in key parameters. Electrical recording is limited by the low multiplexing capacity of electrodes and their lack of intrinsic spatial resolution, optical methods are constrained by the scattering of visible light in brain tissue, magnetic resonance is hindered by the diffusion and relaxation timescales of water protons, and the implementation of molecular recording is complicated by the stochastic kinetics of enzymes. Understanding the physical limits of brain activity mapping may provide insight into opportunities for novel solutions. For example, unconventional methods for delivering electrodes may enable unprecedented numbers of recording sites, embedded optical devices could allow optical detectors to be placed within a few scattering lengths of the measured neurons, and new classes of molecularly engineered sensors might obviate cumbersome hardware architectures. We also study the physics of powering and communicating with microscale devices embedded in brain tissue and find that, while radio-frequency electromagnetic data transmission suffers from a severe power-bandwidth tradeoff, communication via infrared light or ultrasound may allow high data rates due to the possibility of spatial multiplexing. The use of embedded local recording and wireless data transmission would only be viable, however, given major improvements to the power efficiency of microelectronic devices.

## 1 Introduction

Neuroscience depends on monitoring the electrical activities of neurons within functioning brains [1–3] and has advanced through steady improvements in the underlying observational tools. The number of neurons simultaneously recorded using wired electrodes, for example, has doubled every seven years since the 1950s, currently allowing electrical observation of hundreds of neurons at sub-millisecond timescales [4]. Recording techniques have also diversified: activity-dependent optical signals from neurons endowed with fluorescent indicators can be measured by photodetectors, and radio-frequency emissions from excited nuclear spins allow the construction of magnetic resonance images modulated by activity-dependent contrast mechanisms. Ideas for alternative methods have been proposed, including the direct recording of neural activities into information-bearing biopolymers [5–7].

Each modality of neural recording has characteristic advantages and disadvantages. Multi-electrode arrays enable the recording of  $\sim$ 250 neurons at sub-millisecond temporal resolutions. Optical microscopy can currently record  $\sim$ 100 000 neurons at a 1.25 s

timescale in behaving larval zebrafish using light-sheet illumination [8], or hundreds to thousands of neurons at a ~ 100 ms timescale in behaving mice using a 1-photon fiber scope [9]. Magnetic resonance imaging (MRI) allows non-invasive whole brain recordings at a 1s timescale, but is far from single neuron spatial resolution, in part due to the use of hemodynamic contrast. Finally, molecular recording devices have been proposed for scalable physiological signal recording but have not yet been demonstrated in neurons [5–7].

Figure 1 illustrates the recording modalities studied here. While further development of these methods promises to be a crucial driver for future neuroscience research [10], their fundamental scaling limits are not immediately obvious. Furthermore, inventing new technologies for scalable neural recording requires a quantitative understanding of the engineering problems that such technologies must solve, a landscape of constraints which should inform design decisions.

Figure 1. Four generalized neural recording modalities. (a) Extracellular electrical recording probes the voltage due to nearby neurons. (b) Optical microscopy detects light emission from activity-dependent indicators. In two-photon laser scanning microscopy, shown here, an excitation beam at 2× the peak excitation wavelength of the fluorescent indicator is scanned across the sample, while an integrating detector captures the emitted fluorescence. (c) Magnetic resonance imaging detects radio-frequency magnetic induction signals from aqueous protons, after weak thermal alignment of the proton spins by a static magnetic field. A resonant radio-frequency pulse tips the spins into a plane perpendicular to the static field, causing the net magnetization to precess. The resulting signals are affected by the local chemical and magnetic environment, which can be altered dynamically by imaging agents in response to neural activity. Activity-dependent contrast agents are necessary to transduce neural activity into an MRI readout, whereas current functional MRI methods rely on blood oxygenation signals which cannot reach single-neuron resolution. (d) Molecular recording devices have been proposed, in which a "ticker tape" record of neural activity is encoded in the monomer sequence of a biomolecular polymer – a form of nano-scale local data storage. This could be achieved by coupling correlates of neural activity to the nucleotide misincorporation probabilities of a DNA or RNA polymerase as it replicates or transcribes a known DNA strand.

![](_page_1_Picture_4.jpeg)

Our analysis is predicated on assumptions that enable us to estimate scaling limits. These include assumptions about basic properties of the brain, which are treated in section 2 (Basic Constraints), as well as those pertaining to the required measurement resolution and the limits to which a neural recording method may perturb brain tissue, which are treated in section 3 (Challenges for Brain Activity Mapping). Together, these considerations form the basis for our estimates of the prospects for scaling of neural recording

technologies. We analyze four modalities of brain activity mapping — electrical, optical, magnetic resonance and molecular — in light of these assumptions, and conclude with a discussion on opportunities for new developments.

Importantly, our assumptions, analyses and the conclusions thereof are intended as *first approximations and are subject to debate*. We anticipate that as much can be learned from where our logic breaks down as from where it succeeds, and from methods to work around the limits imposed by our assumptions.

## 2 BASIC CONSTRAINTS

Mouse brain The mouse brain contains  $\sim 7.5 \times 10^7$  neurons in a volume of  $\sim 420$  mm<sup>3</sup> [11] and weighs about 0.5 g. The packing density of neurons varies widely between brain regions. In the below, we will use a cell density of  $\rho_{\rm neurons} \approx 92000/\text{mm}^3$ , as measured for mouse cortex [12]. This corresponds roughly to one neuron per 22  $\mu$ m voxel. The density of cortical synapses, on the other hand, approaches  $10^9/\text{mm}^3$ , i.e., one synapse per  $1\,\mu\text{m}^3$  voxel. For comparison, the human brain has roughly  $8 \times 10^{10}$  neurons [13] in a volume of  $1200 \, \text{cm}^3$  [14].

The human brain consumes  $\sim 15$  W of power (performing, at synapses, a rough equivalent of at least  $10^{17}$  floating point computational operations per second on that power budget, according to one definition [15], although the analogy with digital computers should not be taken literally). Because power consumption scales approximately linearly with the number of neurons [16], the mouse brain is expected to utilize  $\sim 15$  mW. For comparison, the metabolic rate of the  $\sim 20-30$  g mouse is  $\sim 200-600$  mW depending on its degree of physical activity [17].

Neural activities Action potentials (spikes) last ~2 ms. The rate of neuronal spiking is highly variable. Some authors have assumed an average rate of 5 Hz [15, 18], but certain neurons spike at 500 Hz or faster [19], while many neurons spike much more slowly. For example, cerebellar granule cells, which make up half of the neurons in the brain, have spontaneous firing rates of ~0.5 Hz [20]. In neocortex, one analysis estimated 0.16 spikes per second per neuron (in primate) as energetically sustainable [21]. There may be as much as a two-fold change in metabolism and hence firing rate across brain states [22]. Certain neurons (possibly up to 90% for some neuron types in some brain areas) may be effectively silent [23, 24], e.g., spiking less than once every ten seconds. Some studies have attempted to measure the *distribution* of neural firing rates in various cortical areas (as opposed to just the average rate), and have observed that these distributions are often long-tailed: a small minority of the neurons fires a majority of the spikes [25–28].

While these estimates of typical firing rates are useful numbers to have in mind, in the below we aim to sample all neurons at 1 kHz rates (or higher for techniques requiring observation of detailed spike waveforms). This choice is informed by several factors. First, measuring spike *timing* with millisecond precision is relevant for understanding network function, due to the possibilities for timing codes, spike-timing dependent plasticity mechanisms, and other effects relying on temporally-precise spiking patterns [29–32]. In this regard, it is also important for a recording method to maintain precise temporal phasing between measurements at different brain locations: activity measurements should be locked to precise global clocks, perhaps with a tolerable phase imprecision between any two measurements in the range of  $\frac{1}{2\pi} \times 1 \text{ ms} \approx 100-200 \,\mu\text{s}$ . Furthermore, the activities of neurons can be highly correlated locally or across large networks [33], suggesting that local activity sensors may be subjected to high instantaneous total firing rates due to simultaneously-active neurons.

Absorption and scattering of radiation All existing methods of neural recording utilize electromagnetic waves, from the near-DC frequencies of wired electrical recordings (~1 kHz) to the radio-frequencies of wireless electronics and fMRI (MHz-GHz) to visible light in optical approaches (~500 THz). These electromagnetic waves are attenuated in brain tissue by absorption and scattering. As an approximation to the electromagnetic absorption by brain tissue, we treat the absorption by water, the brain's main constituent (68–80% by mass in humans [36, 37]). At visible and near-IR wavelengths, scattering dominates absorption: absorption lengths are in the ~1 mm range, while scattering lengths are ~25–200 µm [38]. The combined effect of absorption and scattering is measured by the attenuation length, the distance over which the signal strength is reduced by a factor of 1/e along a path. Figure 2 shows the absorption length of water [35], and the attenuation length in a Mie scattering model (from [39]) intended to approximate the scattering properties of cortical tissue (and see [40] for tissue skin depth measurements in the 10 Hz to 100 GHz range). This gives a preliminary indication of which wavelengths can be used to measure deep-brain signals with external detectors. Note that the attenuation length is only one of several relevant metrics: for example, scattering not only causes signal attenuation, but also causes noise and impairs signal separation, so the magnitude of the scattering is a key figure of merit.

Figure 2. Penetration depth (attenuation length) of electromagnetic radiation in water vs. wavelength (data from [34]). The approximate diameter of the mouse brain is shown as a black dashed line. Inset: approximate tissue model based on Mie scattering theory and water absorption. Absorption length of water [35] (blue), approximate tissue scattering length in a simple Mie scattering model (red) and the resulting attenuation length (green) of infrared light (inset reproduced from [35], with permission).

![](_page_3_Figure_2.jpeg)

## 3 CHALLENGES FOR BRAIN ACTIVITY MAPPING

Any activity mapping technology must extract the required information without disrupting normal neuronal activity. As such, we consider three primary challenges: spatiotemporal resolution and informational throughput, energy dissipation and volume displacement.

## 3.1 SPATIOTEMPORAL RESOLUTION AND INFORMATIONAL THROUGHPUT

A sampling rate of 1 kHz is necessary to capture the fastest trains of action potentials at single-spike resolution. A minimal data rate of  $7.5 \times 10^{10}$  bits processed per second is then required to record 1 bit per mouse neuron at 1 kHz.

In electrical recording, higher sampling rates (e.g. 10-40 kHz) are often necessary to distinguish neurons based on spike shapes when each electrode monitors multiple neurons. More fundamentally, one bit per neuron sampling at 1 kHz would likely not be sufficient to reliably distinguish spikes above noise: transmitting  $\sim 10 \text{ bit}$  samples at  $\sim 10 \text{ kHz}$  (full waveform) or  $\sim 10-20 \text{ bit}$  time-stamps upon spike detection would be more realistic.

Conversely, it may be possible to locally compress measurements of a spike train before transmission. The degree of compressibility of neural activity data is related to the variability in the distribution of neural responses (e.g., such a distribution may be defined across time bins or repeated stimulus presentations) [41]. In the blowfly *Calliphora vicina*, the entropy of spike trains has been measured to be up to  $\sim 180$  bit/s, and the information about a stimulus encoded by a spike train was as high as  $\sim 90$  bit/s [41]. Extrapolating from fly to mouse, this would suggest that a compression factor of  $5\times-10\times$  should be possible, relative to a 1000 bit/s raw binary sampling.

As a naïve estimate of the entropy as a function of firing rate, one can write the entropy H in bit/s, assuming 1 ms long spikes and  $f = 1000 \,\mathrm{Hz}$  sampling rate, as

$$H \approx \left( -P_{\text{spike}} \cdot \log_2 \left( P_{\text{spike}} \right) - \left( 1 - P_{\text{spike}} \right) \cdot \log_2 \left( 1 - P_{\text{spike}} \right) \right) \cdot f$$

where  $P_{\rm spike}$  is the probability of spiking during the sampling interval (average firing rate/f). For an average firing rate of 5 Hz,  $P_{\rm spike}=0.005$  and H=45 bit/s, corresponding to a compression factor of  $\sim\!20\times$ . However, at 500 Hz average firing rate,  $P_{\rm spike}=0.5$  with  $H\approx1000$  bit/s, i.e., there is no compressibility. Therefore, compression could conceivably reduce the data transmission burden for activity mapping by 1–2 orders of magnitude, depending on the neurons and activity regimes under consideration. Note that these compressibility calculations have assumed that firing patterns are independent across cells; they represent the temporal compressibility of the spike train from each cell, treated individually. Patterns across cells could conceivably be compressed by a much larger amount, to the extent that there is redundancy between cells. Nevertheless, we use 1 bit/neuron/ms or 100 Gbit/s as a "minimal whole brain data rate" in what follows. In many cases, this likely constitutes a lower bound on what is feasible in practice.

#### 3.2 ENERGY DISSIPATION

Brain tissue can sustain local temperature increases ( $\Delta T$ ) of  $\sim$ 2 °C without severe damage over a timescale of hours. Indeed, changes of this magnitude may occur naturally in rats in response to varying activity levels [42]. Assuming that the brain is receiving a constant power influx  $P_{\text{delivered}}$  and that the local thermal transport properties of mouse brains are similar to those of humans, we can approximate the temperature change in deep-brain tissue as a function of the applied power [43, 44]:

$$\frac{\mathrm{d}T}{\mathrm{d}t} = \left( P_{\text{delivered}} + P_{\text{metabolic}} - \rho_{\text{blood}} C_{\text{blood}} f_{\text{blood}} \Delta T \right) / C_{\text{tissue}}$$

where  $P_{\rm metabolic} = 0.0116\,{\rm W/g}$  is the power per unit mass of basal metabolism,  $C_{\rm tissue} \approx 3.7\,{\rm J/(K\,g)} \approx 0.88 \cdot C_{\rm water}$  is the specific heat capacity of brain tissue,  $\rho_{\rm blood} = 1.05\,{\rm g/cm^3}$  is the density of blood,  $C_{\rm blood} = 3.9\,{\rm J/(K\,g)}$  is the specific heat capacity of blood,  $f_{\rm blood} = 9.3 \times 10^{-9}\,{\rm m^3/(g\,s)}$  is the volume flow rate of blood, and  $\Delta T$  is the temperature difference between the brain tissue and the blood (at 37 °C). A steady-state temperature increase (d $T/{\rm d}t = 0$ ) of 2 °C corresponds to dissipation of ~40 mW per 500 mg mouse brain. Therefore, a recording technique should not dissipate more than ~40 mW of power in a mouse brain at steady state.

This estimate of the power dissipation limit in mouse brains, based on such a simplified model of the brain's thermal transport mechanisms, is likely an under-estimate of the actual maximum steady-state power dissipation. Radiative heat loss was ignored here since infrared light emitted by deep-brain tissue is quickly re-absorbed by nearby tissue. We have also ignored cooling due to flows in the cerebrospinal ventricles [45] and in the glymphatic system [46]. We have further assumed that conductive heat loss from the

brain surface is negligible compared to the heat extracted volumetrically by blood flow. While this may hold true locally in deep brain voxels and over short timescales (e.g., < 1 min), further work (e.g., a whole-head model [44, 47]) is needed to define the true limits of sustained volumetric heat production by neural recording systems distributed throughout the mouse brain. Indeed, the characteristic length scale of temperature inhomogeneities in the brain is on the order of millimeters [48], whereas heat exchange with the flowing blood dampens the effects of local perturbations over longer length scales. For large brains, this means that sources and sinks of heat exert only local thermal effects; for a mouse brain on the scale of < 10 mm, however, surface and volumetric effects likely combine to influence temperature changes at any site in the brain [49]. Experimentally, increasing the temperature gradient at the brain surface, via a cranial window exposed to ambient air at ~25 °C (i.e., the common craniotomy technique used to access mouse neocortex), has been shown to dis-regulate brain temperature down to a depth of several millimeters [50]. For the above reasons, our estimates of the brain's capacity for heat dissipation should be treated only as first approximations.

Higher power levels, compared to the maximum steady state power, may be introduced into brains transiently. According to the above equation, if a neural recorder dissipates ~40 mW per 500 mg mouse brain, then the brain approaches the steady-state temperature in 2–3 min, making shorter experiments potentially feasible. This is in agreement with the estimate from [48] of a ~1 min time constant for brain temperature changes, as well as with experimental measurements showing similar time constants for temperature variations resulting from sustained neural stimulation [51, 52]. Increasing convective heat loss from the brain by increasing blood flow (e.g. via increased heart rate) or cooling the brain (volumetrically or via its surface [49]), the blood, the cerebrospinal fluid (CSF), or the whole animal [53], could increase the allowable transient or steady-state power dissipation.

There are also limits on the power density of radiation applied to brain tissue. For radio-frequency electromagnetic radiation, the specific absorption rate (SAR) limit on the power density exposed to human tissue is  $\sim 10 \text{ mW/cm}^2$  [54], while for ultrasound (which couples less strongly to dissipative loss mechanisms in tissue) the SAR limit is up to  $72\times$  higher [55]. The power density limit for visible and near-IR light exposures are also in the  $\sim 10-100 \text{ mW/cm}^2$  range for  $\sim 1 \text{ ms}$  long exposures, decreasing as the exposure time lengthens (based on the IEC 60825 formulas [56]).

High local power dissipation (transient or steady-state) can modify the electrical properties of excitable membranes, altering neuronal activity patterns. For example, heating of cell membranes and of the surrounding solution by millisecond-long optical pulses leads to changes in membrane electrical capacitance mediated by the ionic double layer [57]. Slower temperature changes (on a scale of seconds) resulting from RF radiation lead to accelerated ion channel and transporter kinetics [58]. Both of these effects are appreciable when the temperature changes are on the order of 1–10 °C.

For comparison with current practice, common guidelines for chronic heat exposure from biomedical implants [42] use upper limits of 2°C temperature change, 40 mW/cm² heat flux from the surface of implanted brain machine interface (BMI) hardware, and an SAR limit of

$$\frac{\sigma E^2}{2\rho} < 1.6 \,\mathrm{mW/g}$$

for electromagnetic energy absorbed by tissue, where E is the peak electric field amplitude of the applied radiation,  $\sigma \approx 0.18 \,\text{S/m}$  is the electrical conductivity of grey matter and  $\rho \approx 1 \,\text{g/cm}^3$  is the tissue density [44] (this corresponds to an irradiance of  $\epsilon_0 c E^2/2 \approx 2.4 \,\text{mW/cm}^2$ ). A 96-channel BMI system demonstrated in living brains had dissipated areal power density approaching 40 mW/cm<sup>2</sup> [59].

#### 3.3 SENSITIVITY TO VOLUME DISPLACEMENT

To prevent damage to the brain, we assume that a recording technique should not displace > 1 % of the brain's volume. The appropriate damage threshold is not yet established, however, so this constitutes a first guess. It is possible to insert large numbers of probes throughout multiple brain areas without compromising function. In rats, 96 electrodes of 50 µm diameter were simultaneously inserted across four forebrain structures (cortex, thalamus, hippocampus and putamen) [60]. In rhesus macaque, 704 electrodes of diameter 50 µm and average depth 2.5 mm were chronically implanted in cortex [61]. Note, however, that the total volume displacement in these experiments was below 0.1 %, and below 0.01 %, respectively. Furthermore, these studies used a low density of electrodes. Thus, detailed limits on the amount and density of inserted material are unknown.

Furthermore, the nature of the volume displacement is important — sheets of instrumentation that sever long-range connectivity, for example, would disrupt normal brain function regardless of the degree of volume displacement. Conversely, higher volume displacement might be possible if introduced gradually, or during early development, insomuch as the brain can adapt without disrupting natural computation. One important consideration in this regard would be the disruption of blood circulation by inserted material; a high density of implanted material in a brain region could cause stroke due to widespread vascular damage. Recent studies have defined in microscopic detail the complete vascular network of the mouse cortex using high-throughput histology [62]; this type of information could be used to enumerate key vascular pathways which could be spared from damage. To apply this in a particular

animal, however, would require a non-destructive method to image the vasculature at a similar resolution; otherwise, only a broad statistical view can be obtained, since the detailed vascular geometry will vary from animal to animal.

Secondary effects like glial scarring may also pose obstacles to the long-term implantation of large numbers of probes [63, 64], although methods are being developed to alleviate this [65–67]. In the context of electrical recording, the impact of glial scarring may vary depending on geometry. For example, the recording sites at the tip of a Utah or Duke multi-electrode array are typically viable in chronic recordings of up to 18 months in primates [61, 68], whereas in array formats with multiple electrodes along each shaft, such as the Michigan array, chronic recordings of up to 4 months have been reported in rats [69]. Differences in recording lifetime may be due to differences in the pattern of glial encapsulation of the contacts.

## 4 EVALUATION OF MODALITIES

We next evaluate neural recording technologies with respect to the above challenges, using the mouse brain as a model system. Table 1 lists the modalities studied, the assumptions made, the analysis strategies applied, and the conclusions derived.

#### 4.1 ELECTRICAL RECORDING

In the oldest strategy for neural recording, an electrode is used to measure the local voltage at a recording site, which conveys information about the spiking activity of one or more nearby neurons. The number of recording sites may be smaller than the number of neurons recorded since each recording site may detect signals from multiple neurons. As a note for practitioners, we use the term "electrode" interchangeably with the terms "recording site" or "contact", meaning a point-like voltage sensing node: many multi-electrode arrays in common use (e.g., the Duke and Utah arrays) are conductive only at the tip, whereas other designs (such as the Michigan array) have multiple contacts along the shaft. Each shaft in a Michigan array would thus constitute multiple "electrodes" or "recording sites" in our parlance. Traditional electrical recording techniques keep active devices such as amplifiers outside the skull and therefore do not pose a heat dissipation challenge; this may change if amplifiers are brought closer to the signal sources to reduce noise.

Slowly varying (e.g., < 300 Hz) extracellular potentials (LFPs) [71, 72] on the order of 0.1–1 mV, and fields [73] on the order of 1–10 mV/mm, are generated by neural activity. While LFPs can be filtered from the higher-frequency signals associated with extracellular voltage spikes, these and other effects necessitate maintaining precise potential references (i.e., ground levels) for voltage measurements distributed widely across the brain.

#### 4.1.1 SPATIOTEMPORAL RESOLUTION

Limits assuming perfect spike sorting We begin with an idealized estimate of the number of electrodes required to record from the entire mouse brain, neglecting the difficulty of assigning observed spikes to specific cells (spike sorting), and focusing only on what is needed to detect spikes from every neuron on at least one electrode. The key variable here is the maximum distance between an extracellular electrical recorder and a neuron from which it records spikes. In a first approximation, this is determined by two factors: the decay of the signal with distance from the spiking neuron and the background noise level at the recording site. We assume that for an electrode to reliably detect the signal from a given neuron, the magnitude of that neuron's signal must be larger than the electrode's noise level. Note, however, that knowledge of spike shape distributions could potentially be used to extract low-amplitude spikes from noise.

The peak signals of spikes from neurons immediately adjacent to an electrode are in the 0.1–1.0 mV range and scale roughly as  $e^{-r/r_0}$ , where r is the distance from the cell surface and the 1/e falloff distance,  $r_0$ , has been experimentally measured at ~28  $\mu$ m in both salamander retina [74] and cat cortex [75], and computed at ~18  $\mu$ m in a biophysically realistic simulation [76, 77]. However, this decay is strongly influenced by the detailed geometry of neuronal currents and the properties of the extracellular space (e.g., its inhomogeneity, which may lead to a frequency-dependent falloff of the extracellular potential [78]), making analytical calculation of the decay rate difficult (at large distances, a much slower  $1/r^2$  dipole falloff is expected).

Several sources of background noise enter the recordings. Johnson noise, which arises from thermal fluctuations in the electrode, is

$$V_{\text{johnson}} = (4k_{\text{B}}TZBW)^{1/2}$$

which for physiological temperature, electrodes of impedance  $Z=0.5\,\mathrm{M}\Omega$ , and BW = 10 kHz bandwidth is  $V_{\mathrm{johnson}}\approx9\,\mu\mathrm{V}$ . The recordings are also affected by interference from other neurons, which has been reported to exceed the Johnson noise, and is non-stationary due to changes in the cells' firing properties [79]. The noise and interference from these sources realistically produces

Table 1: Summary of modalities, models, assumptions and conclusions

| Modality                           | Analysis Strategy                                                                                                                            | Assumptions                                                                                                                                                                                 | Conclusions                                                                                                                                                                                                                                                                                                                                 |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Extracellular electrical recording | Compute minimal number of recorders based on max distance from recorder to recorded neuron  Compute channel capacity limits to spike sorting | Decay profile of extracellular voltage<br>Approximate noise levels at recording site                                                                                                        | Maximum recording distance $r_{\rm max} \approx 100-200  \mu {\rm m}$ from electrode to neuron measured $\sim 10^5$ recording sites are required per mouse brain at cur-                                                                                                                                                                    |
|                                    |                                                                                                                                              |                                                                                                                                                                                             | rent noise levels assuming perfect spike sorting ~10 <sup>6</sup> recording sites are required at current noise levels                                                                                                                                                                                                                      |
|                                    |                                                                                                                                              |                                                                                                                                                                                             | at the physical limits of spike sorting ~10 <sup>7</sup> recording sites are required using current spike sorting algorithms                                                                                                                                                                                                                |
| Implanted electrical<br>recorders  | Compute power dissipation of electronic devices that digitally sample neuronal activity                                                      | Physical limit: $k_{\rm B}T\ln(2)/{\rm bit}$ erased<br>Practical limit: $\sim 10k_{\rm B}T/{\rm bit}$ processed<br>Current CMOS digital circuits:<br>$> 10^5k_{\rm B}T/{\rm bit}$ processed | Requires 2–3 orders of magnitude increase in the power efficiency of electronics relative to current devices to scale to whole-brain simultaneous recordings                                                                                                                                                                                |
|                                    |                                                                                                                                              |                                                                                                                                                                                             | Minimalist architectures could be developed to reduce local data processing overhead                                                                                                                                                                                                                                                        |
| Wireless data<br>transmission      | Compute tradeoff between<br>power dissipation and chan-<br>nel bandwidth using infor-<br>mation theory                                       | Transmitter must supply enough power to overcome noise and path loss                                                                                                                        | Transmission at optical or near-optical frequencies is needed to achieve sufficient single-channel data rates using electromagnetic radiation. Radio-frequency (RF) electromagnetic transmission of whole-brain activity data draws excessive power due to bandwidth constraints                                                            |
|                                    |                                                                                                                                              |                                                                                                                                                                                             | Bandwidth cannot be split over multiple independent<br>RF channels, but IR light or ultrasound may allow spa-<br>tial multiplexing                                                                                                                                                                                                          |
| Optical imaging                    | Relate the scattering and absorption lengths of optical wavelengths in brain tissue to signal-to-noise ratios for optical imaging            | Approximate values of scattering and absorption lengths as a function of wavelength                                                                                                         | Light scattering imposes severe constraints, but strategies exist which could negate the effects of scattering, such as implantable optics, infrared indicators, signal modulation, and online inversion of the scattering matrix                                                                                                           |
| Multi-photon optical imaging       | Compute minimum total excitation light power to excite multi-photon transitions from indicators within each neuron in every imaging frame    | Approximate values of multi-photon cross-sections                                                                                                                                           | Whole-brain multi-photon excitation will over-heat the brain except in very short experiments, unless ultra-high-cross-section indicators are used                                                                                                                                                                                          |
|                                    |                                                                                                                                              | Pulse durations similar to those currently used in multi-photon imaging                                                                                                                     |                                                                                                                                                                                                                                                                                                                                             |
| Beam scanning microscopies         | Calculate device and indicator parameters necessary for fast beam repositioning and signal detection                                         | Fast optical phase modulators could reposition beams at $\sim$ 1 GHz switching rates Fluorescence lifetimes in the 0.1–1.0 ns range                                                         | Beam repositioning time limits the speed of current systems but these are far from the physical limits                                                                                                                                                                                                                                      |
|                                    |                                                                                                                                              |                                                                                                                                                                                             | Fluorescence lifetimes of indicators constrain design of ultra-fast scanning microscopies                                                                                                                                                                                                                                                   |
| Magnetic resonance<br>imaging      | Calculate spatial and temporal resolution of MRI based on spin relaxation times and spin diffusion                                           | Proton MRI using tissue water Approximate $T_1$ and $T_2$ relaxation times and self-diffusion times for tissue water                                                                        | Proton MRI is limited by the $T_1$ relaxation time of water to $\sim$ 100 ms temporal resolution and by the self-diffusion of water to spatial resolutions of $\sim$ 40 $\mu$ m. $T_1$ premapping could allow $T_2$ contrast on a $\sim$ 10 ms timescale. Achieving these limits for functional imaging requires going beyond BOLD contrast |
| Ultrasound                         | Calculate spatial resolution,<br>signal strength and band-<br>width limits on ultrasound<br>imaging                                          | Speed of sound in brain Attenuation length of ultrasound in brain                                                                                                                           | Attenuation of ultrasound by brain tissue and bone may be prohibitive at the $\sim 100$ MHz frequencies needed for                                                                                                                                                                                                                          |
|                                    |                                                                                                                                              | Accordation longer of unitasound in oralli                                                                                                                                                  | single-cell resolution ultrasound imaging Ultrasound may be viable for spatially multiplexed data                                                                                                                                                                                                                                           |
| Molecular recording                | Compute metabolic load<br>and volume constraint for<br>rapid synthesis of large                                                              | Polymerase biochemical parameter ranges<br>Metabolic requirements of genome<br>replication                                                                                                  | transmission from embedded devices [70]  Molecular recording devices appear to fall within physical limits but their development poses multiple major challenges in synthetic biology                                                                                                                                                       |
|                                    | nucleic acid polymers  Evaluate temporal resolution in simulated experiments using kinetic models [6]                                        | - Common                                                                                                                                                                                    | Synchronization or time-stamping mechanisms are required for temporal resolution to approach the millisecond scale                                                                                                                                                                                                                          |

> 10–20  $\mu$ V of voltage fluctuations [80]. Current recording setups thus have signal to interference-plus-noise ratios (SINRs) of < 100, where the SINR is defined as the ratio of the peak voltage from immediately adjacent neurons to the voltage fluctuation floor of the electrode.

A limit on the maximum recording distance is the distance at which the signal from the farthest neuron falls below the noise floor,  $r_{\text{max}} \approx r_0 \ln(\text{SINR})$ . For SINR  $\approx 100$ ,  $r_{\text{max}} \approx 130 \,\mu\text{m}$ . For comparison, recent experimental data from multi-site silicon probes has shown few detectable neurons beyond  $\sim 100 \,\mu\text{m}$  and none detectable beyond 160  $\mu\text{m}$  [81]. Recordings in the hippocampal CA1 region could not detect spikes from cells farther than 140  $\mu$ m from the electrode tip [82], even after averaging over observations triggered on an intracellularly recorded spike; in hippocampus, this corresponds to a detection volume containing approximately 1000 neurons [83]. Furthermore, in many studies (in monkeys, rats and mice) using multi-electrode arrays with 150–300  $\mu$ m inter-electrode spacings, no neuron is seen by more than one electrode [84–87].

Due to the steep local falloff, even improving the SINR by a factor of 10 only extends the maximal recording distance to  $r_{\rm max}\approx 190\,\mu{\rm m}$ . Assuming packing of the brain into equal sized cubes of side length  $d=\frac{2\sqrt{3}}{3}r_{\rm max}\approx 150\,\mu{\rm m}$  gives  $N>130\,000$  electrodes for whole brain recording using recording sites with  $r_{\rm max}\approx 130\,\mu{\rm m}$ . Note that N varies as the third power of  $r_{\rm max}$  and is therefore highly sensitive to variations in the assumed maximal recording distance; the number of required recorders can range from 38 000 to 210 000 as  $r_{\rm max}$  varies from 190  $\mu{\rm m}$  to 110  $\mu{\rm m}$ .

These calculations, by assuming perfect spike sorting, greatly underestimate the required number of electrodes in practice. First, signals from the weakest cells are far weaker than those from the strongest cells and the signals from some cells decay much faster than others [75]. Second, because of neuronal synchronization, the local noise produced by nearby neurons may sometimes be large. Third, spike waveforms can vary over the course of a recording session [88, 89]. Finally, with many neurons per electrode or at high firing rates, spikes from detectable neurons will often temporally overlap, making spike sorting difficult.

Figure 3. The voltage signal to interference-plus-noise ratio (SINR) for neurons immediately adjacent to the recording site sets an approximate upper bound on the distance,  $r_{\rm max}$ , between the recording site and the farthest neuron it can sense (blue), due to the exponential falloff of the voltage SINR with distance. Assuming at least one electrode per cube of edge length  $\frac{2\sqrt{3}}{3}r_{\rm max}$  in turn limits the number of neurons per recording site (gold), the total number of recording sites (red) and the maximal diameter of wiring consistent with < 1% total brain volume displacement (turquoise). SINR values for current recording setups are <  $10^2$ . In practice, the number of neurons per electrode distinguishable by current spike sorting algorithms is only ~10, with an estimated information theoretic limit of ~100, so these curves *greatly under-estimate* the number of electrodes which would be required based on realistic spike sorting approaches in a pure voltage-sensing scenario.

![](_page_8_Figure_6.jpeg)

Limits from spike sorting The previous calculations have assumed that any spike which is visible above the noise on at least one electrode can be detected and correctly assigned to a particular cell, i.e., that the problem of spike sorting can be solved perfectly. However, perfect spike sorting is far beyond current algorithmic capabilities and in fact may not be possible in principle.

To achieve the scenario described above, with  $N=130\,000$  recording sites per mouse brain, would require each electrode to sort spikes from all  $\frac{4}{3}\pi r_{\rm max}^3 \rho_{\rm neurons}$  neurons in a sphere of radius  $r_{\rm max}\approx 130\,\mu{\rm m}$  surrounding the recording site, where  $\rho_{\rm neurons}\approx 92\,000/{\rm mm}^3$  is the density of neurons. This assigns ~800 neurons to a single electrode. Roughly half (i.e., 400) of these neurons will lie at >  $100\,\mu{\rm m}$  distance from the electrode, and their signals on the electrode will therefore have voltage SINRs of  $< 100e^{-100\,\mu{\rm m}/28\,\mu{\rm m}}\approx 2.8$ , assuming as above that extracellular spike amplitudes decay exponentially in space.

Electrical recording can be viewed as a data transmission problem, with the electrode playing the role of a communication channel (see section 4.4). According to the Shannon Capacity Theorem [90], the information capacity C of a single analog channel (with additive white Gaussian noise) is

$$C = BW \log_2(1 + S/N)$$

where BW is the bandwidth, S is the signal power (proportional to the square of the voltage), and N is the noise power. Here the bandwidth is BW  $\approx 10\,\mathrm{kHz/s}$ , and the ratio of peak signal power to noise power of a single spike for the outer 400 cells is no more than  $2.8^2$ , or  $0.5 \times 2.8^2$  using the RMS signal power instead of the peak. With 400 cells emitting 2 ms spikes at 5 Hz, there will be an average of 4 cells spiking at a time, for  $S/N \approx 0.5 \times 4 \times 2.8^2 \approx 15.7$  counting the signal power from all the spikes. The channel capacity is then  $C \approx 40\,\mathrm{kbit/s}$ . This represents the maximum amount of information (e.g., about which neuron spiked when) that the population of spiking neurons can transmit via the electrode which measures them. To transmit *uniquely identifiable* signals from all 400 neurons at millisecond temporal precision, however, requires  $1\,\mathrm{kbit/s} \times 400 = 400\,\mathrm{kbit/s}$ , which is  $> 10\times$  greater than the channel capacity and is therefore not achievable. Even with optimal temporal compression of  $\sim 5\,\mathrm{Hz}$  spikes (see section 2), we would need to transmit  $\sim 400/20 = 20\,\mathrm{kbit/s}$ , which is strictly less than the channel capacity and thus possible in principle, but barely so. Furthermore, the channel capacity given here is an overestimate, since 2.8 is an upper bound on the SINR of the outer cells. On the other hand, note that the use of a nominal  $5\,\mathrm{Hz}$  average firing rate here (in the estimates of signal to noise ratio and of temporal compressibility) greatly oversimplifies the distribution of firing rates across neurons, as discussed in section 2 above, so this analysis can only be treated as a first approximation.

Based on these rough estimates, perfect spike sorting may not be possible at  $\sim$ 800 neurons per electrode, in a sphere of radius 130  $\mu$ m surrounding a recording site, and at the noise levels typical of current electrodes. In essence, there may not be enough room on the electrode's voltage trace to discriminate such a large number of weak, noisy signals. Note that these information-theoretic limits still apply even if it is possible to resolve temporally overlapping spikes. In fact, the channel capacity is what ultimately limits the ability of a spike sorting algorithm to resolve such overlapping spikes.

To see the regime in which spike sorting becomes feasible, suppose that each electrode is only responsible for spike sorting from the population of ~100 neurons nearest to the electrode, i.e., in a sphere of radius  $r \approx 64 \,\mu\text{m}$ , assuming the 92000/mm³ cell density from mouse cortex. The outermost 50% of these neurons are then positioned > 50  $\mu$ m from the recording site. For these outermost 50 neurons, the voltage SINR is <  $100e^{-50 \,\mu\text{m}/28 \,\mu\text{m}} \approx 17$  and  $S/N < 0.5 \times 17^2 \times (2 \,\text{ms} \times 5 \,\text{Hz} \times 50) \approx 72.3$ . The channel capacity is therefore <  $62 \,\text{kbit/s}$ , whereas 50 kbit/s is needed for signal transmission from 50 neurons without temporal compression versus ~2.5 kbit/s with temporal compression. Even 100 neurons per electrode may therefore still be close to the limits of information transmission through the noisy channel corresponding to a single electrode.

In practice these limits are likely to be highly optimistic, since the set of spikes emerging from a neuronal population is far from an optimally designed code from the perspective of multiplexed signal transmission through a voltage-sensing electrode: the waveforms for different neurons are similarly-shaped rather than orthogonal, the spikes emitted by a given neuron vary somewhat in amplitude and exhibit shape fluctuations (signal-dependent noise), and it is not known in advance what the characteristic signal from each neuron looks like (or even how many neurons there are).

Indeed, current practice is far from the above information-theoretic limits. At present, spike sorting algorithms operating on data from large-scale (250-500 electrodes), densely spaced ( $\sim$ 30 µm), 2D multi-electrode arrays can reliably identify and distinguish spikes from nearly all of the 200-300 retinal ganglion cells [91, 92] in a small patch of retina, and can also infer approximate cell locations through spatial triangulation of spike amplitudes. This represents a roughly 1:1 ratio of cells to electrodes. Electrodes with up to 4 single units can be found in chronically implanted multi-electrode arrays (in both mouse and primate) [61, 93], where the electrodes are sparse, although the average yield of cells per electrode is closer to 1:1; if only electrodes with at least one cell are counted, the average rises to  $\sim$ 1.5–1.7 cells per electrode. Optimistically, simulations of neural activity suggest that 5-10 neurons per electrode may be distinguishable using current spike sorting algorithms [79, 80, 94]. A limit of  $\sim$ 10 neurons per electrode would imply  $N = 7.5 \times 10^6$  electrodes to record from all neurons in the mouse brain, which could be accomplished by positioning recording sites on a cubic lattice with  $\sim$ 40 µm edge length.

Future algorithmic improvements could enable sorting from more than ~10 cells per electrode, but this becomes increasingly challenging. One simple estimate of a reasonable practical limit, for the regime of many neurons per electrode, would be the largest number of neurons that can be sorted without requiring the frequent resolving of temporally overlapping spikes: if the average neuron fires at ~5 Hz and spikes last ~2 ms, then at most roughly 100 neurons per electrode can be sorted without requiring overlaps to be resolved. Note that while some present-day algorithms can successfully resolve overlapping spikes [74, 91, 92, 95, 96], they typically do so only in the case where electrodes are densely spaced and any given spike appears on many electrodes, such that spatial information can be used to resolve the overlap. Resolving overlaps when spikes appear on only one or a few channels is more difficult due to noise and spike-shape variation.

Overall,  $\sim$ 100 cells per electrode may be taken as a rough estimate of the limits of spike sorting, and would imply  $N=750\,000$  electrodes and an edge spacing of  $\sim$ 80  $\mu$ m if a cubic lattice of recording sites were used. However, we should not exclude the possibility of game-changers which could alter the nature of the recorded data to improve the available information. For instance, CCD cameras could be attached to multi-electrode arrays to aid in the identification and localization of cells, or directional information on the source of spikes could be obtained at each recording site, for example by measuring the directions of gradients in voltage. Systems that capture such additional information could circumvent the above information-theoretic limits and improve spike sorting.

#### 4.1.2 VOLUME DISPLACEMENT

We require < 1% total volume displacement from N recorders. Wires from each electrode must make it to the surface of the brain, which implies an average length  $\ell \approx 4$  mm for the mouse brain (depending on assumptions about the wiring geometry).

As a rough approximation, consider each recorder to produce a volume displacement associated with a single cylindrical wire, with length  $\ell$  and radius r. Thus r must satisfy

$$\pi r^2 \ell N_{\text{min rd}} < 0.01 V_{\text{brain}}$$

Using  $N_{\rm min,rd} = 210\,000$  or 38 000 recording sites (lower and upper limits from the perfect spike sorting case from above) and  $\ell \approx 4\,\rm mm$  requires wires of radius  $r_{\rm max} \approx 6.0\,\mu m$ , or 2.5  $\mu m$ , respectively. Alternatively, if 7.5  $\times$  10<sup>6</sup> electrodes must be used (current spike sorting case from above), the required wire radius is  $\sim$  200 nm. While these dimensions are readily achievable using lithographic fabrication, there would be a challenge to produce *isolated* wires of such dimensions at scale (perhaps suggesting the use of wire bundles). Still, volume constraints per se are unlikely to fundamentally limit whole-mouse-brain electrical recording even in the most pessimistic scenario.

Figure 3 illustrates the above considerations as a function of the electrode SINR.

#### 4.1.3 IMPLANTING ELECTRODES IN THE BRAIN

There are several technology options for introducing many electrodes into a brain. For example, flexible nanowire electrodes could, in theory, be threaded through the capillary network [97]. Capillaries are present in the brain at a density of 2500–3000 per mm<sup>3</sup> [98], which equates to one capillary per 73  $\mu$ m, with each neuron lying within ~200  $\mu$ m of a capillary [99]. The minimum capillary diameter is as small as 3–4  $\mu$ m, although the average diameter is ~8  $\mu$ m, comparable to the non-deformed size of the red blood cells [100]. Blocking a significant fraction of capillaries could lead to stroke or to unacceptable levels of tissue necrosis/liquifaction.

The cerebrospinal ventricles may also provide a convenient location for recording hardware. Furthermore, neural tissues could be grown around pre-fabricated electrode arrays [101], or silicon probes arrays with many nano-fabricated recording sites per probe [81] could be inserted into the brain.

Mechanical forces during insertion and retraction of silicon and tungsten microelectrodes from brain tissue have been measured in rat cortex at  $\sim$ 1 mN for electrodes of  $\sim$ 25  $\mu$ m radius [102]. These forces are comparable to the Euler buckling force F of a 2 mm long cylindrical tungsten rod of r =5  $\mu$ m radius

$$F = \frac{\pi^2 EI}{(KL)^2} \approx 1 \,\text{mN}$$

where  $E=411\,\mathrm{GPa}$  is the elastic modulus of tungsten,  $I=(\pi/2)r^4$  is the moment of inertia of the wire cross-section,  $L\approx 2\,\mathrm{mm}$  is the length of the wire, and K is the column effective length factor which depends on the boundary conditions and is set to K=1 here for simplicity. This suggests that it may be possible to push structures of < 10  $\mu$ m diameter into brain tissue (see [103] for related calculations). It might be advantageous to pull rather than push wires into the brain (e.g., using applied fields, or perhaps even cellular oxen [104] to carry the wires), since the thinnest wires could withstand tension forces much higher than the compressive force at which they buckle (although there may also be ways to circumvent buckling, e.g., via rapid vibration).

#### 4.1.4 CONCLUSIONS AND FUTURE DIRECTIONS

Electrical recording has the advantage of high temporal resolution, but the large number of required recording sites poses challenges for delivery mechanisms. Ongoing innovations in electrical recording that could be leveraged for dramatic scaling include the development of highly multiplexed probes, multilayer lithography for routing electrical traces, novel methods to implant large numbers of electrodes, smaller electrode impedances to reduce the Johnson noise, amplifiers with lower input-referred noise levels, spike sorting algorithms capable of handling temporally overlapping spikes and adaptively modeling the noise, and hybrid systems integrating electrical recording with implantable optics or other methods.

One challenge for a purely-electrical recording paradigm pertains to the ability to relate the measured electrical signals to specific cells within a circuit. As the set of neurons recorded by each electrode grows to encompass a large volume around the electrode, it will become more difficult to attribute the recorded spikes to particular neurons. Furthermore, given the complex geometries of neuronal processes, it is not obvious how to determine the spatial position or layout of a neuron from its electrical signature on a nearby electrode. A given electrode will be positioned near the axons or dendrites of some neurons, and near the cell bodies of other neurons, complicating data interpretation. If the spatial density of recording sites is increased such that many electrodes sample the same neuron, however, this could enable imaging of neuronal morphology and signal propagation via voltage signals across multiple electrodes [105]. Currently, extracellular electrical recording also does not allow extraction of molecular information on the cells being recorded, although intracellular electrophysiological recording methods (e.g., [106]) might enable this for a limited number of cells.

## 4.2 OPTICAL RECORDING

Optical techniques measure activity-dependent light emissions from neurons, typically generated by fluorescent indicator proteins, although activity-dependent bioluminescent emissions are an emerging possibility. Current genetically encoded calcium indicators can only distinguish spikes below ~50–100 Hz firing rates without averaging [107] due to slow intra-molecular kinetics and indicator saturation at high firing rates, although significant improvements in speed are ongoing [108]. Intracellular calcium rises and drops can occur within 1 ms and 10–100 ms respectively [109], which sets the ultimate speed limit for calcium imaging. The field of genetically-encoded high-speed fluorescent voltage indicators is also advancing quickly [110–115] and these may find particular use in monitoring sub-threshold events [116].

### 4.2.1 SPATIOTEMPORAL RESOLUTION

For optical approaches, the light originating from the activity of each neuron must be separated from Multiplexing strategies emissions originating from other points in the brain: this can be accomplished in many ways, leading to a variety of architectures for 3D imaging. Epi-fluorescence microscopy images a plane in the specimen (i.e., with depth of field DOF =  $\frac{2n\lambda}{NA^2}$ , where n is the refractive index,  $\lambda$  is the wavelength and NA is the numerical aperture of the imaging system [117]) onto a spatially-resolved twodimensional detector (e.g., a CCD camera). The focal plane is then scanned in order to reconstruct 3D images; because the entire 3D volume is illuminated during image acquisition, out-of-focus neurons cause background emissions. Light sheet imaging is similar to epi-flourescence imaging, except that only neurons near the focal plane are illuminated, reducing out of focus noise. Unfortunately, this requires transparent brains [8]. Volumetric imaging can also be performed in a single snapshot using lightfield microscopes [118, 119], which capture the directions of incoming light rays, trading in-plane resolution for axial resolution, or by using multi-focus microscopes [120]. In multi-photon microscopy, nonlinearities result in fluorescence excitation occurring only near the focal point of the excitation laser, which is scanned across the sample. In confocal scanning microscopy, only photons from a point of interest are measured due to geometric constraints (e.g., pinholes). Alternatively, 3D imaging can be performed via wavefront coding, which extends the depth of field by creating an axially-independent point-spread function using known optical aberrations, in combination with computational deconvolution [121]. With a known 3D pattern of excitation light, wavefront coding can be applied to 3D fluorescence microscopy without scanning using a 2D detector array [117]. Emerging, alternative strategies rely on tagging emissions from different sources with distinguishable modulation patterns [122-126], or precisely controlling and tracking the timing of light emissions [127]. Optical techniques thus achieve signal separation by multiplexing spatially (e.g., direct imaging) or temporally (e.g., beam scanning), or often by a combination of the two.

While optics might seem to require a number of photodetectors comparable to the number of neurons (or a similar number of sampling events in the time domain, e.g., for scanning microscopies), new developments suggest ways of imaging with fewer elements. For example, compressive sensing or ghost imaging techniques based on random mask projections [128–131] might allow a smaller number of photodetectors to be used. In an illustrative case, an imaging system may be constructed simply from a single photodetector and a transmissive LCD screen presenting a series of random binary mask patterns [132], where the number of required mask patterns is much smaller than the number of image pixels due to a compressive reconstruction.

Effects of light scattering Single-photon techniques limit imaging to a depth of a few scattering lengths at the excitation and emission wavelengths of activity indicators: up to ~1–2 mm for certain infrared wavelengths [39, 133, 134] vs. a few hundred microns for visible wavelengths [38]. Activity dependent dyes are currently available only in the visible spectrum; indicators operating in the infrared (see [135–137] for far-red fluorescent proteins) could improve imaging depth.

Multi-photon excitation takes advantage of the deeper penetration of infrared light. Two or more infrared photons may together excite a fluorophore with an excitation peak in the visible range, leading to the emission of a visible photon. If only one neuron is illuminated with sufficient intensity to generate multi-photon excitation, all photons captured by the detector originate from that neuron, regardless of the scattering of the outgoing light. Hence, the emission pathway is limited less by scattering than by absorption. This has resulted in imaging at > 1 mm depth [39, 133, 134].

There are at least five options for overcoming visible light scattering to enable signal separation from deep-brain neurons [1, 138]:

- 1. Infrared light can excite multi-photon fluorescence in an excitation-scanning architecture.
- 2. Fluorophores with both excitation and emission wavelengths in the infrared could be developed.
- 3. By knowing the precise form of the scattering, it can be possible to correct for it. Emerging techniques based on beam shaping allow transmission of focused light through random scattering media by inverting the scattering matrix [139]. Because the scattering properties change over time, this must be done quickly, possibly faster than the imaging frame rate, necessitating high-speed wavefront modulation. This can currently be achieved with digital micro-mirror devices (DMDs), but not with the phase-only spatial light modulators (SLMs) that are used to prevent power losses in the excitation pathways for nonlinear microscopies, although GHz switching of phase-only modulators appears feasible in principle [138]. High speed focusing through turbid media is also achievable using all-optical feedback in a laser cavity [140], and it is even possible to measure the scattering matrix non-invasively [141] using a photo-acoustic technique, or via all-optical approaches based on speckle correlation [142]. Similar techniques are available for incoherent light [143].

When using short optical pulses, scattering can lead to temporal distortions that degrade the peak light intensity at a focal spot. The < 100 fs pulse durations used in two-photon microscopy, for example, are comparable to the time it takes light to travel 30  $\mu$ m in vacuum. Fortunately, wavefront shaping techniques can correct for scattering-induced temporal distortions as well [144, 145].

- 4. Light sources and/or detectors could be positioned close to the measured neurons, necessitating the use of embedded optical devices. This could be done using optical fiber [146] and/or waveguide [147, 148] technologies, which are developing rapidly. For example, single-mode fiber cables can support > 1 TB/s data rates [149, 150] with low light loss over hundreds of kilometers [151]. It is possible to directly image through gradient index of refraction (GRIN) lenses [152] or optical fibers [146, 153, 154], which provides one way to multiplex multiple observed neurons per fiber.
- 5. Light emissions from distinct locations can be tagged with distinguishable time-domain modulation patterns, and the emission time-series for each source can later be decoded from the summed signal resulting from scattering [122–127]. For example, ultrasound encoding [122], which frequency-tags light emissions from a known location via a mechanical Doppler shift of the emitter [155], provides a generic mechanism to sidestep problems of elastic optical scattering, although it requires distinguishing MHz frequency modulations in THz light waves (part per million frequency discrimination). Radio-frequency tagging of light emissions via a digitally synthesized optical approach is also an option and may be applicable to combatting the problem of emission scattering in deep-tissue, multi-point, multi-photon imaging [123].

Speed of beam scanning The speed of scanning microscopes is currently limited by beam repositioning times ( $\sim$ 0.1  $\mu$ s for spinning disk [146, 153, 154],  $\sim$ 3  $\mu$ s for piezo-controlled linear scan mirrors,  $\sim$ 10  $\mu$ s for acousto-optic deflectors [156],  $\sim$ 8 kHz line scans for resonant galvanometer mirrors). The 10  $\mu$ s repositioning time for acousto-optic deflectors is set by the speed of sound in the deflector crystal, while scanning mirrors and spinning disks are limited by inertia. Note that 0.1  $\mu$ s repositioning time for current spinning-disk confocal techniques would require 10 seconds per frame for whole mouse brain imaging with a single scanned beam (10<sup>-7</sup> s/site  $\times$  10<sup>8</sup> sites/brain). There is therefore a need for a 10<sup>4</sup> fold improvement in beam repositioning time and/or beam parallelization in order to achieve 1 kHz imaging frame rates for whole mouse brains.

One strategy to implement parallelization would exploit (yet to be developed) fast, high-resolution phase modulator arrays to arbitrarily re-shape coherent optical wavefronts for multisite holographic multi-photon excitation in 3D [138, 157, 158]. With fast phase modulation (e.g., ~1 GHz), beating each excitation spot at a different frequency could allow a single detector to probe multiple sites in parallel, despite arbitrarily-large scattering of the outgoing light [138]. Emerging optical techniques may provide alternative means to implement similar strategies [123]. Temporal multiplexing of excitation pulses at distinct locations (e.g., via few-nanosecond beam delays) also allows parallelization of the excitation beam while combatting scattering ambiguity of the emitted light [127]. Furthermore, temporal focusing techniques in two-photon microscopy (depth-dependent pulse duration) can excite an entire plane or line

within the sample [159–162], as well as arbitrary patterns of points [157], potentially allowing fast axial scanning (somewhat analogous to light-sheet techniques used with transparent samples). This method intrinsically corrects for scattering of the excitation light [163], although not of the emission light. Like other multi-photon techniques, however, all these methods remain highly dissipative, as discussed below.

Fluorescence lifetimes in the 0.1–1 ns range [164] ultimately constrain the design of scanning fluorescence microscopies. A delay of 0.1 ns per mouse neuron per frame corresponds to only 100 Hz frame rate without parallelization, implying that parallelization into at least 10 to 100 beams is essential. The fluorescence lifetime also limits the achievable modulation frequencies in beat-frequency-multiplexed parallelization strategies [123], bit lengths in encoded strategies [124], and temporal offsets in temporally-multiplexed strategies [127], suggesting that parallelization of detectors may be necessary in a strongly scattering environment. Depending on the degree of parallelization, which constrains the achievable dwell times given a fixed frame rate, photon counts may also become a limiting factor for high-speed scanning in some approaches.

**Diffraction** Using the small angle approximation, the diffraction-limited angular resolution of an aperture is  $\theta \approx \frac{\Delta x}{y} \approx \frac{\lambda}{D}$ , where  $\Delta x$  is the spacing which must be resolved, y is the imaging depth,  $\lambda$  is the wavelength, and D is the aperture diameter. Thus distinguishing neurons which are 10  $\mu$ m apart and at a depth of 10 mm requires a lens aperture D of D 1 mm when D 1  $\mu$ m. Diffraction therefore does not appear to be a limiting factor for cellular resolution imaging, except in the context of microscale apertures that might find use in embedded optics approaches.

#### 4.2.2 ENERGY DISSIPATION

Light that does not leave the brain is ultimately dissipated as heat. The total light power requirements for optical measurement of neuronal activity using fluorescent indicators depend on factors including fluorophore quantum efficiency, absorption cross-section, activity-dependent change in fluorescence, background fluorescence, labeling density, activation kinetics, detector noise, scattering and absorption lengths, and others. Unfortunately, many of these variables are unknown or highly dependent on particular experimental parameters.

A statistical analysis of photon count requirements for spike detection (in the context of calcium imaging) can be found in [165], which derived a relationship between the number of background photon counts ( $N_{\rm bg}$ ) and the number of signal photon counts required for high fidelity spike detection given photon shot noise. This scales roughly as  $N_{\rm signal} > 3\sqrt{2N_{\rm bg}}$ , even at low absolute photon count rates. While this analysis governs the number of detected photons, the number of emitted photons will be higher due to losses. In one example using two-photon excitation, 5% of the emitted photons were captured by the photodetector [166]. One implication of photon shot noise is that faster-responding indicators (e.g., voltage indicators which respond in near-real-time to the membrane potential) must be brighter.

Multi-photon excitation Multi-photon experiments rely on short laser pulses with high peak light intensities at a focused excitation spot to excite nonlinear transitions [166]. This imposes an experimentally relevant physical limit: at least one excitation pulse of sufficient intensity per neuron per frame is required in order to excite multi-photon fluorescence during each frame. Assuming 1 kHz frame rate and 0.1 nJ pulses [127], delivering only one pulse per neuron per frame would dissipate roughly  $(10^8 \times 1 \, \text{kHz} \times 0.1 \, \text{nJ}) \, 10 \, \text{W}$  in the mouse brain, which is clearly prohibitive. This is a lower bound because, in general, more than one excitation pulse per neuron per frame may be required to excite detectable fluorescence (e.g., one reference reported 12 pulses per spot [166]). For three-photon excitation, the situation will be even worse as higher peak light intensities are required to excite three-photon fluorescence.

Could the single-pulse energy be reduced while maintaining efficient two-photon excitation? The number of two-photon (2P) transitions excited per fluorophore per pulse is  $n_a = F^2 C/t$ , where F is the number of photons per pulse per area in units of photon/cm<sup>2</sup>, C is the two-photon cross-section in units of cm<sup>4</sup>s/photon, and t is the pulse duration in seconds. This can be approximated as

$$n_{a} = \left(\frac{\frac{E}{hc/\lambda}}{\left(\frac{\lambda}{2(NA)}\right)^{2}}\right)^{2} \frac{C}{t} = \left(\frac{4E(NA)^{2}}{hc\lambda}\right)^{2} \frac{C}{t}$$

where NA is the numerical aperture of the focusing optics, E is the pulse energy and  $\lambda$  is the stimulation wavelength. For a 2P experiment with 100 fs, 0.1 nJ pulses, assuming a 2P cross section [167, 168] of  $10^{-48}$  cm<sup>4</sup>s/photon (i.e., 100 Goeppert-Mayer units [169], comparable to that of DsRed2 [168]),  $\lambda = 900$  nm and NA = 1.0,  $n_a \approx \frac{1}{2}$ . Thus, a few pulses are likely necessary and sufficient to excite 2P fluorescence by each fluorophore within the focal spot. With a 2P cross section above  $10^{-47}$  cm<sup>4</sup>s/photon (1000 Goeppert-Mayer units, higher than that of any fluorescent protein that we are aware of [168]), one could reduce the pulse energy by an order

of magnitude (and hence  $n_a$  by two orders of magnitude) while maintaining  $n_a > \frac{1}{20}$ , i.e., one in twenty fluorophores excited by each pulse. Reducing the pulse energy much further might lead to unacceptably low excitation levels. Alternatively, shorter pulse durations could increase the light intensity, and hence 2P excitation probability, at fixed pulse energy.

Quantum dots can have 2P cross sections much higher than those of fluorescent proteins: water-soluble cadmium selenide-zinc sulfide quantum dots have been reported with 2P cross sections of 47000 Goeppert-Mayer units and are compatible with in-vivo imaging [170]. These would allow excitation efficiencies of  $n_a > \frac{1}{20}$  at pJ pulse energies, bringing whole-brain 2P imaging into the ~100 mW range. Thus, the use of quantum dots or other ultra-bright multi-photon indicators could be decisive for supporting the energetic feasibility of multi-photon methods at whole brain scale; there are also plausible strategies for coupling quantum dot fluorescence to neuronal voltage [171]. However, some quantum dots have long fluorescence lifetimes [172], which may constrain scan speed.

For comparison to current practice, in a typical multi-photon experiment on mice,  $\sim$ 50 mW of time-averaged laser power at the sample was used with a dwell time of  $\sim$ 3 µs [173], corresponding to  $\sim$ 150 nJ energy dissipation per spot per frame. This dwell time would allow imaging only  $\sim$ 300 neurons at millisecond resolution with a single scanned excitation beam. The average excitation power here is likely already close both to whole-brain thermal dissipation limits, and to photo-damage limits for pulsed two-photon excitation [174, 175].

#### 4.2.3 BIOLUMINESCENCE

To work around the requirement for large amounts of excitation light, bioluminescent rather than fluorescent activity indicators could be used [176–178]. Consider a hypothetical activity-dependent bioluminescent indicator emitting at ~1700 nm (IR), in order to evade light scattering. As a crude estimate, assuming that 100 photons must be collected by the detector per neuron per 1 ms frame, and 1% light collection efficiency by the detector relative to the emitted photons, ~100  $\mu$ W of bioluminescent photons emissions are required for the entire mouse brain (using  $E_{\rm photon} = hc/\lambda$ ). This would be feasible from the perspective of heat dissipation. By contrast, in a 1-photon fluorescent scenario, if 100 excitation photons must be delivered into the brain to generate a single fluorescent emission photon, the power requirement becomes 10 mW, which is on the threshold of the steady-state heat dissipation limit. Therefore, bioluminescent indicators could potentially circumvent problems of heat dissipation even in the 1-photon case.

The widely used bioluminescent protein firefly luciferase is  $\sim$ 80% efficient in converting ATP hydrolysis coupled with luciferin oxidation into photon production, yielding  $\sim$ 0.8 photons per ATP-luciferin pair consumed [179], and has  $\sim$ 90% energetic efficiency in converting free energy to light production. Heat dissipation associated with the luciferase biochemistry itself is therefore not a significant overhead relative to the 100  $\mu$ W of emitted photons calculated above. In the same scenario, however, each neuron would consume  $\sim$ 6 × 10<sup>8</sup> additional ATP molecules per minute in order to power the bioluminescence, which is within the limits of cellular aerobic respiration rates ( $\sim$ 1 fmol O<sub>2</sub> per minute per cell [180], with  $\sim$ 30 ATP per 6 O<sub>2</sub>, hence 3 × 10<sup>9</sup> molecules ATP synthesized per minute from ADP via glucose oxidation), but not by a large margin. Transient increases in metabolic rate are possible: energy dissipation more than doubles in the mouse during high physical activity [17]. Therefore, whole-brain activity-dependent bioluminescence, at speeds high enough to achieve millisecond frame rates, may be metabolically taxing for the cell but is nevertheless plausible as a light generation strategy. Note that we have not treated the energy required to bio-synthesize the luciferin compound, which may create additional overhead (though conceivably luciferin could be provided exogenously).

### 4.2.4 CONCLUSIONS AND FUTURE DIRECTIONS

Scattering of visible light in the brain creates a problem of signal-separation from deep-brain neurons. Multi-photon techniques, which scan an infrared excitation beam, can work around this scattering problem. However, current multi-photon techniques using fluorescent protein indicators, when applied at whole brain scale, would dissipate too much power to avoid thermal damage to brain tissue. Systems (such as plasmonic nano-antennas [181] or subwavelength metallic gratings [182]) that could locally excite multiphoton fluorescence without the need for high-energy laser pulses could conceivably ameliorate this issue. Importantly, quantum dots show promise as ultra-bright multi-photon indicators, if they can be targeted to neurons and optimized in terms of fluorescence lifetime. New methods besides multi-photon techniques could also work around the scattering of visible light in the brain. For example, fluorophores or bio-luminescent proteins could be developed which operate at infrared wavelengths. A compelling example from nature is the black dragonfish, which generates far red light (~705 nm) via a multi-step bioluminescent process (using this light to see in deep ocean waters) [183, 184]. A large set of activity indicators with distinguishable colors, generated through a combinatorial genetic recombination mechanism such as BrainBow [185], could also improve signal separation. Targeting, via protein tags, of activity indicators to specific locations — such as the axon, soma, soma and proximal dendrites, distal dendrites, pre-synaptic terminals, post-synaptic terminals, or intact synapses — could also aid in signal discrimination [186-193]. In addition, implanted optical devices, which place emitters and detectors within a few scattering lengths of the neurons being probed, could potentially obviate the negative effects of scattering and allow visible-wavelength indicators to be used without a need for multi-photon excitation. In principle, excitation and detection do not need to make use of the same modality. For example, photoacoustic microscopy [194] uses pulsed laser excitation to drive ultrasonic emission, leading to optical absorption contrast. Such asymmetric techniques impose

fundamentally different requirements from pure-optical techniques relative to fluorophore properties, required light intensities and other parameters.

## 4.3 EMBEDDED ACTIVE ELECTRONICS

The preceding sections have assumed that electrical or optical signals from the recorded neurons are shuttled out of the brain before digitization and storage, but it is also conceivable to develop embedded electronic systems that locally digitize and then store or transmit (e.g., wirelessly) measurements of the activities of nearby neurons. This could allow for shorter wires in electrical recording approaches, and for shorter light path lengths in optical recording approaches, as well as for more facile (e.g., non-surgical) delivery mechanisms for the recording hardware.

Integrated circuits have shrunk to a remarkable degree: in about 3 years, following the Moore's law trajectory, it will likely be possible to fit the equivalent of Intel's original 4004 micro-processor in a  $10 \,\mu\text{m} \times 10 \,\mu\text{m}$  chip area. Functional wirelessly powered radio-frequency identification (RFID) chips as small as  $50 \,\mu\text{m}$  in diameter have been developed [195] and tags with chip-integrated antennas function at the  $400 \,\mu\text{m}$  scale [196]. Integrated neural sensors including analog front ends are also scaling to unprecedented form factors: a  $250 \,\mu\text{m} \times 450 \,\mu\text{m}$  wireless implant – including the antenna, but not including a  $\sim 1 \,\text{mm}$  electrode shank used to separate signal from ground – draws only  $2.5 \,\mu\text{W}$  per recording channel [197]. The system operates at  $\sim 1 \,\text{mm}$  range in air, powered by a transmitter generating  $\sim 50 \,\text{mW}$  of transmitted power. Note that for a single such embedded recording device, the heat dissipation constraint is set not by the device's own dissipation ( $10 \,\mu\text{W}$  for four recording channels) but rather by the RF specific absorption rate limit associated with the  $50 \,\text{mW}$  transmit power.

Possibilities may exist for non-surgical delivery of embedded electronics to the brain: remarkably, cells such as macrophages (~13 µm in size) can engulf structures up to at least 20 µm in diameter [198] and have been studied as potential delivery vehicles for nanoparticle drugs [199], suggesting that they might be used to deliver tiny microchips. T-cells and other immune cells can trans-migrate across the blood brain barrier [200] and ghost cells (membranes purged of their contents) engineered to encapsulate synthetic cargo [201] can fuse with neurons [202]. It might even be possible to engineer such cell-based delivery vehicles to form electrical gap junctions [203] with neurons or to act as local biochemical sensors [204].

The real-time transmission bandwidth requirements for neural recording could be significantly reduced if it is only desired to take a "snapshot" of neural activity patterns over a limited period of time, but this would require a large amount of local storage. For example, flash memory can store > 10 Mbit of data in a device 100  $\mu$ m on a side: a 64 giga-byte microSD card with 1.5 cm<sup>2</sup> area corresponds to 34 mega-bits per  $(100\,\mu\text{m})^2$  area. Even denser forms of memory storage are under development and could perhaps be used in a one-time-write mode in the context of neural recording long before they become commercially viable for use as rewritable media in the electronics industry.

Here we consider the power dissipation associated with embedded electronic recording devices, as well as the constraints on possible methods to power them. In the next section, we describe how physics constrains the data transmission rates from such devices.

## 4.3.1 POWER REQUIREMENTS FOR RECORDING

Any embedded system needs to process data, in preparation for either local storage or wireless transmission. Physics defines hard limits on the required power consumption associated with data processing (neglecting the possibility of reversible logic architectures [205]), arising from the entropy cost for erasing a bit of information [206]:

$$E_{\text{Landauer}} = \ln(2) k_{\text{B}} T \approx 3 \times 10^{-21} \text{ J/bit}$$
 (the Landauer limit)

Ambitious yet physically realistic values for beyond-CMOS logic lie in the tens of  $k_{\rm B}T$  per bit processed [207]. Scaling 40  $k_{\rm B}T$ /bit to record raw voltage waveforms at a minimal 1 kbit/s/neuron (e.g. 1 kHz sampling rate, 1 bit processed per neuron per sample), the total power consumption for whole mouse brain recording could in principle be as low as ~16 nW. While this leaves > 10<sup>6</sup>-fold more room (energetically) for increased data processing (more required bit flips per second), or energetic inefficiency of the switching device (greater dissipation per bit), realistic devices in the near-term may in fact require this much overhead, if not more. This necessitates a more detailed consideration of limiting factors for today's microelectronic devices.

In the context of electrical recording, the first step that must be performed by an embedded neural recording device is digitization of the voltage waveform. Until mV-scale switching devices are developed (see discussion below), it is necessary to amplify the  $\sim$ 10–100  $\mu$ V spike potential in order to drive digital switching events in downstream gates. During this sub-threshold amplification step, a CMOS (or BJT) device will dissipate static power (associated with a bias current). Importantly, in order to decrease the input-referred

voltage noise of this amplification process, it is necessary to increase the bias current and hence the static power dissipation. For a simple differential transistor amplifier, the minimal bias current scales as

$$I_{\rm d} = \frac{\pi}{2} \frac{4k_{\rm B}T}{V_{\rm noise}^2} \frac{k_{\rm B}T}{q} \text{BW}$$

where  $V_{\text{noise}}$  is the input-referred voltage noise of the amplifier and q is the electron charge. For an extracellular recording with BW = 10 kHz and  $V_{\text{noise}} = 10 \,\mu\text{V}$ , this implies a minimal bias current  $I_{\text{d}} \approx 60 \,\text{nA}$  or a minimal static power of  $\left(I_{\text{d}}V_{\text{dd}}\right) \approx 6 \times 10^{-8} \,\text{W}$  at  $V_{\text{dd}} \approx 1 \,\text{V}$  operating voltage. Assuming 10 neurons per recording channel, there are then 7.5 million recording channels for a mouse brain, which gives a power dissipation associated with signal amplification of  $\sim 500 \,\text{mW}$ . Note that realistic analog front ends (which are subject to 1/f noise and require multiple gain stages) draw  $6 \times -10 \times$  greater bias current, quantified by the noise efficiency factor (NEF) [208], to achieve the same input-referred noise levels.

Local on-chip digital computation also incurs an energy cost. Current CMOS digital circuits consume 5–6 orders of magnitude [207, 209–211] more energy per switching event (~1 fJ/bit including charging of the wires [209]) compared to the Landauer limit (e.g., for a digital CMOS inverter, and ignoring the static power associated with the leakage current). This corresponds to a ~1 fF total load capacitance at 1 V operating voltage. For 100 GHz switching rates (10<sup>8</sup> neurons × 1 kHz) as above, this corresponds to 0.01–0.1 mW. Realistic architectures, however, will incur overhead in the number of switching events required to store, compress and/or transmit neural signals, likely bringing the power consumption into an unacceptable range (e.g., 1000 bits processed per sample would be 100 mW here). To take a concrete example, commercial RFID tags consume ~10 µW [212]. At a chip rate of 256 kbit/s (with a Miller encoding of 2), this yields 7.8 × 10<sup>-11</sup> J/bit, which is ~10 orders of magnitude higher than the Landauer limit. Applying current RFID technology to whole mouse brain recording at 1 kbit/s/neuron would thus draw ~8 W of power. Therefore, at least 2–3 orders of magnitude reduction in power consumption will be necessary in order to apply embedded electronics for whole-brain neural recording.

Until recently, the energy efficiency of digital computing has scaled on an exponential improvement curve [210]. This was a consequence of Moore's law and Dennard scaling, where both the capacitance of each transistor and its associated interconnect, as well as the operating voltages, were reducing with the device dimensions. Unfortunately, issues related to device variability and the 3D structures needed to maintain the on-to-off current ratio have largely stopped the reduction in effective capacitance per device; current devices are stuck at  $\sim 100-200$  aF for a minimum sized transistor. Furthermore, the exponential increase in leakage current that comes along with the scaling of the threshold voltage in this scenario has precluded substantial further decreases in voltage at a given performance level. Indeed, for the past several technology generations (since about 2005), CMOS devices have operated at a supply voltage of  $\sim 1$  V.

While neural signal processing does not demand very stringent transistor speeds and so reductions below ~1 V are certainly feasible, a fundamental limitation in scaling the supply voltage still remains. Specifically, CMOS has a well-defined minimum-energy per bit and an associated minimum-energy operating voltage that is defined by the tradeoff between static (leakage) and dynamic (switching) energy: as the operating voltage is decreased, the capacitive switching energy decreases, but the ratio of currents in the on and off states,  $I_{\rm off}/I_{\rm on}$ , increases exponentially, increasing the energy associated with leakage (this effect is independent of the threshold voltage in the sub-threshold regime). For practical circuits, the supply voltage that leads to this minimum energy is on the order of 300–500 mV, and thus supply voltage scaling will at most provide  $3\times-10\times$  improvement in energy over today's designs.

Thus, a paradigm shift in microelectronic hardware is needed to reduce power by several orders of magnitude if we are to approach the physical limits. Developing a switching device operating in the mV range, rather than the 1 V range of current transistors, would allow  $(1 \text{V}/1 \text{mV})^2 = 10^6$  fold reduction in power consumption [207]. Electronic circuits constructed using analog techniques [213], which sometimes rely on bio-inspired computational architectures, show promise for reducing energy costs by up to five orders of magnitude [213–215], depending on the nature of the computation and the required level of precision.

Figure 4 shows the power consumption per bit processed for several technology classes as well as the corresponding total power consumption required for whole brain readout, assuming a minimal whole-brain bit rate of 100 Gbit/s.

## 4.3.2 POWERING EMBEDDED DEVICES

Embedded systems need power, which could be supplied via electromagnetic or acoustic energy transfer, or could be harvested from the local environment in the brain.

There are two key regimes for wireless electromagnetic power transfer: non-linear device rectification and photovoltaics. If the single-photon energy is sufficient to allow electrons to move from the valence to the conduction band — that is, band gap  $< h\nu/q$ , where q is the electron charge, h is Planck's constant, and  $\nu$  is the frequency of the photon — a photovoltaic effect can occur. Otherwise, electromagnetic energy is converted to voltage by an antenna and non-linear device rectification may occur.

Figure 4. Energy cost of elementary operations across a variety of recording and data transmission modalities, expressed in units of the thermal energy (left axis) and as a power assuming 100 GHz switching rate (right axis). The Landauer limit of  $k_BT \ln 2$  sets the minimum energy associated with a logically irreversible bit flip. The practical limit will likely lie in the tens of  $k_BT$  per bit [207], comparable to the free energy release for hydrolysis of a single ATP molecule (or addition of a single nucleotide to DNA or RNA). The energy of a single infrared photon is ~50  $k_BT$ . Single gates in current CMOS chips dissipate ~1 × 10<sup>5</sup>-10<sup>6</sup>  $k_BT$  per switching event, including the capacitive charging of the wires interconnecting the gates (red curve). The switching energy for the gate, not including wires, is ~100× lower (blue curve). The power efficiency of CMOS has been on an exponential improvement trend due to the miniaturization of components according to Moore's law (data re-digitized from [209]), although power efficiency gains have slowed recently. Current RFID chips compute and communicate at ~1 × 10<sup>9</sup>-10<sup>10</sup>  $k_BT$  (> 10 pJ) per bit transmitted, while the total energy cost per floating point operation in a 2010 laptop was ~1 × 10<sup>12</sup>  $k_BT$ . The power associated with a minimal low-noise CMOS analog front end for signal amplification corresponds to ~500 mW at whole mouse brain scale. A single two-photon laser pulse at 0.1 nJ pulse energy corresponds to ~1 × 10<sup>10</sup>  $k_BT$ . For comparison, the 40 mW approximate maximal allowed power dissipation, according to section 2 (Basic Constraints) above, with its equivalent per-bit energy of ~1 × 10<sup>8</sup>  $k_BT$  at the minimal 100 Gbit/s bit rate.

![](_page_17_Figure_2.jpeg)

When photon energies are much lower than the band gap, power conversion is governed by the total RF power and by the impedances of the antenna and the rectifier, rather than by the individual photon energy. For a monochromatic RF source, there is no thermodynamic or quantum limit to the RF to DC conversion efficiency, other than the resistive losses and threshold voltages for a semi-conductor process. For rectification, when the input voltage to the rectifier is much higher than a semiconductor process threshold, conversion efficiencies of 85 % have been achieved [216]. At low input voltages relative to the semiconductor process threshold, efficiencies as high as 25 % and 2 µW load have been achieved (see [215] for an analysis of power efficiency). Ultimately, rectification improvements are dependent on the same improvements which will be needed for next-generation low-power computing: mV scale switching devices (promising research directions include tunnel FETs [217], electromechanical relays [218] and other options).

While efficient rectification is thus not a fundamental issue, capturing sufficient RF energy in the first place becomes increasingly challenging as microchips become smaller and more deeply embedded in tissue. Wireless electromagnetic power transfer imposes range constraints due to the loss in power density with distance. For directional power transfer, placing the receiver at the edge of the transmitter's near field (the Rayleigh distance  $\frac{D^2}{4\lambda}$  where D is the transmitter aperture) has advantages in terms of energy capture efficiency [219], whereas for omni-directional antennas it is advantageous to place the receiver as close as possible to the transmitter. If embedded chips are oriented randomly with respect to the transmitter, the radiation patterns of their antennas cannot be highly directional, i.e., their gains  $G_r$  (a measure of directionality) must be close to one. In the far field, this lack of directionality limits power capture by the antenna (due antenna reciprocity [220]): the maximal power  $P_A$  available to the chip is

$$P_A = \frac{G_r P_{\rm rad} \lambda^2}{4\pi}$$

where  $P_{\rm rad}$  is the power density of radiation around the antenna,  $\lambda$  is the wavelength and  $G_r \approx 1$  for a non-directional antenna [215].

It may be possible to power devices with pure magnetic fields (which are highly penetrant) via near-field (non-radiative) inductive coupling, which is widely used in systems ranging from biomedical implants to electric toothbrushes, or conceivably by using magneto-electric materials [221–224]. For the case of simple inductive coupling, however, the tiny cross-sections of micro-devices limit the amount of power which can be captured: a loop of 10  $\mu$ m diameter in an applied field of 1 T switching at 1000 Hz produces an induced electromotive force of only 0.1  $\mu$ V. Assuming a copper loop ( $\sim$ 17  $n\Omega$  m resistivity) with 1  $\mu$ m  $\times$  1  $\mu$ m cross-section and 40  $\mu$ m length (around the outer edge of the chip) gives a power ( $V^2/R$ ) of only  $\sim$ 15 fW associated with the induced current. In general, the use of coupled high-Q resonators can increase the range and efficiency of near-field electromagnetic power transfer by orders of magnitude [225] compared to non-resonant inductive power transfer and may be particularly relevant for implanted devices [226]. Unfortunately, at the  $\sim$ 10  $\mu$ m length scale, the achievable on-chip inductances and capacitances are severely limited, which restricts the operating range of any resonant device to high frequencies  $\left(f_{\rm resonant} = \left(2\pi\sqrt{LC}\right)^{-1}\right)$  which will be attenuated by tissue. Electromagnetic near-field power transfer though tissue to ultra-miniaturized microchips may thus be inefficient, again due to low capture efficiency of the applied fields by tiny device cross-sections.

Alternatively, if the photon energy is above the silicon band gap ( $\lambda < \frac{bc}{qV_{\rm th}} \approx 3\,\mu m$  or less for silicon), the chip is essentially acting as a photovoltaic cell. There is no thermodynamic or quantum limit to the conversion efficiency of light to DC electrical power for monochromatic sources, other than resistive losses and dark currents in the material (86% in GaAs for example [227]). Again, however, capturing sufficient light becomes difficult for tiny devices. To supply 10  $\mu$ W (typical of current wirelessly-powered RFID chips) photovoltaically to a 10  $\mu$ m × 10  $\mu$ m (cell sized) chip at 34% photovoltaic efficiency requires a light intensity of ~300 kW/m² at the chip, which is prohibitive. Furthermore, in the use of infrared light for photovoltaics, the penetration of the photons through tissue is decreased compared to radio frequencies.

Piezoelectric harvesting of ultrasound energy by micro-devices is a possibility [70]. The efficiency of electrical harvesting of mechanical strain energy in piezoelectrics can be above 30 % for materials with high electromechanical coupling coefficients (e.g., PZT) [228, 229]. The losses in the piezoelectric transduction process are well described by models such as the KLM model [230, 231].

An alternative to wireless energy transmission is the local harvesting of biochemical energy carriers. Implanted neural recording devices could conceivably be powered by free glucose, the main energy source used by the brain itself. The theoretical maximum thermodynamic efficiency for a fuel cell in aqueous solution is equal to that of the hydrogen fuel cell:  $\Delta G^0/\Delta H^0 = 83\%$  at 25 °C. Furthermore, if glucose is only oxidized to gluconic acid, the Coulombic (electron extraction) efficiency is at most 8.33 % [232], which bounds the thermodynamic efficiency. The blood glucose concentration in rats has been measured at ~7.6 mM, with an extracellular glucose concentration in the brain of ~2.4 mM [233]. A hypothetical highly miniaturized neural recorder with a device area of 25  $\mu$ m × 25  $\mu$ m and efficiency of 80 %, processing a blood flow rate of ~1 mm/s [234] could extract (80%)(7.6 mM)(25  $\mu$ m)<sup>2</sup>(1 mm/s)(2880 kJ/mol)  $\approx$  11  $\mu$ W, which is sufficient for low-power device such as RFID chips [235]. Unfortunately, current non-microbial glucose fuel cells obtain only ~180  $\mu$ W/cm<sup>2</sup> peak power and ~3.4  $\mu$ W/cm<sup>2</sup> steady state power [232]. Thus there is a need for 10<sup>4</sup>- and 10<sup>6</sup>-fold improvements in peak and steady state power densities, respectively, for non-microbial glucose fuel cells to power brain-embedded electronics of the complexity of today's RFID chips (or for the corresponding decrease in power requirements, as emphasized above).

#### 4.3.3 CONCLUSIONS AND FUTURE DIRECTIONS

The power consumption of today's microelectronic devices is more than six orders of magnitude higher than the physical limit for irreversible computing, and 2–3 orders of magnitude higher than would be permissible for use in whole brain millisecond resolution activity mapping, even under favorable assumptions on the required switching rates and neglecting both the power associated with noise rejection in the analog front end and the CMOS leakage current. Thus, the first priority is to reduce the power consumption associated with embedded electronics. In principle, methods such as infrared light photovoltaics, RF harvesting via diode rectification, or glucose fuel cells, could supply power to embedded neural recorders, but again, significant improvements in the power efficiency of electronics are necessary to enable this. Other potential energy harvesting strategies include materials/enzymes harnessing local biological gradients such as in voltage, osmolarity, or temperature. An analysis of the energy transduction potential of each of these systems is beyond the scope of this discussion. Fortunately, with many orders of magnitude potential for improvement before physical limits are reached, we may expect that embedded nano-electronic devices will emerge as an energetically viable neural interfacing option at some point in the future.

## 4.4 EMBEDDED DEVICES: INFORMATION THEORY

Most recording methods envisioned thus far rely on the real-time transmission of neural activity data out of the brain. Physics and information theory impose fundamental limits on this process, including a minimum power consumption required to transmit data through a medium. The most basic of these results hold irrespective of whether the data transmission is wired or wireless, and regardless of the particular physical medium (optical, electrical, acoustic) used as the information carrier.

A communication "channel" is a set of transmitters and receivers that share access to a single physical medium with fixed bandwidth. The bandwidth is the range of frequencies present in the time-varying signals used to transmit information. In wireless communications, information is transmitted by modulating a carrier wave. To allow modulation, the frequency of the carrier wave must be higher than the bandwidth: for example, a 400 THz visible light wave may be modulated at a 100 GHz rate. The physical medium underlying a channel could be a wire (with a bandwidth set by its capacitive RC time constant), an optical fiber, free space electromagnetic waves over a certain frequency range, or other media.

As a concrete example, consider a police department with 100 officers, each possessing a hand-held radio. The radios transmit vocalizations by modulating an 80 MHz carrier wave at  $\sim$ 10 kHz. This constitutes a single shared communications channel with 10 kHz bandwidth. Simultaneously, the fire department may communicate via a separate channel, also with a bandwidth of  $\sim$ 10 kHz, by modulating a 90 MHz carrier wave. The channels are separate because modulation introduced into one does not affect the other. If the neighboring town's police department makes the mistake of also operating at 80 MHz carrier frequency, then they share a channel and conflicts will arise.

#### 4.4.1 POWER REQUIREMENTS FOR SINGLE-CHANNEL DATA TRANSMISSION

We first treat the case in which there is a single channel for transmitting data out of the brain. As discussed above in the context of electrical spike sorting, the Shannon Capacity Theorem [90] sets the maximal bit rate for a channel (assuming additive white Gaussian noise) to

$$R_{\text{max}} = BW \log_2 (1 + SNR)$$

where BW is the channel bandwidth and SNR is the signal-to-noise ratio. If there is only thermal noise the SNR =  $P/(N_0BW)$ , where  $N_0$  is the thermal noise power spectral density of  $k_BT$  W/Hz and  $P = (PL)P_0$  is the power of the transmitted signals  $P_0$ , weakened by path loss PL. Therefore the transmitted power  $P_0$  is lower-bounded:

$$P_0 > k_{\rm B}T$$
 BW  $\frac{2^{R_{\rm max}/{\rm BW}} - 1}{{
m PL}}$ 

as shown in Figure 5 (bottom). In a minimal model of a transmitter-receiver system, there thus exists a tradeoff between the required signal power and the bandwidth of the carrier radiation, due to the thermal noise floor, even in the absence of path loss (PL = 1).

Path loss weakens the proportion of the power that can reach the detector. Using the above equation, we can calculate, as a function of bandwidth, the power necessary to transmit a target whole-brain bit rate of 100 Gbit/s through a medium with path loss dependent on the carrier wavelength, as shown in Figure 5 (top).

For RF wavelengths, the radiation penetrates deeply but the achievable data rates are low without excessive power consumption, due to the limited bandwidth. For wavelengths intermediate between RF and infrared, the penetration depth is low and power must be expended to combat these losses, despite the high carrier bandwidth. Only in the infrared and visible ranges do the tradeoffs between

Figure 5. Power requirements imposed by information theory on data transmission through a single (additive white Gaussian noise) channel with carrier frequency ν (an upper bound on the bandwidth), given thermal noise and path loss. Bottom: absorption length of water as a function of frequency (blue), minimal power to transmit data at 100, 1000 and 10 000 Gbit/s (green) as a function of frequency, assuming thermal noise but no path loss. Top: minimal power to transmit data at 100, 1000 and 10 000 Gbit/s as a function of frequency, assuming thermal noise and a path loss corresponding to the attenuation by water absorption over a distance of 2 mm. While formulated for a single channel, at certain wavelengths (e.g., RF) these factors also constrain multiplexed data transmissions between many transmitters and many receivers, depending on capacity of the system for spatial multiplexing. Horizontal dashed lines: 40 mW, the approximate maximal whole-brain power dissipation in steady state.

![](_page_20_Figure_2.jpeg)

power, bandwidth and penetration depth allow transmission of > 100 Gbit/s out of the brain through a single channel without unacceptable power consumption.

The analysis above has ignored the effects of noise sources other than thermal noise, but many additional noise sources will increase the amount of power needed to transmit data, via a decrease in the SNR at fixed input power. For optical transmission in the brain, the noise is dominated by time-correlated "speckle noise" below 200 kHz, which arises mostly from local blood flow [236]. This correlated noise, which cannot be filtered by simple averaging, could be avoided by modulating optical signals at frequencies above 200 kHz.

#### 4.4.2 SPATIALLY MULTIPLEXED DATA TRANSMISSION

As discussed above, transmitting information through a single channel imposes direct limits on bit rate, carrier frequency and input power. However, it is conceivable to divide the data transmission burden over many independent channels, i.e., over many pairs of transmitters and receivers, each operating at lower bandwidth (e.g., at radio frequencies). Indeed, this would be optimal in a scenario where many embedded devices measure and then transmit the activities of nearby neurons. As a concrete example of such "spatial multiplexing," an effective capacity of 1 Tbit/s could conceivably be obtained by splitting the data over 1000 transmitter-receiver pairs each operating at 1 Gbit/s, with the transmitters arranged in a 10 × 10 × 10 grid. Importantly, in order to exceed the above limits for single-channel data transmission, it must be possible for these transmitter receiver pairs to share the same bandwidth and operate simultaneously without conflicts, for example by modulating distinguishable carrier waves or by transferring data over separate wires. The conditions under which this may occur, however, can be counter-intuitive. For example, for antennas to operate independently, they must be spaced apart from one another by roughly a wavelength. For 10 GHz microwaves, the wavelength is ~3 cm, so no more than a handful of microwave transmitters (e.g., operating at frequencies in the 100 GHz–1 THz range) can co-occupy the mouse brain while operating independently.

Even with many non-independent transmitters co-occupying the brain and operating simultaneously over the same frequency spectrum, it may be possible under some conditions to "factor out" the effects of the coupling and allow an increase in channel capacity relative the single-channel result. To treat such scenarios, a generalization to Shannon's capacity theorem to multi-input-multi-output (MIMO) channels has shown that the maximal total data rate is

$$R_{\text{max}} = BW \cdot \log_2 |\boldsymbol{I} + (SNR)\boldsymbol{H}\boldsymbol{H}^*|$$

where I is the identity matrix,  $|\cdot|$  denotes the matrix determinant, H is the  $(M \times N)$  for N transmitters and M receivers) channel matrix giving the coupling between the vector of transmitted signals and the vector of received signals and  $H^*$  denotes the matrix adjoint of H [237]. The vector of received signals is then y = Hx + n where x is the vector of transmitted signals and n is a noise vector. Any matrix can be written as  $H = U \Sigma V^*$  where U and V are unitary matrices, and  $\Sigma$  is a diagonal matrix whose elements are the singular values  $\lambda_i$ . One can re-write the above equation as

$$R_{\text{max}} = \text{BW} \cdot \sum_{i=1}^{\min(M,N)} \log_2 \left(1 + \text{SNR} \cdot \lambda_i^2\right)$$

If the matrix H is of full rank, then the capacity for the multi-channel system can increase over the single-input-single-output (SISO) result by  $\min(M,N)$  times [238]. Note that the rank of the matrix corresponds to the number of non-zero singular values, so an analysis of the singular values of channel matrices can inform us about the multiplexing capacity of the channel. Furthermore, this multiplexing capacity can in principle be achieved even when the transmitters are not in communication with each other, which could potentially be important for scenarios involving many brain embedded transmitters [239].

Transmission through a medium with negligible scattering is the simplest situation to analyze. In this case, evaluating the matrix H requires knowledge of the transmitter-transmitter, transmitter- receiver, and receiver-receiver distances, as well as the orientations and radiation patterns of the antennas (e.g., high gain antennas will have a highly directional radiation pattern). Depending on these factors, the beam from each transmitter will spread to impinge upon multiple receivers and the effective number of spatially independent beams will be reduced. With transmitter-transmitter and receiver-receiver distances larger than the wavelength, and highly directional antennas with appropriately chosen orientations, it is possible to increase the channel capacity linearly with  $\min(M, N)$ .

Random scattering, in a coherent disordered medium where the mean free-path  $\ell$  is much larger than the wavelength  $\lambda$  and much smaller than the size of the disordered medium, is another condition where the matrix H is a random scattering matrix of full rank [240, 241]. Intuitively, for the case of two transmitters and two receivers separated by a disordered medium larger than the mean free path: if transmitter 1 is at least a mean-free path from transmitter 2 (or potentially as close as a few wavelengths [242]), the path from transmitter 1 to receiver 1 and the path from transmitter 2 to receiver 2 would be uncorrelated with respect to one another (in terms of physical path, phase, amplitude fluctuations, and other properties). The rank of the matrix H would then be 2. Devising

a code on the transmitter such that the receivers can distinguish between these two uncorrelated streams results in a doubling of the capacity, rather than simply averaging the noise floor, which would provide only a logarithmic capacity gain due to the increased SNR.

Thus, contrary to intuition, a high degree of random scattering can potentially be useful for data transmission, by enabling spatial multiplexing of channels. This idea has been demonstrated experimentally in the context of ultrasound transmissions [243]. Biological tissue in the infrared range is well described as such a random scattering medium (e.g., mean free path ~200  $\mu$ m at ~800 nm *in vivo*). Therefore infrared light could be used for spatially multiplexed data transmission out of the brain. At wavelengths  $\lambda$  comparable to critical brain dimensions in the mouse, however, an insufficient number of scattering events will occur to create multiple independent pathways for N transmitters. Mathematically, the matrix H will have one highly dominant singular value and a number of much smaller remaining terms, such that the signals appearing at a receiver from two separate transmitters will be highly linearly dependent, differing by only a small phase angle. Therefore, there will be no capacity gain from multiple transmitters, and distinct transmitters will effectively share a single channel (reducing to the SISO result).

Little is known about the biological interaction with electromagnetic fields at wavelengths much shorter than the critical brain dimensions but beyond the infrared, approximately 100 GHz (~3 mm) to 100 THz (~3 µm) in the mouse. If multiple scattering occurs and the absorption is low, this may also be a regime conducive to MIMO communications [244]. Efficiently generating and processing radiation in this regime by embedded devices is an outstanding problem, however. The so-called "THz-gap" [245] exists because (moving towards higher frequencies starting from DC electronics), parasitic capacitances and passive losses limit the maximum frequency at which a field-effect transistor (FET) may oscillate and on the other hand (moving downward in frequency starting from optics), the band-gaps of opto-electronic devices limit the minimum frequency at which quantum transitions occur. Thus there is no high-power, low-cost, portable, room temperature THz source available. Advances in THz light generation, e.g. through the use of tunneling transistors, could be enabling.

#### 4.4.3 ULTRASOUND AS A DATA TRANSMISSION MODALITY

An important caveat to these conclusions on wireless data transfer occurs if we consider the use of ultrasound rather than electromagnetic radiation. Because the speed of sound is dramatically slower than that of light, the wavelength of 10 MHz ultrasound is only  $\sim 150 \, \mu m$  (approximating the speed of sound in brain as the speed of sound in water,  $\sim 1500 \, m/s$ ). Thus, many 10 MHz ultrasound transmitters/receiver could be placed inside a mouse brain while maintaining their spatial separation above the wavelength, and a linear scaling of the MIMO channel capacity with the number of devices is likely possible in this regime, assuming that appropriate antenna gains and orientations can be achieved inside brain tissue. Beam orientation could present a challenge if micro-devices are oriented randomly after implantation. With an attenuation of 0.5 dB/(cm MHz) [246], the attenuation at 10 MHz is only 5 dB/cm. Thus ultrasound-based transmission of power and data from embedded recording devices may be viable [70].

In contrast, direct imaging of neural activity by ultrasound (e.g., using contrast agents which create local variations in tissue elastic modulus or density) may be more difficult. While the theoretical (diffraction-limited) and currently practical resolutions of 100 MHz ultrasound are ~15 µm, and 15–60 µm [247], respectfully, at these frequencies, power is attenuated by brain tissue with a coefficient of ~50 dB/cm [246] (10<sup>5</sup>-fold attenuation per cm), which imposes a penetration limit (e.g., for measurements with a dynamic range of 80 dB [247]). Attenuation of ultrasound by bone is stronger still, at 22 dB/(cm MHz) [246]. Attenuation could therefore limit the use of ultrasound as a high-resolution neural recording modality in direct imaging modes, but multiplexed transmission of lower-frequency ultrasound from embedded devices could sidestep this issue.

#### 4.4.4 CONCLUSIONS AND FUTURE DIRECTIONS

Physics and information theory impose a tradeoff between bandwidth and power consumption in sending data through any communication channel. Considering only thermal noise and no path loss, achieving 100 Gbit/s data rates through a single channel necessitates either a bandwidth above a few GHz or a transmitted power above ~ 100 mW, the latter of which may be prohibitive from a heat dissipation perspective if the signals are to be generated by dissipative microelectronic devices. Researchers have proposed to use thousands or millions of tiny [248] wireless transmitters embedded in the brain to transmit local neural activity measurements to an external receiver via microwave radiation [249]. However, based on the above power-bandwidth tradeoff, this will require a bandwidth above a few GHz. At the corresponding carrier frequencies, the penetration depth of the microwave radiation drops significantly, requiring increased power to combat the resulting signal loss. While one might hope that multiple independent channels could be multiplexed inside the brain, reducing the bandwidth and power requirements for each individual channel, the long wavelengths of microwave radiation compared to the mouse brain diameter suggest that such channels cannot be independent, as is confirmed by an analysis of the multi-input-multi-output (MIMO) channel capacity for this scenario. Therefore, radio-frequency electromagnetic transmission of whole brain activity data from embedded devices does not appear to be a viable option for brain activity mapping.

On the other hand, an analysis of the channel capacity for IR transmissions in a diffusive medium suggests that, because of its high frequency and decent penetration depth, infrared radiation may provide a viable substrate for transmitting activity data from

embedded devices. For example, data could be transmitted via modulating the multiple-scattering speckle pattern of infrared light by varying the backscatter from an embedded optical device, such as an LCD pixel [250], in an activity-dependent fashion. Because the speckle pattern is sensitive to the motion of a single scatterer [242, 251], coherent multiple scattering could effectively act as an optical amplifier and as a means to create independent communication pathways. Furthermore, multiplexed data transmission via ultrasound is likely possible because of its short wavelength in tissue at reasonable carrier frequencies. It may also be of interest to explore network architectures [252] in which data is transmitted at low transmit power over short distances via local hops between neighboring nodes capable of signal restoration.

## 4.5 MAGNETIC RESONANCE IMAGING

Magnetic resonance imaging (MRI) uses the resonant behavior of nuclear spins in a magnetic field to non-invasively probe the spatiotemporally varying chemical and magnetic properties of tissues. Although originally conceived as a means to image anatomy, MRI can be used to observe neural activity provided that correlates of such activity are reflected in dynamic changes in local chemistry or magnetism.

In an MRI study, a strong static field ( $B = 1-15\,\mathrm{T}$ ) is applied to polarize nuclear spins (usually  $^1\mathrm{H}$ ), causing them to resonate at a field-dependent Larmor frequency

 $f = \frac{\gamma}{2\pi} B$ 

where  $\gamma$  is the gyromagnetic ratio of the nucleus (e.g., <sup>1</sup>H has a gyromagnetic ratio of 267.522 MHz/T [253] and therefore resonates at 42.577 MHz in a 1 T field). To obtain positional information, spatial field gradients are applied such that nuclei at different positions in the sample resonate at slightly different frequencies. Sequences of RF pulses and gradients are then applied to the sample, eliciting resonant emissions that contain information about spins' local chemical environment, magnetic field anisotropy and various other properties.

Most functional studies rely on dynamic changes in two forms of relaxation experienced by RF-excited spins. The first form results from energy dissipation through interactions with other species (e.g. other spins or unpaired electrons), causing the spins to recover their lowest energy state on a timescale,  $T_1$ , of 100–1000 ms [254]. The second form of relaxation reflects the dephasing of spin signals in a given sampling volume (voxel) over a timescale,  $T_2$ , of 10–100 ms [255] due to non-uniform Larmor frequencies caused, e.g., by the presence of local magnetic field inhomogeneities.

In blood-oxygen level dependent [256] functional MRI (BOLD-fMRI), the most widely used form of neural MR imaging, increased neural activity in a given brain region alters the vascular concentration of paramagnetic deoxy-hemoglobin, which affects local magnetic field homogeneity and thereby alters  $T_2$ . Although the existence of this paramagnetic reporter of oxygen metabolism is fortuitous, the data it provides is only an indirect readout of neural activity [257–259], which is limited in its spatial and temporal resolution to the dynamics of blood flow in the brain's capillary network (1–2 s). The spatial point-spread function of the hemodynamic BOLD response is in the 1 mm range, although sub-millimeter measurements, revealing cortical laminar and columnar features, have been obtained by filtering out the signals from larger blood vessels [260]. A significant area of current and future work is aimed at developing new molecular reporters that can be introduced into the brain to transduce aspects of neural signaling such as calcium spikes and neurotransmitter release into MRI- detectable magnetic or chemical signals [261–263], as described in section 4.5.3, below.

#### 4.5.1 SPATIOTEMPORAL RESOLUTION

The temporal resolution of MRI is limited by the dynamics of spin relaxation. For sequential MR signal acquisitions to be fully independent, spins must be allowed to recover their equilibrium magnetization on the timescale of  $T_1$  (100–1000 ms). However, if local  $T_1$  is static its pre-mapping could enable temporally variant  $T_2$  effects to be observed at refresh rates on the faster  $T_2$  timescale (10–100 ms) [255]. It may also be possible to detect events that occur on a timescale shorter than  $T_1$  and  $T_2$ , if the magnitude of the resulting change in spin dynamics overcomes the lack of independence between acquisitions. Note that these limitations on the repetition time of the underlying pulse sequence are not eliminated by "fast" pulse sequences such as echo-planar imaging (EPI) [264] and fast low-angle shot (FLASH) [265] or by the use of multiple detector coils [266]. These techniques accelerate the acquisition of 2D and 3D images, but still require spins to be prepared for readout.

The spatial resolution of current MRI techniques is limited by the diffusion of water molecules during the acquisition time [267], since contrast at scales above the diffusion length will be attenuated by diffusion. The RMS distance of a water molecule from its origin, after diffusing in 3D for a time  $T_{\rm acq}$ , is

$$d_{\rm rms} = \sqrt{6D_{\rm water}T_{\rm acq}}$$

where  $D_{\rm water}=2300\,\mu{\rm m}^2/{\rm s}$  is the self-diffusion coefficient of water. For  $T_{\rm acq}\approx 100\,{\rm ms},\,d_{\rm rms}\approx 37\,\mu{\rm m},$  which sets the approximate spatial resolution. For ultra-short acquisitions at  $T_{\rm acq}\approx 10\,{\rm ms},\,d_{\rm rms}\approx 12\,\mu{\rm m}.$ 

More technically, as described above, MRI uses field gradients to encode spatial positions in the RF frequency (wavenumber) components of the emitted radiation. The quality of the reconstruction of frequency space thus limits the achievable spatial resolution. The sampling interval of the detector  $\Delta t$ , and the field gradient G, determine the wavenumber increment as

$$\Delta k = \gamma G \Delta t$$

The spatial resolution (here considering only one dimension) is then given by [267]:

$$\Delta x_{k\text{-space}} = \frac{\pi}{\frac{T_{\text{acq}}}{\Delta t} \Delta k} = \frac{\pi}{T_{\text{acq}} \gamma G}$$

Note that it is the gradient field, not the polarizing field  $B_0$ , which determines the resolution. For a gradient field of 100 mT/m and an acquisition time of 100 ms

$$\Delta x_{k\text{-space}} = \frac{\pi}{(100 \text{ ms}) (267 \text{MHz/T}) (100 \text{ mT/m})} \approx 1.17 \,\mu\text{m}$$

Due to relaxation, however, the emissions from a spin at a given position do not constitute a pure tone with a well-defined frequency. Instead, each spin exhibits a frequency spread, which gives rise to another limit on the spatial resolution [267]:

$$\Delta x_{\text{relaxation}} = \frac{2}{\gamma G T_2^*}$$

where  $T_2^*$  is the shortest relaxation time. Assuming  $T_2^* = 5 \text{ ms}$  and G = 100 mT/m, gives

$$\Delta x_{\rm relaxation} \approx 14 \,\mu {\rm m}$$

Therefore, for water protons, the resolution limit is set by diffusion over  $\sim 100$  ms acquisition timescales, rather than by k-space sampling or relaxation. For other spin species (e.g., with lower diffusion rate), it may be possible to achieve resolutions limited by frequency discrimination.

Notably, there exists a practical trade-off between spatial resolution, temporal resolution, and sensitivity (SNR). In particular, to achieve high spatial resolution, it is necessary to densely sample k-space. Fast sampling sequences such as FLASH and EPI achieve speed by sampling each point of k-space using less signal and often at a lower resolution. Even at high field strengths (11.7 T), this tradeoff results in practical EPI-fMRI with a spatial resolution of  $150 \,\mu\text{m} \times 150 \,\mu\text{m} \times 500 \,\mu\text{m}$  and a temporal resolution of  $200 \,\text{ms}$  [268]. Achieving much higher spatial resolutions requires longer acquisitions and/or lower temporal sampling. For example, achieving a  $20 \,\mu\text{m}$  anatomical resolution in MRI of *Drosophila* embryos required 54 minutes for a small field of view of  $2.5 \,\text{mm} \times 2.5 \,\text{mm} \times 5 \,\text{mm}$  [269]. Furthermore, the flies were administered paramagnetic gadolinium chelates to shorten  $T_1$  and thereby the acquisition time. Separately, frame rates of 50 ms have been obtained for dynamic imaging of the human heart, but required the use of strong priors to reduce data collection requirements [270].

#### 4.5.2 ENERGY DISSIPATION

Energy is dissipated into the brain when the excited spins relax to their equilibrium magnetization in the applied field. The energy associated with this relaxation is of order the Zeeman energy:

$$\Delta E_{\text{Zeeman}} = \frac{\gamma}{2\pi} h B_0$$

To obtain an upper bound on the heat dissipation of MRI, we first assume that the brain is entirely water, that every proton spin is initially aligned by the field and then excited by the RF pulse, and that all spins relax during a  $T_1$  relaxation time of  $\sim$ 600 ms. In this scenario, even an applied field of as high as  $\sim$ 200 T would generate dissipation within the  $\sim$ 50 mW energy dissipation limit. In reality, the energy dissipation is 4–5 orders of magnitude smaller, because only a tiny fractional excess of the spins are initially aligned by the field ( $\sim$ 1 × 10<sup>-5</sup> for fields on the order of 1 T). Therefore, thermal dissipation associated with spin excitation in MRI is unlikely to cause problems unless field strengths much greater than the largest currently used fields ( $\sim$ 20 T) are invoked, or spins with much higher gyromagnetic ratios are used.

Practically, the main energy consideration in MRI is the absorption by tissues of RF energy applied during imaging pulse sequences and the switching of magnetic field gradients. Such absorption is often calculated through numerical solutions of the Maxwell Equations taking into account the precise geometry, tissue properties and applied fields for a particular experimental setup [271]. The typical specific absorption rate (SAR) is well under 10 W/kg (or 5 mW per 500 mg), and is restricted by the FDA to less than 3 W/kg for human studies.

#### 4.5.3 IMAGING AGENTS

All the preceding discussion about spatiotemporal resolution presumes the existence of local time-varying signals (e.g., changes in  $T_1$  or  $T_2$ ) corresponding to the dynamics of neural activity. The hemodynamic BOLD response is the most prominent such signal, the limitations of which are discussed above. There have been studies working towards direct detection of minute (e.g.,  $\sim$ 0.2 nT) magnetic fields associated with action potentials through their effects on MRI phase or magnitude contrast [272, 273], but reliably detecting these fields above the physiological noise will likely require novel strategies [274, 275] and estimates of the feasibility of these methods have been complicated by the lack of a realistic model for the local distribution of neuronal currents. MRI detection of the mechanical displacement of active neurons due to the Lorentz force in an applied magnetic field [276] has also been explored, as has the detection of activity-dependent changes in the diffusion of tissue water [277, 278], possibly due to neuronal or glial [279] cell swelling [280, 281], although strongly diffusion-weighted scans may have disadvantages in terms of SNR [282]. Manganese influx through voltage-gated calcium channels [283, 284] generates MRI contrast, but exhibits slow uptake kinetics and even slower efflux, such that manganese monotonically accumulates in the neurons over time. Conceivably, over-expression of manganese efflux pumps such as the iron transporter ferroportin [285] could allow time-dependent activity imaging using manganese contrast.

In the past 15 years, efforts have been undertaken to develop chemical and biomolecular imaging agents that can be introduced into the brain to produce MRI detectable signals corresponding to specific aspects of neural function (analogously to fluorescent dyes and proteins). One critical advantage of using genetically encoded indicators would be the ability to target these indicators to specific cell types [286, 287] and/or cellular compartments [186–193]. Notable examples of engineered molecular MRI contrast agents include  $T_1$  and  $T_2$  sensors of calcium [288, 289] and a  $T_1$  sensor of neurotransmitter release [261]. Depending on their mode of action, these imaging agents can provide temporal resolutions ranging from 10 ms to 10 s [290]. However, a major current limitation for fast agents is the requirement that they be present in tissues at  $\mu$ M concentrations, posing major challenges for delivery and genetic expression. Model organisms lacking hemoglobin (e.g., the blowfly), and hence lacking a hemodynamic BOLD response (as is also the case for ex-vivo brain slices), may be particularly useful for in-vivo testing of novel activity-dependent contrast mechanisms, and specialized setups have been constructed to perform MRI at near-cellular spatial resolution in this context (though still requiring several hours to generate whole-brain anatomical images at this resolution) [291].

Figure 6 shows the achievable temporal resolution for various classes of activity-dependent MRI contrast agents as well as the spatial resolution limit due to water proton diffusion.

**Figure 6.** Key factors determining the spatiotemporal resolution of dynamic MRI imaging. (a) Temporal resolution and contrast agent concentration allowing > 5 % contrast, for different classes of dynamic MRI contrast agent (reproduced from [290], with permission). (b) Diffusion limited spatial resolution for water proton MRI as a function of temporal resolution.

![](_page_25_Figure_6.jpeg)

## 4.5.4 CONCLUSIONS AND FUTURE DIRECTIONS

Moving beyond hemodynamic contrast is crucial for improving the spatiotemporal resolution of fMRI, and several avenues may be available for doing so, especially through the use of novel molecular contrast agents and/or genetic engineering. More fundamentally, current MRI techniques rely on the excitation of proton spins in water: this limits imaging to > 100 ms timescales, unless SNR is severely compromised, due to the low polarizability and long  $T_1$  relaxation times of proton spins. There is also a spatial resolution limit of tens of microns over these timescales due to water's fast diffusion. Methods which couple neural activity to non-diffusible, highly polarized spins could, in principle, ameliorate this situation.

## 4.6 MOLECULAR RECORDING

An alternative to electrical, optical or MRI recording is the local storage of data in molecular substrates. Each neuron could be engineered to write a record of its own time-varying electrical activities onto a biological macromolecule, allowing off-line extraction of data after the experiment. Such systems could, in principle, be genetically encoded, and would thus naturally record from all neurons at the same time.

One proposed implementation of such a "molecular ticker tape" would utilize an engineered DNA polymerase with a Ca<sup>2+</sup>-sensitive or membrane-voltage-sensitive error-rate [5] to record time-varying neural activities onto DNA [6] as patterns of nucleotide misin-corporations relative to a known template DNA strand (for alternative local recording techniques see [292, 293]). The time-varying signal would later be recovered by DNA sequencing and subsequent statistical analysis [6]. DNA polymerases found in nature can add up to ~1000 nucleotides per second [294], and certain non-replicative polymerases such as DNA polymerase iota have error rates of >70 % on template T bases [295]. Similar strategies could be implemented using RNA polymerases or potentially using other enzyme/hetero-polymer systems.

#### 4.6.1 SPATIOTEMPORAL RESOLUTION

Polymerases proceed along their template DNA strands in a stochastic, thermally driven fashion; thus, polymerases that are initially synchronized will de-phase with respect to one another over time, occupying a range of positions on their respective templates at the time when a neural impulse occurs. The rate of this de-phasing is a key parameter governing the temporal resolution of molecular recording. By averaging over many simultaneously replicated templates, it is theoretically possible to associate variations in nucleotide misincorporation rate with the times at which these variations occurred, and thus to obtain temporally resolved recordings of the cation concentration [6].

An analysis of the projected temporal resolution of molecular ticker tapes as a function of polymerase biochemical parameters can be found in [6]. This work suggests that molecular ticker tapes require synchronization mechanisms if they are to record at < 10 ms temporal resolution for durations longer than seconds, even when 10000 templates per cell are recorded simultaneously, unless engineered polymerases with kinetic parameters beyond the limits of those found in nature can be developed. Recording at lower temporal resolutions, however, appears feasible using naturalistic biochemical parameters, even in the absence of synchronization mechanisms.

The development of mechanisms to improve synchronization of the ensemble of polymerases within each cell, or to encode time-stamps into the synthesized DNA (e.g., molecular clocks), could improve temporal resolution and decrease the number of required template strands per neuron. Mutation-based molecular clocks over evolutionary timescales are widely used in the field of phylogenetics [296], and new tools from synthetic biology [297] and optogenetics or thermogenetics [298] also suggest strategies for building molecular clocks on faster timescales. As an example sketch of a possible synchronization mechanism, optogenetic methods (e.g., similar to [299]) could be used to halt, and thus re-phase, a sub-population of polymerases at a light-dependent pause site in the template DNA, while another sub-population of polymerases reads through this pause site to maintain temporal continuity of recording; then the second population could be re-synchronized at an orthogonal light-dependent pause site while the first population reads through. Alternatively, some form of optogenetics could be used to directly write bit strings encoding time stamps into the synthesized DNA. These strategies would require one or two, sufficiently strong global clock signals to be optically broadcast to all neurons. The optics involved would be comparatively simple: this could be done using far fewer optical fibers than would be required for fiber-based activity readout, for instance. Alternatively, if the brain could be flash-frozen at a precisely known time, this could serve as a global time-stamp corresponding to the termination of DNA synthesis (e.g., the DNA 3' end).

Spatial resolution for molecular recording would naturally reach the single cell level. To determine which nucleic acid tape originated from which neuron, static cell-specific DNA barcoding could be used [300] to associate the synthesized DNA strands with nodes in a topological connectome map obtained via DNA sequencing. Fluorescence in-situ DNA sequencing (FISSEQ) [301] on serially-sectioned or intact tissue (fixed post-mortem) [302] could be used to obtain explicit geometric information.

#### 4.6.2 ENERGY DISSIPATION

Nucleotide metabolism DNA polymerization imposes a metabolic load on the cell. Replication of the 3 billion bp human genome takes approximately eight hours in normally dividing cells, which equates to a nucleotide incorporation rate of ~100 kHz. Therefore, in order not to exceed the metabolic rates associated with normal genome replication, molecular ticker tapes operating at 1 kHz polymerization speed [294] would be limited to approximately 100 simultaneously replicated templates per cell. Even more recordings would be possible for RNA ticker tapes. The mammalian cell polymerizes at least 10<sup>11</sup> NTPs per 16-hour cell cycle [303]. Therefore, ~1700 RNA tickertapes, each operating at 1 kHz, could be placed in a cell before generating a metabolic impact equal to that of the cell's baseline transcription rate. While these comparisons to baseline physiological levels are reasonable guidelines, it is likely that a neuron can support higher metabolic loads associated with larger numbers of templates. The maximal rate of

neuronal aerobic respiration is  $\sim$ 5 fmol of ATP minute via oxidative respiration (see the section on bio-luminescence). Assuming  $\sim$ 1 ATP equivalent consumed per nucleotide incorporation, if neuronal metabolism were entirely dedicated to polymerization, it could support the incorporation of up to  $6 \times 10^9$  nucleotides per minute, or  $10^5$  simultaneously replicated DNA templates at 1 kHz.

**Power dissipation** Normal DNA and RNA synthesis do not produce problematic energy dissipation and molecular tickertapes will likewise not be highly dissipative, at least in the regime where nucleic acid polymerization rates do not exceed those associated with genome replication or transcription.

#### 4.6.3 VOLUME DISPLACEMENT

The nucleus of a neuron occupies  $\sim$ 6% of a neuron's volume  $((4 \mu m)^3/(10 \mu m)^3)$ . Ticker tapes operating at 1 kHz with 10000 simultaneously replicated templates could record for 300 seconds before the total length of DNA synthesized equals the human genome length. In the case of RNA polymerase II-based transcription, 2.75 h of recording by 10000 recorders is required to reach the net transcript length in the cell. Therefore, with appropriate mechanisms to fold/pack the nucleic acids generated by molecular ticker tapes, they would not impose unreasonable requirements on cellular volume displacement over minutes to hours.

#### 4.6.4 CONCLUSIONS AND FUTURE DIRECTIONS

Molecular recording of neural activity has the advantages of inherent scalability, single-cell precision, and low energy and volume footprints. Making molecular recording work at temporal resolutions approaching 1 kHz, however, will require multiple new developments in synthetic biology, including protein engineering to create a fast polymerase (> 1 kHz) that strongly couples proxies for neural activity to nucleotide incorporation probabilities. Synchronization mechanisms would likely be required to perform molecular recording at single-spike temporal resolution. An attractive potential payoff for molecular approaches to activity mapping is the prospect of seamlessly combining — within a single brain — the readout of activity patterns with the readout of structural connectome barcodes [300, 304], transcriptional profiles [301] (e.g., to determine cell type) or other (epi-)genetic signatures [305] which are accessible via high-throughput nucleic acid sequencing.

## 5 DISCUSSION

We have analyzed the physical constraints on scalable neural recording for selected modalities of measurement, data storage, data transmission and power harvesting. Each analysis is based on assumptions – about the brain, device physics, or system architecture – which may be violated. Understanding these assumptions can point towards strategies to work around them, and in some cases we have suggested possible directions for such workarounds. Even valid assumptions about natural brains may be subject to modification through synthetic biology or external perturbation. For example, methods for rapidly removing heat from the brain could work around our assumptions about its natural cooling capacity, supporting a range of highly dissipative recording modalities. Likewise, assumptions about the necessary bandwidth for data transmission could be relaxed if some information is stored locally and read out after the fact.

In some cases, theoretical extensions of our first-order analyses could reveal important insights. The power-bandwidth tradeoffs identified in section 4.4 for electromagnetic data transmission may place limits on the informational throughput of fMRI, for example, or a realistic simulation of heat fluxes in the brain could reveal the true limits of power dissipation. In many other cases, new experiments will be required to move beyond crude estimates of feasibility.

The analysis of physical limits illustrates challenges and opportunities for technology development. While the opportunities can only be touched upon here, and some directions have been treated elsewhere [1, 138, 306], we anticipate further analyses which could explore design spaces in detail. Here we briefly summarize a sampling of new directions suggested by our analysis.

Electrical recording The signal to noise ratio for a voltage sensing electrode imposes limits on the number of neurons per electrode from which signals can be detected and spike-sorted, likely requiring roughly one electrode per 100 neurons. To go beyond this, pure voltage sensing nodes could be augmented with the ability to directionally resolve distinct sources. For example, the 3D motion of a charged nanoparticle in an electric field, or of a dielectric nanoparticle in an electric field gradient, could be monitored at each recording site [307].

Optical recording While light scattering creates severe limitations on optical imaging, embedded optical microscopies could overcome these limits. Embedded optical imaging systems with high signal multiplexing capacity would be desirable, to minimize the required number and size of implanted optical probes.

One option might be to use time-of-flight information to multiplex many sensor readouts into a single optical fiber: this could potentially be realized using time-domain reflectometry techniques, commonly used to determine the positions of defects in optical

fibers, coupled to neural activity sensors arranged along the fiber, which would modulate the fiber's local absorption or backscatter [307]. Time-domain reflectometry techniques have already reached 40 µm resolution [308].

Alternatively, novel fluorescent or bio-luminescent activity indicators could in principle relax the limits associated with light scattering, either by enabling efficient two-photon excitation at lower light dosages, or through all-infrared imaging schemes. Infrared bio-luminescence may be a particularly high-value target.

Delivery For both embedded optical and electrical recording strategies, new delivery mechanisms will be needed to scale to whole mammalian brains. Many of the basic parameters for scalable delivery mechanisms are still unknown. For example, can a large number of ultra-thin nano-wire electrodes or optical fibers be delivered via the capillary network? Can cells such as macrophages engulf ultra-miniaturized microchips and transport them into brain tissue? Can the blood brain barrier be locally opened (e.g., using ultrasonic stimulation [309]) to allow targeted delivery of recording probes?

Intrinsic signals The ideal technique would not require exogenous contrast agents or genetically encoded indicators, instead relying on signals intrinsic to neurophysiology. Neurons exhibit few-nano-meter scale [310] membrane displacements (e.g., in response to Maxwell stresses from large local electric field variations) during the action potential [311]. These can be measured using optical interferometry [312], but in principle they could also be monitored acoustically (and related activity-associated membrane swellings have been directly observed by atomic force microscopy [313] in cultured neurons). Sensors could be embedded in or around tissue to transduce the resulting acoustic vibrations into an electrical or optical readout. This could potentially allow recording at larger distances than the ~130 µm maximum recording radius for a voltage sensing node. Other intrinsic signals include changes in refractive index associated with neural activity, which will modulate the reflection and scattering of light [314]. These intrinsic changes in optical properties can be measured with optical coherence tomography (OCT) [315]. Local metabolic and hemodynamic signatures are also detectable optically, such as hemoglobin oxygenation (e.g., via functional near-infrared spectroscopy [316]) and the partial pressure of oxygen [317, 318]. For minimal invasiveness, diffuse optical tomography uses near-infrared light (600–950 nm), which passes sufficiently-readily through the skin and skull to allow imaging of hemodynamics in cortex [319–321], although currently with limited spatial and temporal resolution.

Data transmission through diffusive media Unlike radio-frequency electromagnetics, infrared wavelengths may allow spatially multiplexed data transmissions from embedded recording devices, creating multiple independent channels by taking advantage of the stochasticity of light paths in strongly-scattering tissue. Alternatively, techniques are emerging to dynamically measure and invert the optical scattering matrix of a turbid medium, using pure-optical or hybrid techniques.

Ultrasound Certain wavelengths of ultrasound exhibit potentially-favorable combinations of wavelength (spatial resolution), bandwidth (frequency) and attenuation compared to radio-frequency electromagnetics. Ultrasound could be used as a mechanism for powering and communicating with embedded local recording chips [70]. Novel indicators [322] would likely need to be developed to perform neural activity imaging using pure ultrasound. Hybrid techniques such as photo-acoustic [194] or ultrasound-encoded optical [122] microscopies are also of interest.

**Molecular recording** For local recording, molecular recording devices could sidestep power constraints on embedded electronics, at the cost of increased engineering complexity. For molecular recording to become practical at temporal resolutions approaching the millisecond scale, sophisticated protein and viral engineering would likely be required to create a high-speed polymerase-based recorder operating in the neuronal cytoplasm. This would also necessitate molecular synchronization or time-stamping mechanisms to maintain phasing between multiple polymerases within a single cell, as well as between different cells.

On the other hand, molecular recording devices operating at slower timescales (e.g., seconds) could perhaps be engineered via more conservative combinations of known mechanisms, such as CREB-mediated signaling to the nucleus [323] or nuclear-localized calcium sensing [324]. In either case, the nucleic acid strands resulting from such molecular recorders could be space-stamped with cell-specific viral connectome barcodes [300] for later readout by bulk sequencing. Alternatively, the ticker tapes could be read within their anatomical contexts by in-situ sequencing, i.e., nucleic acid sequencing performed inside intact tissue [301].

Combining static and dynamic datasets Combining dynamic activity information with static structural or molecular information could allow these datasets to disambiguate one another. For example, a diversity of colors for fluorescent activity indicators (i.e., a form of BrainBow [185] calcium imaging) could ease requirements on spatial separation of optical signals, and the color pattern across cells could be mapped post-mortem at single-cell resolution using in-situ microscopy. Generalizing further, in-situ sequencing enables the extraction of vast quantities of molecular data from fixed tissue, in effect allowing observations with a palette of  $4^N$  colors, where N is the length of the nucleic acid polymer. It may be possible to harness this exponential informational resource to enhance the readout of dynamic activity information as well, e.g., through molecular recording.

MRI Current MRI is limited by its reliance on intrinsic hemodynamic contrast mechanisms and on rapidly diffusing aqueous protons. Indicators coupling neural activity to spin relaxation rates are being developed to move beyond hemodynamic contrast. Novel excitation and detection schemes that could sensitize MRI to fast, local, intrinsically activity-dependent mechanisms (e.g., cell swelling, neuronal magnetic fields), while filtering out the slower BOLD response, are also of interest and should initially be tested in organisms or slice preparations lacking hemodynamic responses. Detailed computational models of neuronal currents within a tissue voxel (e.g., in the spirit of [71]), and of the resulting mechanical and chemical changes, could be useful for evaluating potential new methods. In principle, MRI could also abandon the use of water protons as the signal sources, although this would pose significant implementation challenges.

Readout methods New signal processing frameworks such as compressive sensing could reduce bandwidth requirements and inspire new microscope designs exploiting computational imaging principles [325–328]. Fast readout mechanisms [329] applied to giga-pixel arrays (e.g., the 3.2 giga-pixel CCD camera planned for the Large Synoptic Survey Telescope, which will have ~1 s readout time) might be adapted to large-scale electrical or optical recording methods. Linear photodiode arrays can achieve 70 kHz line readout rates [330], and many such linear arrays could be read out in parallel. Optoelectronic methods that convert between time, space and frequency representations of signals [331–337] could inspire designs for even faster readouts (e.g., ~10 MHz frame rates have been demonstrated in brightfield imaging). Although these methods are not directly compatible with fluorescence measurements due to their use of spectral dispersion, related ideas (e.g., beat frequency multiplexing) may enable fluorescence microscopy at rates above that of CCD-based imaging [123, 124], limited ultimately by fluorescence lifetimes, while also exhibiting favorable properties with respect to scattering.

Alternative modalities X-ray imaging has been used on live cells [338] and might find use in neural recording if suitable contrast agents could be devised. X-rays interact with electron shells via photoelectric absorption and Compton scattering and with band structure in materials. X-ray phosphors utilize substitutions in an ionic lattice to generate visible or UV light emission upon X-ray absorption [339]. In principle, some of these mechanisms could be engineered as neural activity sensors, e.g., in an absorption-contrast mode suitable for tomographic reconstruction [340]. While tissue damage due to ionizing radiation would ultimately be prohibitive (e.g., on a timescale of minutes [307]), very brief experiments might still be possible.

Likewise, electron spin resonance (ESR) operates at  $\sim 100 \times$  higher Larmor frequency compared to proton MRI, which improves polarizability of the spins. Due to Pauli exclusion, use of this technique requires an indicator with unpaired electrons. These can be found in nitrogen vacancy diamond nano-crystals [341] (nano-diamonds), which are also sensitive to voltage [342] and to magnetic fields [343], and are amenable to optical control and fluorescent readout of the spin state (although the 2P cross-section of the  $(N-V)^-$  center appears to be relatively low [344]).

Hybrid systems New mergers of input, sensing, and readout modalities can work around complex engineering constraints. Electrical or acoustic sensors could be used with optical [345] (e.g., fiber) or ultrasonic readouts and power supplies. An MRI machine could interact with embedded electrical circuits powered by neural activity [346]. Linking electrical recording with embedded optical microscopies or other spatially-resolved methods could circumvent the limits of purely electrical spike sorting. Optical techniques such as holography or 4D light fields could generalize to ultrasound or microwave implementations. Consideration of analogies and synergies between fields suggests a combinatorial space of possibilities.

Our goal here has not been to pick winning technologies (which may not yet have been conceived), but to aid a multi-disciplinary community of researchers in analyzing the problem. The challenge of observing the real-time operation of entire mammalian brains requires a return to first principles, and a fundamental reconsideration of the architectures of neural recording systems. We hope that knowledge of the constraints governing scalable neural recording will enable the invention of entirely new, transformative approaches.

### 6 ACKNOWLEDGMENTS

We thank K. Esvelt for helpful discussions on bioluminescent proteins; D. Boysen for help on the fuel cell calculations; R. Tucker and E. Yablonovitch (http://www.e3s-center.org) for helpful discussions on the energy efficiency of CMOS; C. Xu and C. Schaffer for data on optical attenuation lengths; T. Dean and the participants in his CS379C course at Stanford/Google, including Chris Uhlik and Akram Sadek, for helpful discussions and informative content in the discussion notes (http://www.stanford.edu/class/cs379c/); and L. Wood, R. Koene, S. Rezchikov, A. Bansal, J. Lovelock, A. Payne, R. Barish, N. Donoghue, J. Pillow, W. Shih, P. Yin and J. Hewitt for helpful discussions and feedback on earlier drafts.

A. Marblestone is supported by the Fannie and John Hertz Foundation fellowship. D. Dalrymple is supported by the Thiel Foundation. K. Kording is funded in part by the Chicago Biomedical Consortium with support from the Searle Funds at The Chicago Community Trust. E. Boyden is supported by the National Institutes of Health (NIH), the National Science Foundation, the MIT McGovern Institute and Media Lab, the New York Stem Cell Foundation Robertson Investigator Award, the Human Frontiers Science

Program, and the Paul Allen Distinguished Investigator in Neuroscience Award. B. Stranges, B. Zamft, R. Kalhor and G. Church acknowledge support from the Office of Naval Research and the NIH Centers of Excellence in Genomic Science. M. Shapiro is supported by the Miller Research Institute, the Burroughs Wellcome Career Award at the Scientific Interface and the W.M. Keck Foundation.

## REFERENCES

- [1] ALIVISATOS A Paul, Miyoung CHUN, George M CHURCH, Ralph J GREENSPAN, Michael L ROUKES, and Rafael YUSTE. "The brain activity map project and the challenge of functional connectomics," *Neuron* 74.6 (2012), 970–4, URL: http://academiccommons.columbia.edu/catalog/ac:147969 (cited on pp. 1, 13, 28).
- [2] BANSAL Arjun K, Wilson TRUCCOLO, Carlos E VARGAS-IRWIN, and John P DONOGHUE. "Decoding 3-d reach and grasp from hybrid signals in motor and premotor cortices: spikes, multiunit activity and local field potentials," *Journal of Neurophysiology* 107.3 (2012), 1337–55 (cited on p. 1).
- [3] GERHARD Felipe, Tilman KISPERSKY, Gabrielle J GUTIERREZ, Eve MARDER, Mark KRAMER, and Uri EDEN. "Successful reconstruction of a physiological circuit with known connectivity from spiking activity alone," *PLoS computational biology* 9.7 (2013), e1003138 (cited on p. 1).
- [4] STEVENSON Ian H. and Konrad P. KORDING. "How advances in neural recording affect data analysis," *Nature Neuroscience* 14 (online Jan. 26, 2011), 139–42, DOI: 10.1038/nn.2731 (cited on p. 1).
- [5] ZAMFT Bradley M., Adam H. MARBLESTONE, Konrad P. KORDING, Daniel SCHMIDT, Daniel MARTIN-ALARCON, et al. "Measuring cation dependent DNA polymerase fidelity landscapes by deep sequencing," *PLoS ONE* 7.8 (Aug. 22, 2012), e43876, DOI: 10.1371/journal.pone.0043876 (cited on pp. 1, 2, 27).
- [6] GLASER Joshua I., Bradley M. ZAMFT, Adam H. MARBLESTONE, Jeffrey R. MOFFITT, Keith TYO, et al. "Statistical analysis of molecular signal recording," *PLoS Computational Biology* (2013), DOI: 10.1371/journal.pcbi.1003145 (cited on pp. 1, 2, 8, 27).
- [7] KORDING Konrad P. "Of toasters and molecular ticker tapes," *PLoS Computational Biology* 7.12 (Dec. 29, 2011), e1002291, DOI: 10.1371/journal.pcbi.1002291 (cited on pp. 1, 2).
- [8] AHRENS Misha B., Michael B. ORGER, Drew N. ROBSON, Jennifer M. LI, and Philipp J. KELLER. "Whole-brain functional imaging at cellular resolution using light-sheet microscopy," *Nature Methods* 10 (online Mar. 18, 2013), 413–20, DOI: 10.1038/nmeth.2434 (cited on pp. 2, 12).
- [9] ZIV Yaniv, Laurie D. BURNS, Eric D. COCKER, Elizabeth O. HAMEL, Kunal K. GHOSH, et al. "Long-term dynamics of CA1 hippocampal place codes," *Nature Neuroscience* 16 (online Feb. 10, 2013), 264–6, DOI: 10.1038/nn.3329 (cited on p. 2).
- [10] KANDEL Eric R., Henry MARKRAM, Paul M. MATTHEWS, Rafael YUSTE, and Christof KOCH. "Neuroscience thinks big (and collaboratively)," *Nature Reviews Neuroscience* 14 (9 2013), DOI: 10.1038/nrn3578 (cited on p. 2).
- [11] VINCENT Trevor J., Jonathan D. THIESSEN, Laryssa M. KURJEWICZ, Shelley L. GERMSCHEID, Allan J. TURNER, et al. "Longitudinal brain size measurements in APP/PS1 transgenic mice," *Magnetic Resonance Insights* 4 (Oct. 2010), 19–26, DOI: 10.4137/MRI.S5885 (cited on p. 3).
- [12] **BRAITENBERG** Valentino and Almut SCHÜZ. *Anatomy of the cortex: Statistics and geometry.* Springer-Verlag Publishing, 1991 (cited on p. 3).
- [13] AZEVEDO Frederico A. C., Ludmila R. B. CARVALHO, Lea T. GRINBERG, José MARCELO FARFEL, Renata E. L. FERRETTI, et al. "Equal numbers of neuronal and nonneuronal cells make the human brain an isometrically scaled-up primate brain," *Journal of Comparative Neurology* 513.5 (Feb. 2009), 532–41, DOI: 10.1002/cne.21974 (cited on p. 3).
- [14] ALLEN John S., Hanna DAMASIO, and Thomas J. GRABOWSKI.

  "Normal neuroanatomical variation in the human brain: an MRI-volumetric study,"

  American Journal of Physical Anthropology 118.4 (Aug. 2002), 341–58, DOI: 10.1002/ajpa.10092 (cited on p. 3).
- [15] SARPESHKAR Rahul. *Ultra Low Power Bioelectronics: Fundamentals, Biomedical Applications, and Bio-inspired Systems,* New York: Cambridge University Press, 2010, 747–8 (cited on p. 3).
- [16] HERCULANO-HOUZEL Suzana. "Scaling of brain metabolism with a fixed energy budget per neuron: implications for neuronal activity, plasticity, and evolution," *PLoS ONE* 6 (2011), e17514, DOI: 10.1371/journal.pone.0017514 (cited on p. 3).

- [17] SPEAKMAN John R. "Measuring energy metabolism in the mouse theoretical, practical, and analytical considerations," Frontiers in Integrative Physiology 4.34 (Mar. 14, 2013), DOI: 10.3389/fphys.2013.00034 (cited on pp. 3, 15).
- [18] HARRIS Julia J, Renaud JOLIVET, and David ATTWELL. "Synaptic energy use and supply," *Neuron* 75.5 (2012), 762–77 (cited on p. 3).
- [19] GITTIS Aryn H., Setareh H. MOGHADAM, and Sascha du LAC. "Mechanisms of sustained high firing rates in two classes of vestibular nucleus neurons: differential contributions of resurgent Na, Kv3, and BK currents,"

  Journal of Neurophysiology 104.3 (June 2010), 1625–34, DOI: 10.1152/jn.00378.2010 (cited on p. 3).
- [20] CHADDERTON Paul, Troy W MARGRIE, and Michael HÄUSSER.

  "Integration of quanta in cerebellar granule cells during sensory processing," *Nature* 428.6985 (2004), 856–60 (cited on p. 3).
- [21] LENNIE Peter. "The cost of cortical computation," Current biology 13.6 (2003), 493–7 (cited on p. 3).
- [22] HOWARTH Clare, Padraig GLEESON, and David ATTWELL. "Updated energy budgets for neural computation in the neocortex and cerebellum," *Journal of Cerebral Blood Flow & Metabolism* 32.7 (2012), 1222–32 (cited on p. 3).
- [23] SHOHAM Shy, Daniel H. O'CONNOR, and Ronen SEGEV.

  "How silent is the brain: is there a dark matter problem in neuroscience?"

  Journal of Comparative Physiology A 192.8 (2006), 777–84, ISSN: 0340-7594, DOI: 10.1007/s00359-006-0117-6 (cited on p. 3).
- [24] BARTH Alison L. and James F.A. POULET. "Experimental evidence for sparse firing in the neocortex," *Trends in Neurosciences* 35.6 (2012), 345 –355, DOI: 10.1016/j.tins.2012.03.008 (cited on p. 3).
- [25] ROXIN Alex, Nicolas BRUNEL, David HANSEL, Gianluigi MONGILLO, and Carl van VREESWIJK.

  "On the distribution of firing rates in networks of cortical neurons," *The Journal of Neuroscience* 31.45 (2011), 16217–26 (cited on p. 3).
- [26] O'CONNOR Daniel H, Simon P PERON, Daniel HUBER, and Karel SVOBODA.

  "Neural activity in barrel cortex underlying vibrissa-based object localization in mice," *Neuron* 67.6 (2010), 1048–61 (cited on p. 3).
- [27] HROMÁDKA Tomáš, Michael R DEWEESE, and Anthony M ZADOR. "Sparse representation of sounds in the unanesthetized auditory cortex," *PLoS biology* 6.1 (2008), e16 (cited on p. 3).
- [28] SHAFI M, Y ZHOU, J QUINTANA, C CHOW, J FUSTER, and M BODNER. "Variability in neuronal activity in primate cortex during working memory tasks," *Neuroscience* 146.3 (2007), 1082–108 (cited on p. 3).
- [29] MARKRAM Henry, Wulfram GERSTNER, and Per Jesper SJÖSTRÖM. "A history of spike-timing-dependent plasticity," Frontiers in synaptic neuroscience 3 (2011) (cited on p. 3).
- [30] BABADI Baktash and Larry F. ABBOTT.

  "Pairwise analysis can account for network structures arising from spike-timing dependent plasticity,"

  PLoS Comput Biol 9.2 (Feb. 2013), e1002906, DOI: 10.1371/journal.pcbi.1002906 (cited on p. 3).
- [31] TAILLEFUMIER Thibaud and Marcelo O. MAGNASCO. "A phase transition in the first passage of a brownian process through a fluctuating boundary with implications for neural coding," *Proceedings of the National Academy of Sciences* (2013), DOI: 10.1073/pnas.1212479110 (cited on p. 3).
- [32] GIRE David H., Diego RESTREPO, Terrence J. SEJNOWSKI, Charles GREER, Juan A. DE CARLOS, and Laura LOPEZ-MASCARAQUE. "Temporal processing in the olfactory system: can we see a smell?"

  Neuron 78.3 (2013), 416 –432, ISSN: 0896-6273, DOI: 10.1016/j.neuron.2013.04.033 (cited on p. 3).
- [33] SCHNEIDMAN Elad, Michael J BERRY, Ronen SEGEV, and William BIALEK. "Weak pairwise correlations imply strongly correlated network states in a neural population," *Nature* 440.7087 (2006), 1007–12 (cited on p. 3).
- [34] JONASZ Miroslaw. "Absorption coefficient of water: data sources," *Topics in Particle and Dispersion Science* (2007), URL: http://www.tpdsci.com/Tpc/AbsCfOfWaterDat.php (visited on 06/16/2013) (cited on p. 4).
- [35] KOU Linhong, Daniel LABRIE, and Petr CHYLEK. "Refractive indices of water and ice in the 0.65- to 2.5-µm spectral range," Applied Optics 32.19 (July 1, 1993), 3531–40, DOI: 10.1364/A0.32.003531 (cited on pp. 3, 4).
- [36] DOBBING John and Jean SANDS. "Quantitative growth and development of human brain," *Archives of Disease in Childhood* 48.10 (Oct. 1973), 757–67, DOI: 10.1136/adc.48.10.757 (cited on p. 3).
- [37] FATOUROS Panos P. and Anthony MARMAROU.

  "Use of magnetic resonance imaging for in vivo measurements of water content in human brain: method and normal values,"

  Journal of Neurosurgery 90.1 (Jan. 1999), 109–15, DOI: 10.3171/jns.1999.90.1.0109 (cited on p. 3).

- [38] WILT Brian A., Laurie D. BURNS, Eric Tatt WEI HO, Kunal K. GHOSH, Eran A. MUKAMEL, and Mark J. SCHNITZER. "Advances in light microscopy for neuroscience," *Annual Review of Neuroscience* 32.1 (2009), PMID: 19555292, 435–506, DOI: 10.1146/annurev.neuro.051508.135540 (cited on pp. 3, 13).
- [39] HORTON Nicholas G., Ke WANG, Demirhan KOBAT, Catharine G. CLARK, Frank W. WISE, et al. "In vivo three-photon microscopy of subcortical structures within an intact mouse brain," Nature Photonics 7 (online Jan. 20, 2013), 205–9, DOI: 10.1038/nphoton.2012.336 (cited on pp. 3, 13).
- [40] GABRIEL S, R.W. LAU, and C GABRIEL.

  "The dielectric properties of biological tissues: iii. parametric models for the dielectric spectrum of tissues,"

  Phys. Med. Biol 41.11 (Nov. 1, 1996), URL: http://niremf.ifac.cnr.it/tissprop/ (cited on p. 3).
- [41] STRONG Steven P., Roland KOBERLE, Robert R. de RUYTER VAN STEVENINCK, and William BIALEK. "Entropy and information in neural spike trains," *Physical Review Letters* 80 (Jan. 5, 1998), 197–200, DOI: 10.1103/PhysRevLett.80.197 (cited on p. 5).
- [42] WOLF Patrick D. Thermal Considerations for the Design of an Implanted Cortical Brain-Machine Interface (BMI) In: Reichert WM, editor. Indwelling Neural Implants: Strategies for Contending with the In Vivo Environment.

  Boca Raton (FL): CRC Press, 2008, URL: http://www.ncbi.nlm.nih.gov/books/NBK3932/ (cited on pp. 5, 6).
- [43] SOTERO Roberto C. and Yasser ITURRIA-MEDINA.

  "From blood oxygenation level dependent (BOLD) signals to brain temperature maps,"

  Bulletin of Mathematical Biology 73.11 (Nov. 2011), 2731–47, DOI: 10.1007/s11538-011-9645-5 (cited on p. 5).
- [44] LAZZI Gianluca. "Thermal effects of bioimplants," Engineering in Medicine and Biology Magazine, IEEE 24.5 (2005), 75–81, ISSN: 0739-5175, DOI: 10.1109/MEMB.2005.1511503 (cited on pp. 5, 6).
- [45] SMITH Katisha D and Liang ZHU.

  "Brain hypothermia induced by cold spinal fluid using a torso cooling pad: theoretical analyses,"

  Medical & biological engineering & computing 48.8 (2010), 783–91 (cited on p. 5).
- [46] ILIFF Jeffrey J, Minghuan WANG, Yonghong LIAO, Benjamin A PLOGG, Weiguo PENG, et al. "A paravascular pathway facilitates csf flow through the brain parenchyma and the clearance of interstitial solutes, including amyloid beta," *Sci Transl Med* 4.147 (2012), 147ra111 (cited on p. 5).
- [47] SUKSTANSKII AL and DA YABLONSKIY. "An analytical model of temperature regulation in human head," *Journal of thermal biology* 29.7 (2004), 583–7 (cited on p. 6).
- [48] SUKSTANSKII Alexander L and Dmitriy A YABLONSKIY.

  "Theoretical model of temperature regulation in the brain during changes in functional activity,"

  Proceedings of the National Academy of Sciences 103.32 (2006), 12144–9 (cited on p. 6).
- [49] SUKSTANSKII AL and DA YABLONSKIY. "Theoretical limits on brain cooling by external head cooling devices," *European journal of applied physiology* 101.1 (2007), 41–9 (cited on p. 6).
- [50] **KALMBACH** Abigail S and Jack WATERS. "Brain surface temperature under a craniotomy," *Journal of Neurophysiology* 108.11 (2012), 3138–46 (cited on p. 6).
- [51] MCELLIGOTT JG and Ro MELZACK. "Localized thermal changes evoked in the brain by visual and auditory stimulation," *Experimental neurology* 17.3 (1967), 293–312 (cited on p. 6).
- [52] TRÜBEL Hubert KF, Laura I SACOLICK, and Fahmeed HYDER.

  "Regional temperature changes in the brain during somatosensory stimulation,"

  Journal of Cerebral Blood Flow & Metabolism 26.1 (2005), 68–78 (cited on p. 6).
- [53] POLDERMAN Kees H. "Application of therapeutic hypothermia in the icu: opportunities and pitfalls of a promising treatment modality. part 1: indications and evidence," *Intensive Care Medicine* 30.4 (Apr. 1, 2004), DOI: 10.1007/s00134-003-2152-x (cited on p. 6).
- [54] "C95.1-2005. ieee standard for safety levels with respect to human exposure to radio frequency electromagnetic fields, 3 khz to 300 ghz," (2006), URL: http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1626482 (cited on p. 6).
- [55] "Guidance for industry and fda staff: information for manufacturers seeking marketing clearance of diagnostic ultrasound systems and transducers," (2008), URL: http://www.fda.gov/downloads/MedicalDevices/
  DeviceRegulationandGuidance/GuidanceDocuments/ucm070911.pdf (cited on p. 6).
- [56] *IEC 60825*, Safety of laser products, *Equipment classification and requirements*, 2nd ed., Geneva, Switzerland: International Electrotechnical Commission, Mar. 30, 2007 (cited on p. 6).
- [57] SHAPIRO Mikhail G., Kazuaki HOMMA, Sebastian VILLARREAL, Claus-Peter RICHTER, and Francisco BEZANILLA. "Infrared light excites cells by changing their electrical capacitance," *Nature Communications* 3.736 (Mar. 13, 2012), DOI: 10.1038/ncomms1742 (cited on p. 6).

- [58] SHAPIRO Mikhail G., Michael F. PRIEST, Peter H. SIEGEL, and Francisco BEZANILLA. "Temperature-mediated effects of millimeter wave stimulation on membrane protein function," *Biophysical Journal* 104.2 (Jan. 2013), 679a, DOI: 10.1016/j.bpj.2012.11.3747 (cited on p. 6).
- [59] RIZK Michael, Chad A BOSSETTI, Thomas A JOCHUM, Stephen H CALLENDER, Miguel AL NICOLELIS, et al. "A fully implantable 96-channel neural data acquisition system," *Journal of Neural Engineering* 6.2 (2009), 6002 (cited on p. 6).
- [60] RIBEIRO Sidarta, Damien GERVASONI, Ernesto S SOARES, Yi ZHOU, Shih-Chieh LIN, et al. "Long-lasting novelty-induced neuronal reverberation during slow-wave sleep in multiple forebrain areas," *PLoS Biol* 2.1 (Jan. 2004), e24, DOI: 10.1371/journal.pbio.0020024 (cited on p. 6).
- [61] NICOLELIS Miguel A. L., Dragan DIMITROV, Jose M. CARMENA, Roy CRIST, Gary LEHEW, et al. "Chronic, multisite, multielectrode recordings in macaque monkeys," *Proceedings of the National Academy of Sciences* 100.19 (2003), 11041–6, DOI: 10.1073/pnas.1934665100 (cited on pp. 6, 7, 10).
- [62] BLINDER Pablo, Philbert S TSAI, John P KAUFHOLD, Per M KNUTSEN, Harry SUHL, and David KLEINFELD. "The cortical angiome: an interconnected vascular network with noncolumnar patterns of blood flow," *Nature Neuroscience* (June 9, 2013), DOI: 10.1038/nn.3426 (cited on p. 6).
- [63] POLIKOV Vadim S, Patrick A TRESCO, and William M REICHERT.

  "Response of brain tissue to chronically implanted neural electrodes," *Journal of neuroscience methods* 148.1 (2005), 1–18 (cited on p. 7).
- [64] WARD Matthew P, Pooja RAJDEV, Casey ELLISON, and Pedro P IRAZOQUI. "Toward a comparison of microelectrodes for acute and chronic recordings," *Brain research* 1282 (2009), 183–200 (cited on p. 7).
- [65] TAUB Aryeh H, Roni HOGRI, Ari MAGAL, Matti MINTZ, and Yosi SHACHAM-DIAMAND. "Bioactive anti-inflammatory coating for chronic neural electrodes," *Journal of Biomedical Materials Research Part A* 100.7 (2012), 1854–8 (cited on p. 7).
- [66] REICHERT William M, Wei HE, and Ravi V BELLAMKONDA. "A molecular perspective on understanding and modulating the performance of chronic central nervous system (cns) recording electrodes," (2008) (cited on p. 7).
- [67] REICHERT William M. Indwelling neural implants: strategies for contending with the in vivo environment, CRC Press, 2010 (cited on p. 7).
- [68] SUNER Selim, Matthew R FELLOWS, Carlos VARGAS-IRWIN, Gordon Kenji NAKATA, and John P DONOGHUE. "Reliability of signals from a chronically implanted, silicon-based electrode array in non-human primate primary motor cortex," *Neural Systems and Rehabilitation Engineering, IEEE Transactions on* 13.4 (2005), 524–41 (cited on p. 7).
- [69] VETTER Rio J, Justin C WILLIAMS, Jamille F HETKE, Elizabeth A NUNAMAKER, and Daryl R KIPKE. "Chronic neural recording using silicon-substrate microelectrode arrays implanted in cerebral cortex," *Biomedical Engineering, IEEE Transactions on* 51.6 (2004), 896–904 (cited on p. 7).
- [70] SEO Dongjin, Jose M. CARMENA, Jan M. RABAEY, Elad ALON, and Michel M. MAHARBIZ. "Neural dust: an ultrasonic, low power solution for chronic brain-machine interfaces," *ArXiv preprint* (2013), URL: http://arxiv.org/abs/1307.2196 (cited on pp. 8, 19, 23, 29).
- [71] REIMANN Michael W, Costas A ANASTASSIOU, Rodrigo PERIN, Sean L HILL, Henry MARKRAM, and Christof KOCH. "A biophysically detailed model of neocortical local field potentials predicts the critical role of active membrane currents," *Neuron* 79.2 (2013), 375–90 (cited on pp. 7, 30).
- [72] **BUZSÁKI** György, Costas A ANASTASSIOU, and Christof KOCH. "The origin of extracellular fields and currents eeg, ecog, lfp and spikes," *Nature Reviews Neuroscience* 13.6 (2012), 407–20 (cited on p. 7).
- [73] ANASTASSIOU Costas A, Sean M MONTGOMERY, Mauricio BARAHONA, György BUZSÁKI, and Christof KOCH. "The effect of spatially inhomogeneous extracellular electric fields on neurons," *The Journal of neuroscience* 30.5 (2010), 1925–36 (cited on p. 7).
- [74] SEGEV Ronen, Joe GOODHOUSE, Jason PUCHALLA, and Michael J. BERRY II. "Recording spikes from a large fraction of the ganglion cells in a retinal patch," *Nature Neuroscience* 7.10 (Oct. 2004), 1155–62, DOI: 10.1038/nn1323 (cited on pp. 7, 11).
- [75] GRAY Charles M., Pedro E. MALDONADO, Matthew WILSON, and Bruce MCNAUGHTON. "Tetrodes markedly improve the reliability and yield of multiple single-unit isolation from multi-unit recordings in cat striate cortex," *Journal of Neuroscience Methods* 63.1–2 (Dec. 1995), 43–54, DOI: 10/ddmxdh (cited on pp. 7, 9).

- [76] GOLD Carl, Darrell A. HENZE, and Christof KOCH.

  "Using extracellular action potential recordings to constrain compartmental models,"

  Journal of Computational Neuroscience 23 (Aug. 23, 2007), 39–58, DOI: 10.1007/s10827-006-0018-2 (cited on p. 7).
- [77] ANASTASSIOU Costas A, Cyorgy BUZSAKI, and Christof KOCH. "Biophysics of extracellular spikes," *Principles of Neural Coding* (2013), 15 (cited on p. 7).
- [78] **BÉDARD** Claude, Helmut KRÖGER, and Alain DESTEXHE. "Modeling extracellular field potentials and the frequency-filtering properties of extracellular space," *Biophysical journal* 86.3 (2004), 1829–42 (cited on p. 7).
- [79] SAHANI Maneesh. "Latent variable models for neural data analysis,"
  PhD dissertation, California Institute of Technology, May 14, 1999,
  URL: http://www.gatsby.ucl.ac.uk/~maneesh/thesis/thesis.double.pdf (visited on 06/16/2013)
  (cited on pp. 7, 10).
- [80] CAMUÑAS-MESA Luis A. and Rodrigo QUIAN QUIROGA. "A detailed and fast model of extracellular recordings," *Neural Computation* 25.5 (May 2013), 1191–212, DOI: 10.1162/NECO\_a\_00433 (cited on pp. 9, 10).
- [81] DU Jiangang, Timothy J. BLANCHE, Reid R. HARRISON, Henry A. LESTER, and Sotiris C. MASMANIDIS. "Multiplexed, high density electrophysiology with nanofabricated neural probes," *PLoS ONE* 6.10 (Oct. 12, 2011), e26204, DOI: 10.1371/journal.pone.0026204 (cited on pp. 9, 11).
- [82] HENZE Darrell A., Zsolt BORHEGYI, Jozsef CSICSVARI, Akira MAMIYA, Kenneth D. HARRIS, and György BUZSÁKI. "Intracellular features predicted by extracellular recordings in the hippocampus in vivo,"

  Journal of Neurophysiology 84.1 (2000), 390–400, eprint: http://jn.physiology.org/content/84/1/390.full.pdf+html (cited on p. 9).
- [83] BUZSÁKI György. "Large-scale recording of neuronal ensembles," *Nature Neuroscience* 7 (5 May 2004), DOI: 10.1038/nn1233 (cited on p. 9).
- [84] WESSBERG Johan, Christopher R. STAMBAUGH, Jerald D. KRALIK, Pamela D. BECK, Mark LAUBACH, et al. "Real-time prediction of hand trajectory by ensembles of cortical neurons in primates," *Nature* 408 (6810 Nov. 16, 2000), DOI: 10.1038/35042582 (cited on p. 9).
- [85] CARMENA Jose M, Mikhail A LEBEDEV, Roy E CRIST, Joseph E O'DOHERTY, David M SANTUCCI, et al. "Learning to control a brain-machine interface for reaching and grasping by primates," *PLoS Biol* 1.2 (Oct. 2003), e42, DOI: 10.1371/journal.pbio.0000042 (cited on p. 9).
- [86] KORALEK Aaron C., Xin JIN, John D. LONG II, Rui M. COSTA, and Jose M. CARMENA. "Corticostriatal plasticity is necessary for learning intentional neuroprosthetic skills," *Nature* 483 (7389 Mar. 15, 2012), DOI: 10.1038/nature10845 (cited on p. 9).
- [87] JIN Xin and Rui M. COSTA. "Start/stop signals emerge in nigrostriatal circuits during sequence learning," *Nature* 466 (7305 July 22, 2010), DOI: 10.1038/nature09263 (cited on p. 9).
- [88] FEE MICHALE S, PARTHA P MITRA, and DAVID KLEINFELD.

  "Variability of extracellular spike waveforms of cortical neurons," *Journal of neurophysiology* 76.6 (1996), 3823–33 (cited on p. 9).
- [89] STRATTON Peter, Allen CHEUNG, Janet WILES, Eugene KIYATKIN, Pankaj SAH, and François WINDELS. "Action potential waveform variability limits multi-unit separation in freely behaving rats," *PloS one* 7.6 (2012), e38482 (cited on p. 9).
- [90] COVER Thomas M. and Joy A. THOMAS. *Elements of Information Theory*, 2nd ed., Wiley Series in Telecommunications and Signal Processing, Hoboken, NJ: Wiley-Interscience, July 18, 2006, URL: http://www.amazon.com/dp/0471241954 (cited on pp. 10, 20).
- [91] MARRE Olivier, Dario AMODEI, Nikhil DESHMUKH, Kolia SADEGHI, Frederick SOO, et al. "Mapping a complete neural population in the retina," *The Journal of Neuroscience* 32.43 (2012), 14859–73 (cited on pp. 10, 11).
- [92] PILLOW Jonathan W, Jonathon SHLENS, E J CHICHILNISKY, and Eero P SIMONCELLI. "A model-based spike sorting algorithm for removing correlation artifacts in multi-neuron recordings.," *PLoS ONE* 8.5 (2013), e62123 (cited on pp. 10, 11).
- [93] COSTA Rui M, Dana COHEN, and Miguel AL NICOLELIS. "Differential corticostriatal plasticity during fast and slow motor skill learning in mice," *Current Biology* 14.13 (2004), 1124–34 (cited on p. 10).

- [94] **PEDREIRA** Carlos, Juan MARTINEZ, Matias J. ISON, and Rodrigo QUIAN QUIROGA. "How many neurons can we see with current spike sorting algorithms?" *Journal of Neuroscience Methods* 211.1 (Oct. 2012), 58–65 (cited on p. 10).
- [95] GE Di, Eric LE CARPENTIER, Jérôme IDIER, and Dario FARINA. "Spike sorting by stochastic simulation,"

  IEEE transactions on neural systems and rehabilitation engineering: a publication of the IEEE Engineering in Medicine and Biology Society 19.3 (June 2011), 249–59 (cited on p. 11).
- [96] PRENTICE Jason S, Jan HOMANN, Kristina D SIMMONS, Gasper TKACIK, Vijay BALASUBRAMANIAN, and Philip C NELSON. "Fast, scalable, bayesian spike identification for multi-electrode arrays," *PLoS ONE* 6.7 (2011), e19884 (cited on p. 11).
- [97] LLINÁS Rodolfo R., Kerry D. WALTON, Masayuki NAKAO, Ian HUNTER, and Patrick A. ANQUETIL. "Neuro-vascular central nervous recording/stimulating system: Using nanotechnology probes," Journal of Nanoparticle Research 7.2–3 (June 2005), 111–27, DOI: 10.1007/s11051-005-3134-4 (cited on p. 11).
- [98] SCHMIDT Richard F. and Gerhard THEWS, eds. *Human Physiology*, 2nd ed., New York: Springer-Verlag, Sept. 1989, URL: http://www.amazon.com/dp/0387194320/ (cited on p. 11).
- [99] LOFFREDO Francesco and Richard T. LEE. "Therapeutic vasculogenesis: It takes two," *Circulation Research* 103.2 (July 18, 2008), 128–30, DOI: 10.1161/CIRCRESAHA.108.180604 (cited on p. 11).
- [100] FREITAS JR. Robert A. Nanomedicine, Volume I: Basic Capabilities, Georgetown, Texas: Landes Bioscience, 1999, URL: http://www.nanomedicine.com/NMI/8.2.1.2.htm (cited on p. 11).
- [101] JADHAV Amol D., Ivan AIMO, Daniel COHEN, Peter LEDOCHOWITSCH, and Michel M. MAHARBIZ. "Cyborg eyes: Microfabricated neural interfaces implanted during the development of insect sensory organs produce stable neurorecordings in the adult," *Micro Electro Mechanical Systems*, IEEE 25th International Conference on, IEEE MEMS 2012 (Paris), Jan. 29–Feb. 2, 2012, 937–40, DOI: 10.1109/MEMSYS.2012.6170340 (cited on p. 11).
- [102] JENSEN Winnie, Ulrich G. HOFMANN, and Ken YOSHIDA.

  "Assessment of subdural insertion force of single-tine microelectrodes in rat cerebral cortex,"

  Engineering in Medicine and Biology Society, Proceedings of the 25th Annual International Conference of the IEEE,

  EMBC'03 (Cancun, Mexico), vol. 3, Sept. 17–21, 2003, 2168–71, DOI: 10.1109/IEMBS.2003.1280170 (cited on p. 11).
- [103] NAJAFI Khalil and Jamille F. HETKE. "Strength characterization of silicon microprobes in neurophysiological tissues," *IEEE Transactions on Biomedical Engineering* 37.5 (May 1990), 474–81, DOI: 10.1109/10.55638 (cited on p. 11).
- [104] WEIBEL Douglas B., Piotr GARSTECKI, Declan RYAN, Willow R. DILUZIO, Michael MAYER, et al. "Microoxen: Microorganisms to move microscale loads," Proceedings of the National Academy of Sciences of the United States of America 102.34 (2005), 11963–7, DOI: 10.1073/pnas.0505481102 (cited on p. 11).
- [105] BAKKUM Douglas J, Urs FREY, Milos RADIVOJEVIC, Thomas L RUSSELL, Jan MÜLLER, et al. "Tracking axonal action potential propagation on a high-density microelectrode array across hundreds of sites," *Nature communications* 4 (2013) (cited on p. 12).
- [106] KODANDARAMAIAH Suhasa B, Giovanni Talei FRANZESI, Brian Y CHOW, Edward S BOYDEN, and Craig R FOREST. "Automated whole-cell patch-clamp electrophysiology of neurons in vivo," *Nature Methods* 9 (6), DOI: 10.1038/nmeth.1993 (cited on p. 12).
- [107] SMETTERS Diana, Ania MAJEWSKA, and Rafael YUSTE.

  "Detecting action potentials in neuronal populations with calcium imaging,"

  METHODS: A Companion to Methods in Enzymology 18 (2 June 1999), DOI: 10.1006/meth.1999.0774 (cited on p. 12).
- [108] SUN Xiaonan R, Aleksandra BADURA, Diego A PACHECO, Laura A LYNCH, Eve R SCHNEIDER, et al. "Fast gramps for improved tracking of neuronal activity," *Nature communications* 4 (2013) (cited on p. 12).
- [109] HIGLEY Michael J and Bernardo L SABATINI.

  "Calcium signaling in dendrites and spines: practical and functional considerations," *Neuron* 59.6 (2008), 902–13 (cited on p. 12).
- [110] BARNETT Lauren, Jelena PLATISA, Marko POPOVIC, Vincent A. PIERIBONE, and Thomas HUGHES. "A fluorescent, genetically-encoded voltage probe capable of resolving action potentials," *PLoS ONE* 7.9 (Sept. 2012), e43454, DOI: 10.1371/journal.pone.0043454 (cited on p. 12).
- [111] KRALJ Joel M, Adam D DOUGLASS, Daniel R HOCHBAUM, Dougal MACLAURIN, and Adam E COHEN. "Optical recording of action potentials in mammalian neurons using a microbial rhodopsin," *Nature Methods* 9 (1), DOI: 10.1038/nmeth.1782 (cited on p. 12).
- [112] GONG Yiyang, Jin Zhong LI, and Mark J SCHNITZER. "Enhanced archaerhodopsin fluorescent protein voltage indicators," *PLOS ONE* 8.6 (2013), e66959 (cited on p. 12).

- [113] STORACE Douglas. A., Uhna. SUNG, Jelena. PLATISA, Lawrence. B. COHEN, and Vincent. A. PIERIBONE. "In Vivo Imaging of Odor-Evoked Responses in the Olfactory Bulb using Arclight, a Novel Fp Voltage Probe," *Biophysical Journal* 104 (Jan. 2013), 679, DOI: 10.1016/j.bpj.2012.11.3751 (cited on p. 12).
- [114] CAO Guan, Jelena PLATISA, Vincent A PIERIBONE, Davide RACCUGLIA, Michael KUNST, and Michael N NITABACH. "Genetically targeted optical electrophysiology in intact neural circuits," *Cell* (2013) (cited on p. 12).
- [115] AKEMANN Walther, Mari SASAKI, Hiroki MUTOH, Takeshi IMAMURA, Naoki HONKURA, and Thomas KNÖPFEL. "Two-photon voltage imaging using a genetically encoded voltage indicator," *Scientific reports* 3 (2013) (cited on p. 12).
- [116] SCANZIANI Massimo and Michael HÄUSSER. "Electrophysiology in the age of light," *Nature* 461.7266 (2009), 930–9 (cited on p. 12).
- [117] QUIRIN Sean, Darcy S PETERKA, and Rafael YUSTE.

  "Instantaneous three-dimensional sensing using spatial light modulator illumination with extended depth of field imaging,"

  Optics express 21.13 (2013), 16007–21 (cited on p. 12).
- [118] LEVOY Marc, Zhengyun ZHANG, and Ian MCDOWALL. "Recording and controlling the 4d light field in a microscope using microlens arrays," *Journal of Microscopy* 235.2 (2009), 144–62 (cited on p. 12).
- [119] **BROXTON** Michael, Logan GROSENICK, Samuel YANG, Noy COHEN, Aaron ANDALMAN, et al. *Wave Optics Theory and 3-D Deconvolution for the Light Field Microscope*, 2013 (cited on p. 12).
- [120] ABRAHAMSSON Sara, Jiji CHEN, Bassam HAJJ, Sjoerd STALLINGA, Alexander Y KATSOV, et al. "Fast multicolor 3d imaging using aberration-corrected multifocus microscopy," *Nature methods* 10.1 (2012), 60–3 (cited on p. 12).
- [121] DOWSKI Edward R, W Thomas CATHEY, et al. "Extended depth of field through wave-front coding," *Applied Optics* 34.11 (1995), 1859–66 (cited on p. 12).
- [122] WANG Ying Min, Benjamin JUDKEWITZ, Charles A. DIMARZIO, and Changhuei YANG. "Deep-tissue focal fluorescence imaging with digitally time-reversed ultrasound-encoded light," *Nature Communications* 3.928 (June 26, 2012), DOI: 10.1038/ncomms1925 (cited on pp. 12, 13, 29).
- [123] DIEBOLD Eric D, Brandon W BUCKLEY, Daniel R GOSSETT, and Bahram JALALI. "Digitally-synthesized beat frequency multiplexing for sub-millisecond fluorescence microscopy,"

  Novel Techniques in Microscopy, Optical Society of America, 2013, URL: http://arxiv.org/abs/1303.1156 (cited on pp. 12–14, 30).
- [124] **DUCROS** Mathieu, Yannick Goulam HOUSSEN, Jonathan BRADLEY, Vincent de SARS, and Serge CHARPAK. "Encoded multisite two-photon microscopy," *Proceedings of the National Academy of Sciences* 110.32 (2013), 13138–43 (cited on pp. 12–14, 30).
- [125] YIN Stuart. "Frequency division multiplexed fluorescence confocal microscopy," *BIOPHOTONICS INTERNATIONAL* 13.12 (2006), 38 (cited on pp. 12, 13).
- [126] WU Fei, Xueqian ZHANG, Joseph Y CHEUNG, Kebin SHI, Zhiwen LIU, et al. "Frequency division multiplexed multichannel high-speed fluorescence confocal microscope," *Biophysical journal* 91.6 (2006), 2290–6 (cited on pp. 12, 13).
- [127] CHENG Adrian, J Tiago GONÇALVES, Peyman GOLSHANI, Katsushi ARISAKA, and Carlos PORTERA-CAILLIAU. "Simultaneous two-photon calcium imaging at different depths with spatiotemporal multiplexing,"

  Nature methods 8.2 (2011), 139–42 (cited on pp. 12–14).
- [128] WAKIN Michael B., Jason N. LASKA, Marco F. DUARTE, Dror BARON, Shriram SARVOTHAM, et al. "Compressive imaging for video representation and coding," *Picture Coding Symposium*, Proceedings of the 25th International, PCS'06 (Beijing, China), Apr. 24–26, 2006, URL: http://inside.mines.edu/~mwakin/papers/pcs-camera.pdf (visited on 06/16/2013) (cited on p. 12).
- [129] STUDER Vincent, Jérome BOBIN, Makhlad CHAHID, Hamed Shams MOUSAVI, Emmanuel CANDES, and Maxime DAHAN. "Compressive fluorescence microscopy for biological and hyperspectral imaging," *Proceedings of the National Academy of Sciences of the United States of America* 109.26 (June 26, 2012), E1679–E1687, DOI: 10.1073/pnas.1119511109 (cited on p. 12).
- [130] TIAN Nian, Qingchun GUO, Anle WANG, Dongli XU, and Ling FU. "Fluorescence ghost imaging with pseudothermal light," *Optics Letters* 36.16 (Aug. 15, 2011), 3302–4, DOI: 10.1364/0L.36.003302 (cited on p. 12).
- [131] SUN Baoqing, Matthew P. EDGAR, Richard BOWMAN, Liberty E. VITTERT, Stuart WELSH, et al. "3D computational imaging with single-pixel detectors," *Science* 340.6134 (May 17, 2013), 844–7, DOI: 10.1126/science.1234454 (cited on p. 12).

- [132] HUANG Gang, Hong JIANG, Kim MATTHEWS, and Paul WILFORD. "Lensless imaging by compressive sensing," *International Conference on Image Processing*, Proceedings of the 20th IEEE, ICIP 2013 (Melbourne, Australia), Sept. 15–18, 2013, forthcoming (cited on p. 12).
- [133] KOBAT Demirhan, Michael E. DURST, Nozomi NISHIMURA, Angela W. WONG, Chris B. SCHAFFER, and Chris XU. "Deep tissue multiphoton microscopy using longer wavelength excitation," *Optics Express* 17.16 (Aug. 3, 2009), 13354–64, DOI: 10.1364/0E.17.013354 (cited on p. 13).
- [134] KOBAT Demirhan, Nicholas G. HORTON, and Chris XU.

  "In vivo two-photon microscopy to 1.6-mm depth in mouse cortex," *Journal of Biomedical Optics* 16.10 (2011), DOI: 10.1117/1.3646209 (cited on p. 13).
- [135] FILONOV Grigory S., Kiryi D. PLATKEVICH, Li-Min TING, Jinghang ZHANG, Kami KIM, and Vladislav V. VERKHUSHA. "Bright and stable near-infrared fluorescent protein for *in vivo* imaging," *Nature Biotechnology* 29.8 (online July 17, 2011), 757–61, DOI: 10.1038/nbt.1918 (cited on p. 13).
- [136] SHCHERBAKOVA Daria M. and Vladislav V. VERKHUSHA.

  "Near-infrared fluorescent proteins for multicolor *in vivo* imaging," *Nature Methods* (online June 16, 2013),
  DOI: 10.1038/nmeth.2521, prepublished (cited on p. 13).
- [137] SHCHERBO Dmitry, Christopher S. MURPHY, Galina V. ERMAKOVA, Elena A. SOLOVIEVA, Tatiana V. CHEPURNYKH, et al. "Far-red fluorescent tags for protein imaging in living tissues," *Biochem J* 418.3 (2009), 567–74, DOI: 10.1042/BJ20081949 (cited on p. 13).
- [138] ALIVISATOS A. Paul, Anne M. ANDREWS, Edward S. BOYDEN, Miyoung CHUN, George M. CHURCH, et al. "Nanotools for neuroscience and brain activity mapping," *ACS Nano* 7.3 (Mar. 26, 2013), 1850–66, DOI: 10.1021/nn4012847 (cited on pp. 13, 28).
- [139] CONKEY Donald B., Antonio M. CARAVACA-AGUIRRE, and Rafael PIESTUN.

  "High-speed scattering medium characterization with application to focusing light through turbid media,"

  Optics Express 20.2 (Jan. 16, 2012), 1733–40, DOI: 10.1364/0E.20.001733 (cited on p. 13).
- [140] NIXON Micha, Ori KATZ, Eran SMALL, Yaron BROMBERG, Asher A. FRIESEM, et al. "Real-time wavefront-shaping through scattering media by all optical feedback," (Mar. 13, 2013), arXiv:1303.3161 [physics.optics] (cited on p. 13).
- [141] CHAIGNE Thomas, Ori KATZ, A. Claude BOCCARA, Mathias FINK, Emmanuel BOSSY, and Sylvain GIGAN. "Controlling light in scattering media noninvasively using the photo-acoustic transmission-matrix," (May 27, 2013), arXiv:1305.6246 [physics.optics] (cited on p. 13).
- [142] **BERTOLOTTI** Jacopo, Elbert G van PUTTEN, Christian BLUM, Ad LAGENDIJK, Willem L VOS, and Allard P MOSK. "Non-invasive imaging through opaque scattering layers," *Nature* 491.7423 (2012), 232–4 (cited on p. 13).
- [143] KATZ Ori, Eran SMALL, and Yaron SILBERBERG.

  "Looking around corners and through thin turbid layers in real time with scattered incoherent light,"

  Nature Photonics (2012) (cited on p. 13).
- [144] MCCABE David J, Ayhan TAJALLI, Dane R AUSTIN, Pierre BONDAREFF, Ian A WALMSLEY, et al. "Spatio-temporal focusing of an ultrafast pulse through a multiply scattering medium," *Nature Communications* 2 (2011), 447 (cited on p. 13).
- [145] KATZ Ori, Eran SMALL, Yaron BROMBERG, and Yaron SILBERBERG.

  "Focusing and compression of ultrashort pulses through scattering media," *Nature photonics* 5.6 (2011), 372–7 (cited on p. 13).
- [146] MAHALATI Reza Nasiri, Ruo Yu GU, and Joseph M. KAHN. "Resolution limits for imaging through multi-mode fiber," Optics Express 21.2 (Jan. 28, 2013), 1656–68, DOI: 10.1364/0E.21.001656 (cited on p. 13).
- [147] **ZORZOS** Anthony N., Edward S. BOYDEN, and Clifton G. FONSTAD. "Multiwaveguide implantable probe for light delivery to sets of distributed brain targets," *Optics Letters* 35.24 (Dec. 15, 2010), 4133–5, DOI: 10.1364/OL.35.004133 (cited on p. 13).
- [148] ZORZOS Anthony N., Jorg SCHOLVIN, Edward S. BOYDEN, and Clifton G. FONSTAD. "Three-dimensional multiwaveguide probe array for light delivery to distributed brain circuits," Optics Letters 37.23 (Dec. 1, 2012), 4841–3, DOI: 10.1364/0L.37.004841 (cited on p. 13).
- [149] ONO Takashi and Yutaka YANO.

  "Key technologies for terabit/second wdm systems with high spectral efficiency of over 1 bit/s/hz,"

  Quantum Electronics, IEEE Journal of 34.11 (1998), 2080–8 (cited on p. 13).

- [150] BOZINOVIC Nenad, Yang YUE, Yongxiong REN, Moshe TUR, Poul KRISTENSEN, et al. "Terabit-scale orbital angular momentum mode division multiplexing in fibers," *Science* 340.6140 (2013), 1545–8 (cited on p. 13).
- [151] MIYA Tetsuo, Y TERUNUMA, T HOSAKA, and Tadashi MIYASHITA. "Ultimate low-loss single-mode fibre at 1.55 um," *Electronics Letters* 15.4 (1979), 106–8 (cited on p. 13).
- [152] MURRAY Teresa A. and Michael J. LEVENE. "Singlet gradient index lens for deep in vivo multiphoton microscopy," *Journal of Biomedical Optics* 17.2 (Mar. 2, 2012), 021106, DOI: 10/mvt (cited on p. 13).
- [153] KANG Jeon Woong, Pilhan KIM, Carlo AMADEO ALONZO, Hyunsung PARK, and Seok H. YUN. "Two-photon microscopy by wavelength-swept pulses delivered through single-mode fiber," *Optics Letters* 35.2 (Jan. 15, 2010), 181–3, DOI: 10.1364/0L.35.000181 (cited on p. 13).
- [154] FLUSBERG Benjamin A., Eric D. COCKER, Wibool PIYAWATTANAMETHA, Jürgen C. JUNG, Eunice L. M. CHEUNG, and Mark J. SCHNITZER. "Fiber-optic fluorescence imaging," *Nature Methods* 2.12 (online Nov. 18, 2005), 941–50, DOI: 10.1038/nmeth820 (cited on p. 13).
- [155] MAHAN Gerald D., William E. ENGLER, Jerome J. TIEMANN, and Egidijus E. UZGIRIS. "Ultrasonic tagging of light: Theory,"

  Proceedings of the National Academy of Sciences of the United States of America 95.24 (Nov. 24, 1998), 14015–9, DOI: 10.1073/pnas.95.24.14015 (cited on p. 13).
- [156] VUČINIĆ Dejan and Terrence J. SEJNOWSKI.

  "A compact multiphoton 3d imaging system for recording fast neuronal activity," *PLoS ONE* 2.8 (Aug. 8, 2007), e699, DOI: 10.1371/journal.pone.0000699 (cited on p. 13).
- [157] PAPAGIAKOUMOU Eirini, Francesca ANSELMI, Aurélien BÈGUE, Vincent de SARS, Jesper GLÜCKSTAD, et al. "Scanless two-photon excitation of channelrhodopsin-2," *Nature methods* 7.10 (2010), 848–54 (cited on pp. 13, 14).
- [158] VAZIRI Alipasha and Valentina EMILIANI. "Reshaping the optical dimension in optogenetics," *Current Opinion in Neurobiology* 22.1 (2012), 128–37 (cited on p. 13).
- [159] ORON Dan, Eran TAL, Yaron SILBERBERG, et al. "Scanningless depth-resolved microscopy," *Opt. Express* 13.5 (2005), 1468–76 (cited on p. 14).
- [160] TAL Eran, Dan ORON, and Yaron SILBERBERG.
  "Improved depth resolution in video-rate line-scanning multiphoton microscopy using temporal focusing,"

  Optics letters 30.13 (2005), 1686–8 (cited on p. 14).
- [161] SELA Gali, Hod DANA, and Shy SHOHAM. "Ultra-deep penetration of temporally-focused two-photon excitation," *SPIE BiOS*, International Society for Optics and Photonics, 2013, 858824–4 (cited on p. 14).
- [162] PACKER Adam M, Botond ROSKA, and Michael HÄUSSER. "Targeting neurons and photons for optogenetics," *Nature neuroscience* 16.7 (2013), 805–15 (cited on p. 14).
- [163] PAPAGIAKOUMOU Eirini, Aurélien BÈGUE, Ben LESHEM, Osip SCHWARTZ, Brandon M STELL, et al. "Functional patterned multiphoton excitation deep inside scattering tissue," *Nature Photonics* (2013) (cited on p. 14).
- [164] STRIKER George, Vinod SUBRAMANIAM, Claus A. M. SEIDEL, and Andreas VOLKMER. "Photochromicity and fluorescence lifetimes of green fluorescent protein," *The Journal of Physical Chemistry B* 103.40 (Oct. 7, 1999), 8612–7, DOI: 10.1021/jp991425e (cited on p. 14).
- [165] WILT Brian A., James E. FITZGERALD, and Mark J. SCHNITZER.

  "Photon shot noise limits on optical detection of neuronal spikes and estimation of spike timing,"

  Biophysical Journal 104.1 (Jan. 8, 2013), 51–62, DOI: 10.1016/j.bpj.2012.07.058 (cited on p. 14).
- [166] KIM Ki Hean, Christof BUEHLER, and Peter T. C. SO. "High-speed, two-photon scanning microscope," *Applied Optics* 38.28 (Oct. 1, 1999), 6004–9, DOI: 10.1364/A0.38.006004 (cited on p. 14).
- [167] MASTERS Barry R. Confocal Microscopy and Multiphoton Excitation Microscopy: The Genesis of Live Cell Imaging, SPIE Press Monograph PM161, Bellingham, WA, USA: SPIE Publications, Feb. 9, 2006, ISBN: 9780819461186, URL: http://spie.org/x648.html?product\_id=660403 (cited on p. 14).
- [168] **DROBIZHEV** Mikhail, Nikolay S MAKAROV, Shane E TILLO, Thomas E HUGHES, and Aleksander REBANE. "Two-photon absorption properties of fluorescent proteins," *Nature methods* 8.5 (2011), 393–9 (cited on p. 14).
- [169] GOEPPERT-MAYER Maria. "Elementary processes with two quantum transitions," Annalen der Physik 18.7-8 (), ISSN: 1521-3889, DOI: 10.1002/andp.200910358, URL: http://dx.doi.org/10.1002/andp.200910358 (cited on p. 14).
- [170] LARSON Daniel R., Warren R. ZIPFEL, Rebecca M. WILLIAMS, Stephen W. CLARK, Marcel P. BRUCHEZ, et al. "Water-soluble quantum dots for multiphoton fluorescence imaging in vivo," *Science* 300.5624 (2003), 1434–6, DOI: 10.1126/science.1083780 (cited on p. 15).

- [171] MARSHALL Jesse D. and Mark J. SCHNITZER.

  "Optical strategies for sensing neuronal voltage using quantum dots and other semiconductor nanocrystals,"

  ACS Nano 7.5 (2013), 4601–9, DOI: 10.1021/nn401410k (cited on p. 15).
- [172] DAHAN M., T. LAURENCE, F. PINAUD, D. S. CHEMLA, A. P. ALIVISATOS, et al. "Time-gated biological imaging by use of colloidal quantum dots," *Opt. Lett.* 26.11 (June 2001), 825–7, DOI: 10.1364/0L.26.000825 (cited on p. 15).
- [173] WILSON Jennifer M., Daniel A. DOMBECK, Manuel DÍAZ-RÍOS, Ronald M. HARRIS-WARRICK, and Robert M. BROWNSTONE. "Two-photon calcium imaging of network activity in XFP-expressing neurons in the mouse," *Journal of Neurophysiology* 97.4 (Apr. 2007), 3118–25, DOI: 10.1152/jn.01207.2006 (cited on p. 15).
- [174] HOPT Alexander and Erwin NEHER. "Highly nonlinear photodamage in two-photon fluorescence microscopy," *Biophysical Journal* 80.4 (2001), 2029–36 (cited on p. 15).
- [175] KÖNIG K, PTC SO, WW MANTULIN, and E GRATTON.

  "Cellular response to near-infrared femtosecond laser pulses in two-photon microscopes," *Optics letters* 22.2 (1997), 135–6 (cited on p. 15).
- [176] NAUMANN Eva A, Adam R KAMPFF, David A PROBER, Alexander F SCHIER, and Florian ENGERT. "Monitoring neural activity with bioluminescence during natural behavior," *Nature neuroscience* 13.4 (2010), 513–20 (cited on p. 15).
- [177] MARTIN Jean-René, Kelly L ROGERS, Carine CHAGNEAU, and Philippe BRÛLET.

  "In vivo bioluminescence imaging of ca2+ signalling in the brain of drosophila," *PLoS One* 2.3 (2007), e275 (cited on p. 15).
- [178] MARTIN Jean-René. "In vivo brain imaging: fluorescence or bioluminescence, which to choose?" *Journal of Neurogenetics* 22.3 (2008), 285–307 (cited on p. 15).
- [179] SELIGER Howard H. and William D. MCELROY. "Spectral emission and quantum yield of firefly bioluminescence," *Archives of Biochemistry and Biophysics* 88.1 (May 1960), 136–41, DOI: 10/d2v3gx (cited on p. 15).
- [180] MOLTER Timothy W., Sarah C. MCQUAIDE, Martin T. SUCHOROLSKI, Tim J. STROVAD, Lloyd W. BURGESS, et al. "A microwell array device capable of measuring single-cell oxygen consumption rates,"

  Sensors and Actuators B: Chemical 135.2 (Jan. 15, 2009), 678–86, DOI: 10.1016/j.snb.2008.10.036 (cited on p. 15).
- [181] BLANCHARD Romain, Svetlana V. BORISKINA, Patrice GENEVET, Mikhail A. KATS, Jean-Philippe TETIENNE, et al. "Multi-wavelength mid-infrared plasmonic antennas with single nanoscale focal point," Optics Express 19.22 (Oct. 24, 2011), 22113–24, DOI: 10.1364/0E.19.022113 (cited on p. 15).
- [182] HARATS Moshe G., Ilai SCHWARZ, Adiel ZIMRAN, Uri BANIN, Gang CHEN, and Ronen RAPAPORT. "Enhancement of two photon processes in quantum dots embedded in subwavelength metallic gratings," *Opt. Express* 19.2 (Jan. 2011), 1617–25, DOI: 10.1364/0E.19.001617 (cited on p. 15).
- [183] WIDDER Edith A., Michael I. LATZ, Peter J. HERRING, and James F. CASE. "Far red bioluminescence from two deep-sea fishes," *Science* 225.4661 (Aug. 3, 1984), 512–4, DOI: 10.1126/science.225.4661.512 (cited on p. 15).
- [184] CAMPBELL Anthony K. and Peter J. HERRING.

  "A novel red fluorescent protein from the deep sea luminous fish *Malacosteus niger*,"

  Comparative Biochemistry and Physiology Part B: Comparative Biochemistry 86B.2 (1987), 411–7, DOI: 10/cz84np (cited on p. 15).
- [185] LIVET Jean, Tamily A. WEISSMAN, Hyuno KANG, Ryan W. DRAFT, Ju LU, et al. "Transgenic strategies for combinatorial expression of fluorescent proteins in the nervous system," *Nature* 450 (Nov. 1, 2007), 56–62, DOI: 10.1038/nature06293 (cited on pp. 15, 29).
- [186] ARNOLD Don B. "Polarized targeting of ion channels in neurons," *Pflügers Archiv-European Journal of Physiology* 453.6 (2007), 763–9 (cited on pp. 15, 26).
- [187] **EL-HUSSEINI** Alaa El-Din, Sarah E CRAVEN, Susannah C BROCK, and David S BREDT. "Polarized targeting of peripheral membrane proteins in neurons," *Journal of Biological Chemistry* 276.48 (2001), 44984–92 (cited on pp. 15, 26).
- [188] CORRÊA Sônia AL, Jürgen MÜLLER, Graham L COLLINGRIDGE, and Neil V MARRION. "Rapid endocytosis provides restricted somatic expression of a k+ channel in central neurons," *Journal of cell science* 122.22 (2009), 4186–94 (cited on pp. 15, 26).
- [189] JACOBS Erin C, Ernesto R BONGARZONE, Celia W CAMPAGNONI, Kathy KAMPF, and Anthony T CAMPAGNONI. "Soma-restricted products of the myelin proteolipid gene are expressed primarily in neurons in the developing mouse nervous system," *Developmental neuroscience* 25.2-4 (2003), 96–104 (cited on pp. 15, 26).

- [190] VACHER Helene, Durga P MOHAPATRA, and James S TRIMMER.

  "Localization and targeting of voltage-dependent ion channels in mammalian central neurons," 
  Physiological reviews 88.4 (2008), 1407–47 (cited on pp. 15, 26).
- [191] BOECKERS Tobias M, Thomas LIEDTKE, Christina SPILKER, Thomas DRESBACH, Jürgen BOCKMANN, et al. "C-terminal synaptic targeting elements for postsynaptic density proteins prosap1/shank2 and prosap2/shank3," *Journal of neurochemistry* 92.3 (2005), 519–24 (cited on pp. 15, 26).
- [192] FEINBERG Evan H, Miri K VANHOVEN, Andres BENDESKY, George WANG, Richard D FETTER, et al. "Gfp reconstitution across synaptic partners (grasp) defines cell contacts and synapses in living nervous systems," *Neuron* 57.3 (2008), 353–63 (cited on pp. 15, 26).
- [193] YAMAGATA Masahito and Joshua R SANES.

  "Transgenic strategy for identifying synaptic connections in mice by fluorescence complementation (grasp),"

  Frontiers in molecular neuroscience 5 (2012) (cited on pp. 15, 26).
- [194] FILONOV Grigory S., Arie KRUMHOLZ, Jun XIA, Junjie YAO, Lihong V. WANG, and Vladislav V. VERKHUSHA. "Deep-tissue photoacoustic tomography of a genetically encoded near-infrared fluorescent probe,"

  Angewandte Chemie: International Edition 51.6 (Feb. 6, 2012), 1448–51, DOI: 10.1002/anie.201107026 (cited on pp. 15, 29).
- [195] USAMI Mitsuo, Hisao TANABE, Akira SATO, Isao SAKAMA, Yukio MAKI, et al.

  "A 0.05×0.05 mm² RFID chip with easily scaled-down ID-memory,"

  International Solid-State Circuits Conference, 2007 IEEE, Digest of Technical Papers, ISSCC 2007 (San Francisco, CA, USA),
  Feb. 11–15, 2007, 482–3, DOI: 10.1109/ISSCC.2007.373504 (cited on p. 16).
- [196] Monza 5 Tag Chip datasheet, Impinj, Inc., Seattle, WA, USA, Feb. 14, 2012,
  URL: http://www.impinj.com/Documents/Tag\_Chips/Monza\_5\_Datasheet (visited on 06/20/2013) (cited on p. 16).
- [197] **BIEDERMAN** William, Daniel J. YAEGER, Nathan NAREVSKY, Aaron C. KORALEK, Jose M. CARMENA, et al. "A fully-integrated, miniaturized (0.125 mm²) 10.5 µW wireless neural sensor," *IEEE Journal of Solid-State Circuits* 48.4 (Apr. 2013), 960–70, DOI: 10.1109/JSSC.2013.2238994 (cited on p. 16).
- [198] CANNON Gregory J. and Joel A. SWANSON. "The macrophage capacity for phagocytosis,"

  Journal of Cell Science 101 (Apr. 1, 1992), 907–13, URL: http://jcs.biologists.org/content/101/4/907 (cited on p. 16).
- [199] KADIU Irena, Ari NOWACEK, Joellyn MCMILLAN, and Howard E. GENDELMAN. "Macrophage endocytic trafficking of antiretroviral nanoparticles," *Nanomedicine* 6.6 (Mar. 27, 2011), DOI: 10.2217/nnm.11.27 (cited on p. 16).
- [200] ENGELHARDT Britta. "Molecular mechanisms involved in t cell migration across the blood-brain barrier,"

  Journal of Neural Transmission 113.4 (2006), 477–85, ISSN: 0300-9564, DOI: 10.1007/s00702-005-0409-y (cited on p. 16).
- [201] CINTI Caterina, Monia TARANTA, Ilaria NALDI, and Settimio GRIMALDI.

  "Newly engineered magnetic erythrocytes for sustained and targeted delivery of anti-cancer therapeutic compounds," 
  PLoS ONE 6.2 (Feb. 2011), e17132, DOI: 10.1371/journal.pone.0017132 (cited on p. 16).
- [202] HIKAWA Naoshi, Hidenori HORIE, Tadashi KAWAKAMI, Kenji OKUDA, and Toshifumi TAKENAKA. "Introduction of macromolecules into primary cultured neuronal cells by fusion with erythrocyte ghosts," *Brain Research* 481.1 (1989), 162 –164, ISSN: 0006-8993, DOI: 10.1016/0006-8993(89)90497-6 (cited on p. 16).
- [203] SPRUSTON Nelson. "Axonal gap junctions send ripples through the hippocampus," *Neuron* 31.5 (2001), 669 –671, ISSN: 0896-6273, DOI: 10.1016/S0896-6273(01)00426-3 (cited on p. 16).
- [204] NGUYEN Quoc-Thang, Lee F SCHROEDER, Marco MANK, Arnaud MULLER, Palmer TAYLOR, et al. "An in vivo biosensor for neurotransmitter release and in situ receptor activity," *Nature neuroscience* 13.1 (2009), 127–32 (cited on p. 16).
- [205] BENNETT Charles H. "Logical reversibility of computation,"

  IBM Journal of Research and Development 17.6 (Nov. 1973), 525–32, DOI: 10.1147/rd.176.0525 (cited on p. 16).
- [206] LANDAUER Rolf W. "Irreversibility and heat generation in the computing process," *IBM Journal of Research and Development* 5.3 (July 1961), 183–91, DOI: 10.1147/rd.53.0183 (cited on p. 16).
- [207] YABLONOVITCH Eli. Replacing the transistor: searching for the milli-volt switch,
  Hong Kong University of Science and Technology, Nov. 21, 2011, URL: http://ustlib.ust.hk/record=b1161099
  (cited on pp. 16–18).
- [208] STEYAERT Michel S. J., Willy M. C. SANSEN, and Zhong-Yuan CHANG.

  "A micropower low-noise monolithic instrumentation amplifier for medical purposes,"

  IEEE Journal of Solid-State Circuits 22.6 (Dec. 1987), 1163–8, DOI: 10.1109/JSSC.1987.1052869 (cited on p. 17).

- [209] TUCKER Rodney S. and Kerry HINTON.

  "Energy consumption and energy density in optical and electronic signal processing,"

  IEEE Photonics Journal 3.5 (Oct. 2011), 821–33, DOI: 10.1109/JPHOT.2011.2166254 (cited on pp. 17, 18).
- [210] KOOMEY Jonathan G., Stephen BERARD, Marla SANCHEZ, and Henry WONG.

  "Implications of historical trends in the electrical efficiency of computing,"

  IEEE Annals of the History of Computing 33.3 (Mar. 2011), 46–54, DOI: 10.1109/MAHC.2010.28 (cited on p. 17).
- [211] TUCKER Rodney S. "Green optical communications Part II: Energy limitations in networks,"

  IEEE Journal of Selected Topics in Quantum Electronics 17.2 (Apr. 7, 2011), 261–74, DOI: 10.1109/JSTQE.2010.2051217 (cited on p. 17).
- [212] UHF / Microwave RFID Transponder ASIC IPMS\_MWST1, Datasheet,
  Fraunhofer Institute for Photonic Microsystems, Nov. 7, 2011, URL: http://www.ipms.fraunhofer.de/content/dam/ipms/common/products/WMS/Transponder/uhf-transponder-e.pdf (visited on 06/16/2013) (cited on p. 17).
- [213] SARPESHKAR Rahul. "Analog versus digital: extrapolating from electronics to neurobiology," *Neural Computation* 10.7 (Oct. 1, 1998), 1601–38, DOI: 10/fvhwq3 (cited on p. 17).
- [214] RAPOPORT Benjamin I., Woradorn WATTANAPANITCH, Héctor Luis PENAGOS VARGAS, Sam MUSALLAM, Richard A. ANDERSEN, and Rahul SARPESHKAR.

  "A biomimetic adaptive algorithm and low-power architecture for implantable neural decoders,"

  \*\*Engineering in Medicine and Biology Society, Annual International Conference of the IEEE,

  EMBC 2009 (Minneapolis, MN, USA), Sept. 3–6, 2009, 4214–7, DOI: 10.1109/IEMBS.2009.5333793 (cited on p. 17).
- [215] MANDAL Soumyajit and Rahul SARPESHKAR. "Low-power CMOS rectifier design for RFID applications," *IEEE Transactions on Circuits and Systems I: Regular Papers* 54.6 (June 11, 2007), 1177–88, DOI: 10.1109/TCSI.2007.895229 (cited on pp. 17, 19).
- [216] SUN Young-Ho and Kai CHANG.

  "A high-efficiency dual-frequency rectenna for 2.45- and 5.8-GHz wireless power transmission,"

  IEEE Transactions on Microwave Theory and Techniques 50.7 (July 2002), 1784–9, DOI: 10.1109/TMTT.2002.800430 (cited on p. 19).
- [217] IONESCU Adrian M. and Heike RIEL. "Tunnel field-effect transistors as energy-efficient electronic switches," *Nature* 479 (Nov. 17, 2011), 329–37, DOI: 10.1038/nature10679 (cited on p. 19).
- [218] LIU Tsu-Jae King, Dejan MARKOVIĆ, Vladimir STOJANOVIĆ, and Elad ALON. "The relay reborn," *IEEE Spectrum* 49.4 (Apr. 2012), 40–3, DOI: 10.1109/MSPEC.2012.6172808 (cited on p. 19).
- [219] OZERI Shaul and Doron SHMILOVITZ. "Ultrasonic transcutaneous energy transfer for powering implanted devices," *Ultrasonics* 50.6 (May 2010), 556–66, DOI: 10.1016/j.ultras.2009.11.004 (cited on p. 19).
- [220] GERSHENFELD Neil. *The physics of information technology*, Cambridge University Press, 2000 (cited on p. 19).
- [221] KITAGAWA Yutaro, Yuji HIRAOKA, Takashi HONDA, Taishi ISHIKURA, Hiroyuki NAKAMURA, and Tsuyoshi KIMURA. "Low-field magnetoelectric effect at room temperature," *Nature Materials* 10 (2010), 797–802, DOI: 10.1038/nmat2826 (cited on p. 19).
- [222] PRIYA Shashank, Jungho RYU, Chee-Sung PARK, Josiah OLIVER, Jong-Jin CHOI, and Dong-Soo PARK. "Piezoelectric and magnetoelectric thick films for fabricating power sources in wireless sensor nodes," *Sensors* 9.8 (2009), DOI: 10.3390/s90806362 (cited on p. 19).
- [223] YUE Kun, Rakesh GUDURU, HONGJEONGMIN, Ping LIANG, Madhavan NAIR, and Sakhrat KHIZROEV. "Magneto-electric nano-particles for non-invasive brain stimulation," *PLoS ONE* 7.9 (Sept. 2012), e44040, DOI: 10.1371/journal.pone.0044040 (cited on p. 19).
- [224] FIEBIG Manfred. "Revival of the magnetoelectric effect," *Journal of Physics D: Applied Physics* 38.8 (2005), R123, URL: http://stacks.iop.org/0022-3727/38/i=8/a=R01 (cited on p. 19).
- [225] KARALIS Aristeidis, J.D. JOANNOPOULOS, and Marin SOLJAČIĆ.

  "Efficient wireless non-radiative mid-range energy transfer," *Annals of Physics* 323.1 (2008), 34 –48, ISSN: 0003-4916, DOI: http://dx.doi.org/10.1016/j.aop.2007.04.017 (cited on p. 19).
- [226] HO J.S., Sanghoek KIM, and A.S.Y. POON. "Midfield wireless powering for implantable systems," Proceedings of the IEEE 101.6 (2013), 1369–78, ISSN: 0018-9219, DOI: 10.1109/JPROC.2013.2251851 (cited on p. 19).
- [227] BETT Andreas W., Frank DIMROTH, Rüdiger LÖCKENHOFF, Eduard OLIVA, and Johannes SCHUBERT. "III-V solar cells under monochromatic illumination," *Photovoltaic Specialists Conference*, 2008, Proceedings of the 33rd IEEE, PVSC'08 (San Diego, CA, USA), May 11–16, 2008, 1–5, DOI: 10.1109/PVSC.2008.4922910 (cited on p. 19).

- [228] SAFARI Ahmad and E. Koray AKDOĞAN, eds. *Piezoelectric and Acoustic Materials for Transducer Applications*, Boston, MA, USA: Springer, 2008, URL: http://www.amazon.com/dp/0387765387 (cited on p. 19).
- [229] XU Tian-Bing, Emilie J. SIOCHI, Jin Ho KANG, Lei ZUO, Wanlu ZHOU, et al. "Energy harvesting using a PZT ceramic multilayer stack," *Smart Materials and Structures* 22 (Apr. 2013), 065015, DOI: 10/mvs (cited on p. 19).
- [230] KRIMHOLTZ Richard S., David A. LEEDOM, and George L. MATTHAEI.

  "New equivalent circuits for elementary piezoelectric transducers," *Electronics Letters* 6.13 (June 25, 1970), 398–9, DOI: 10.1049/el:19700280 (cited on p. 19).
- [231] CASTILLO Martha, Pedro ACEVEDO, and Eduardo MORENO. "KLM model for lossy piezoelectric transducers," *Ultrasonics* 41.8 (Nov. 2003), 671–9, DOI: 10/b4hpcw (cited on p. 19).
- [232] RAPOPORT Benjamin I., Jakub T. KEDZIERSKI, and Rahul SARPESHKAR.

  "A glucose fuel cell for implantable brain-machine interfaces," *PLoS ONE* 7.6 (June 12, 2012), e38436, DOI: 10.1371/journal.pone.0038436 (cited on p. 19).
- [233] SILVER Ian A. and Maria ERECIŃSKA.

  "Extracellular glucose concentration in mammalian brain: continuous monitoring of changes during increased neuronal activity and upon limitation in oxygen supply in normo-, hypo-, and hyperglycemic animals,"

  The Journal of Neuroscience 14.8 (Aug. 1994), 5068–76, URL: http://www.jneurosci.org/content/14/8/5068.long (cited on p. 19).
- [234] IVANOV Kirill P., M. K. KALININA, and Yu I. LEVKOVICH.

  "Blood flow velocity in capillaries of brain and muscles and its physiological significance,"

  \*\*Microvascular Research 22.2 (Sept. 1981), 143–55, DOI: 10/cc8sqk (cited on p. 19).
- [235] CHO Namjun, Seong-Jun SONG, Sunyoung KIM, Shiho KIM, and Hoi-Jun YOO.

  "A 5.1-µw UHF RFID tag chip integrated with sensors for wireless environmental monitoring,"

  Solid-State Circuits Conference, 2005, Proceedings of the 31st European, ESSCIRC 2005 (Grenoble, France),
  Sept. 12–16, 2005, 279–82, DOI: 10.1109/ESSCIR.2005.1541614 (cited on p. 19).
- [236] CARP Stefan A., Nadàege ROCHE-LABARBE, Maria-Angela FRANCESCHINI, Vivek J. SRINIVASAN, Sava SAKADŽIĆ, and David A. BOAS. "Due to intravascular multiple sequential scattering, diffuse correlation spectroscopy of tissue primarily measures relative red blood cell motion within vessels," *Biomedical Optics Express* 2.7 (June 24, 2011), 2047–54, DOI: 10.1364/B0E.2.002047 (cited on p. 22).
- [237] TULINO Antonia M. and Sergio VERDÚ. "Random matrix theory and wireless communications," Foundations and Trends in Communications and Information Theory 1.1 (June 28, 2004), 1–182, DOI: 10.1561/0100000001 (cited on p. 22).
- [238] SHIU Da-shan, Gerard J. FOSCHINI, Michael J. GANS, and Joseph M. KAHN.

  "Fading correlation and its effect on the capacity of multielement antenna systems,"

  IEEE Transactions on Communications 48.3 (Mar. 2000), 502–13, DOI: 10.1109/26.837052 (cited on p. 22).
- [239] SPENCER Quentin H., Christian B. PEEL, A. Lee SWINDLEHURST, and Martin HAARDT.

  "An introduction to the multi-user MIMO downlink," *IEEE Communications Magazine* 42.10 (Oct. 2004), 60–7, DOI: 10.1109/MCOM.2004.1341262 (cited on p. 22).
- [240] MOUSTAKAS Aris L., Harold U. BARANGER, Leon BALENTS, Anirvan SENGUPTA, and Steven H. SIMON. "Communication through a diffusive medium: Coherence and capacity," *Science* 287.5451 (Jan. 14, 2000), 287–90, DOI: 10.1126/science.287.5451.287 (cited on p. 22).
- [241] POPOFF Sebastien M., Geoffroy LEROSEY, Rémi CARMINATI, Mathias FINK, A. Claude BOCCARA, and Sylvain GIGAN. "Measuring the transmission matrix in optics: An approach to the study and control of light propagation in disordered media," *Physical Review Letters* 104.10 (Mar. 12, 2010), e100601, DOI: 10.1103/PhysRevLett.104.100601 (cited on p. 22).
- [242] **BERKOVITS** Richard. "Sensitivity of the multiple-scattering speckle pattern to the motion of a single scatterer," *Physical Review B* 43.10 (Apr. 1, 1991), 8638–40, DOI: 10.1103/PhysRevB.43.8638 (cited on pp. 22, 24).
- [243] DERODE Arnaud, Arnaud TOURIN, Julien de ROSNY, Mickaël TANTER, Sylvain YON, and Mathias FINK. "Taking advantage of multiple scattering to communicate with time-reversal antennas," *Physical Review Letters* 90.1 (Jan. 10, 2003), e014301, DOI: 10.1103/PhysRevLett.90.014301 (cited on p. 23).
- [244] BAKOPOULOS Paraskevas, Irene S. KARANASIOU, Nikos PLEROS, Panagiotis ZAKYNTHINOS, Nikolaos UZUNOGLU, and Hercules AVRAMOPOULOS.

  "A tunable continuous wave (CW) and short-pulse optical source for THz brain imaging applications,"

  Measurement Science and Technology 20 (Sept. 4, 2009), e104001, DOI: 10/dw7hhz (cited on p. 23).

- [245] TONOUCHI Masayoshi. "Cutting-edge terahertz technology," *Nature Photonics* 1 (Feb. 2007), 97–105, DOI: 10.1038/nphoton.2007.3 (cited on p. 23).
- [246] HOSKINS Peter R., Kevin MARTIN, and Abigail THRUSH, eds. *Diagnostic Ultrasound: Physics and Equipment*, 2nd ed., New York: Cambridge University Press, July 26, 2010, URL: http://www.amazon.com/dp/052175710X (cited on p. 23).
- [247] FOSTER F. Stuart, Charles J. PAVLIN, Kasia A. HARASIEWICZ, Donald A. CHRISTOPHER, and Daniel H. TURNBULL. "Advances in ultrasound biomicroscopy," *Ultrasound in Medicine & Biology* 26.1 (Jan. 2000), 1–27, DOI: 10/fmzgkm (cited on p. 23).
- [248] GÓMEZ-MARTÍNEZ Rodrigo, Patricia VÁZQUEZ, Marta DUCH, Alejandro MURIANO, Daniel PINACHO, et al. "Intracellular silicon chips in living cells," *Small* 6.4 (Feb. 22, 2010), 499–502, DOI: 10.1002/sml1.200901041 (cited on p. 23).
- [249] DYSON Freeman. "'Radiotelepathy', the direct communication of feelings and thought from brain to brain," What Will Change Everything? The Edge Annual Question, ed. by BROCKMAN John, Edge.org, Jan. 2009, URL: http://www.edge.org/q2009/q09\_3.html#dyson (cited on p. 23).
- [250] KOMANDURI Ravi K, Chulwoo OH, and Michael J ESCUTI.

  "Reflective liquid crystal polarization gratings with high efficiency and small pitch," *Photonic Devices+ Applications*, International Society for Optics and Photonics, 2008, 70500J–70500J (cited on p. 24).
- [251] PAPPU Ravikanth, Ben RECHT, Jason TAYLOR, and Neil GERSHENFELD. "Physical one-way functions," *Science* 297.5589 (2002), 2026–30 (cited on p. 24).
- [252] BUSH Stephen F. "Toward in vivo nanoscale communication networks: utilizing an active network architecture," Front. Comput. Sci China 5.3 (Sept. 2011), 316–26, ISSN: 1673-7350, DOI: 10.1007/s11704-011-0116-9, URL: http://dx.doi.org/10.1007/s11704-011-0116-9 (cited on p. 24).
- [253] MOHR Peter J., Barry N. TAYLOR, and David B. NEWELL.

  "CODATA recommended values of the fundamental physical constants: 2010,"

  Reviews of Modern Physics 84.4 (Nov. 13, 2012), 1527–605, DOI: 10.1103/RevModPhys.84.1527 (cited on p. 24).
- [254] ROONEY William D., Glyn JOHNSON, Xin LI, Eric R. COHEN, Seong-Gi KIM, et al. "Magnetic field and tissue dependencies of human brain longitudinal <sup>1</sup>H<sub>2</sub>O relaxation in vivo," *Magnetic Resonance in Medicine* 57.2 (Feb. 2007), 308–18, DOI: 10.1002/mrm.21122 (cited on p. 24).
- [255] **DEICHMANN** Ralf, Holger ADOLF, Ulrich NÖTH, Sean P. MORRISSEY, Christian SCHWARZBAUER, and Axel HAASE. "Fast  $T_2$ -mapping with SNAPSHOT FLASH imaging," *Magnetic Resonance Imaging* 13.4 (1995), 633–9, DOI: 10/bpj92v (cited on p. 24).
- [256] OGAWA Seiji, Tso-Ming LEE, Asha S NAYAK, and Paul GLYNN.
  "Oxygenation-sensitive contrast in magnetic resonance image of rodent brain at high magnetic fields,"

  Magnetic resonance in medicine 14.1 (1990), 68–78 (cited on p. 24).
- [257] SIROTIN Yevgeniy B and Aniruddha DAS.

  "Anticipatory haemodynamic signals in sensory cortex not predicted by local neuronal activity,"

  Nature 457.7228 (2009), 475–9 (cited on p. 24).
- [258] LOGOTHETIS Nikos K. "What we can do and what we cannot do with fmri," *Nature* 453.7197 (2008), 869–78 (cited on p. 24).
- [259] JUKOVSKAYA Natalya, Pascale TIRET, Jérôme LECOQ, and Serge CHARPAK. "What does local functional hyperemia tell about local neuronal activation?" *The Journal of Neuroscience* 31.5 (2011), 1579–82 (cited on p. 24).
- [260] BANDETTINI Peter A. "Functional mri limitations and aspirations," *Neural Correlates of Thinking*, Springer, 2009, 15–38 (cited on p. 24).
- [261] SHAPIRO Mikhail G., Gil G. WESTMEYER, Philip A. ROMERO, Jerzy O. SZABLOWSKI, Benedict KÜSTER, et al. "Directed evolution of a magnetic resonance imaging contrast agent for noninvasive imaging of dopamine," *Nature Biotechnology* 28.3 (Mar. 2010), 264–70, DOI: 10.1038/nbt.1609 (cited on pp. 24, 26).
- [262] KORETSKY Alan P. "Is there a path beyond BOLD? Molecular imaging of brain function," *NeuroImage* 62.2 (Aug. 15, 2012), 1208–15, DOI: 10/mvr (cited on p. 24).
- [263] HSIEH Vivian and Alan JASANOFF.

  "Bioengineered probes for molecular magnetic resonance imaging in the nervous system,"

  ACS Chemical Neuroscience 3.8 (online July 11, 2012), 593–602, DOI: 10.1021/cn300059r (cited on p. 24).
- [264] STEHLING Michael K., Robert TURNER, and Peter MANSFIELD, *Science* 254.5028 (Oct. 4, 1991), 43–50, DOI: 10.1126/science.1925560 (cited on p. 24).

- [265] HAASE Axel, Jens FRAHM, Dieter MATTHAEI, Wolfgang HÄNICKE, and Klaus-Dietmar MERBOLDT. "FLASH imaging: Rapid NMR imaging using low flip-angle pulses," *Journal of Magnetic Resonance* 67.2 (Apr. 1986), 258–66, DOI: 10/cz8k2n (cited on p. 24).
- [266] WIESINGER Florian, Pierre-François VAN DE MOORTELE, Gregor ADRIANY, Nicola DE ZANCHE, Kamil UGURBIL, and Klaas P. PRÜSSMAN. "Potential and feasibility of parallel MRI at high field," *NMR in Biomedicine* 19 (2006), 368–78, DOI: 10.1002/nbm.1050 (cited on p. 24).
- [267] GLOVER Paul and Sir Peter MANSFIELD. "Limits to magnetic resonance microscopy," Reports on Progress in Physics 65.10 (Oct. 2002), DOI: 10/cv63tt (cited on pp. 24, 25).
- [268] YU Xin, Daniel GLEN, Shumin WANG, Stephen DODD, Yoshiyuki HIRANO, et al. "Direct imaging of macrovascular and microvascular contributions to BOLD fMRI in layers IV-V of the rat whisker-barrel cortex," *NeuroImage* 59.2 (Jan. 16, 2012), 1451–60, DOI: 10/ff6cts (cited on p. 25).
- [269] NULL Brian, Corey W. LIU, Maj HEDEHUS, Steven CONOLLY, and Ronald W. DAVIS.

  "High-resolution, *in vivo* magnetic resonance imaging of *Drosophila* at 18.8 tesla," *PLoS ONE* 3.7 (July 30, 2008), e2817, DOI: 10.1371/journal.pone.0002817 (cited on p. 25).
- [270] ZHANG Shuo, Kai Tobias BLOCK, and Jens FRAHM.

  "Magnetic resonance imaging in real time: Advances using radial FLASH,"

  Journal of Magnetic Resonance Imaging 31.1 (Jan. 2010), 101–9, DOI: 10.1002/jmri.21987 (cited on p. 25).
- [271] COLLINS Christopher M., Wanzhan LIU, Jinghua WANG, Rolf GRUETTER, J. Thomas VAUGHAN, et al. "Temperature and SAR calculations for a human head within volume and surface coils at 64 and 300 MHz," *Journal of Magnetic Resonance Imaging* 19.5 (May 2004), 650–6, DOI: 10.1002/jmri.20041 (cited on p. 25).
- [272] BODURKA Jerzy and Peter A. BANDETTINI.

  "Toward direct mapping of neuronal activity: mri detection of ultraweak, transient magnetic field changes,"

  Magnetic Resonance in Medicine 47.6 (2002), 1052–8, ISSN: 1522-2594, DOI: 10.1002/mrm.10159 (cited on p. 26).
- [273] PETRIDOU Natalia, Dietmar PLENZ, Afonso C SILVA, Murray LOEW, Jerzy BODURKA, and Peter A BANDETTINI. "Direct magnetic resonance detection of neuronal electrical activity," *Proceedings of the National Academy of Sciences* 103.43 (2006), 16015–20 (cited on p. 26).
- [274] WITZEL Thomas, Fa-Hsuan LIN, Bruce R ROSEN, and Lawrence L WALD. "Stimulus-induced rotary saturation (sirs): a potential method for the detection of neuronal currents with mri," *Neuroimage* 42.4 (2008), 1357–65 (cited on p. 26).
- [275] HALPERN-MANNERS Nicholas W, Vikram S BAJAJ, Thomas Z TEISSEYRE, and Alexander PINES. "Magnetic resonance imaging of oscillating electrical currents," *Proceedings of the National Academy of Sciences* 107.19 (2010), 8519–24 (cited on p. 26).
- [276] ROTH Bradley J and Peter J BASSER. "Mechanical model of neural tissue displacement during lorentz effect imaging," *Magnetic Resonance in Medicine* 61.1 (2009), 59–64 (cited on p. 26).
- [277] TSURUGIZAWA Tomokazu, Luisa CIOBANU, and Denis LE BIHAN. "Water diffusion in brain cortex closely tracks underlying neuronal activity," *Proceedings of the National Academy of Sciences* 110.28 (2013), 11636–41 (cited on p. 26).
- [278] LE BIHAN Denis, Shin-ichi URAYAMA, Toshihiko ASO, Takashi HANAKAWA, and Hidenao FUKUYAMA. "Direct and fast detection of neuronal activation in the human brain with diffusion mri," *Proceedings of the National Academy of Sciences* 103.21 (2006), 8263–8 (cited on p. 26).
- [279] KITAURA Hiroki, Mika TSUJITA, Vincent J HUBER, Akiyoshi KAKITA, Katsuei SHIBUKI, et al. "Activity-dependent glial swelling is impaired in aquaporin-4 knockout mice," *Neuroscience research* 64.2 (2009), 208–12 (cited on p. 26).
- [280] ISOKAWA M. "N-methyl-d-aspartic acid-induced and ca-dependent neuronal swelling and its retardation by brain-derived neurotrophic factor in the epileptic hippocampus," *Neuroscience* 131.4 (2005), 801–12 (cited on p. 26).
- [281] HOLTHOFF Knut and Otto W WITTE. "Intrinsic optical signals in rat neocortical slices measured with near-infrared dark-field microscopy reveal changes in extracellular space," *The Journal of neuroscience* 16.8 (1996), 2740–9 (cited on p. 26).
- [282] JASANOFF Alan. "Bloodless fmri," Trends in neurosciences 30.11 (2007), 603-10 (cited on p. 26).
- [283] LIN Yi-Jen and Alan P KORETSKY.

  "Manganese ion enhances t1-weighted mri during brain activation: an approach to direct imaging of brain function,"

  \*Magnetic resonance in medicine 38.3 (1997), 378–88 (cited on p. 26).
- [284] VAN DER LINDEN A, Marleen VERHOYE, Vincent VAN MEIR, Ilse TINDEMANS, Marcel EENS, et al. "In vivo manganese-enhanced magnetic resonance imaging reveals connections and functional properties of the songbird vocal control system," *Neuroscience* 112.2 (2002), 467–74 (cited on p. 26).

- [285] MADEJCZYK Michael S and Nazzareno BALLATORI.

  "The iron transporter ferroportin can also function as a manganese exporter,"

  Biochimica et Biophysica Acta (BBA)-Biomembranes 1818.3 (2012), 651–7 (cited on p. 26).
- [286] MADISEN Linda, Theresa A ZWINGMAN, Susan M SUNKIN, Seung Wook OH, Hatim A ZARIWALA, et al. "A robust and high-throughput cre reporting and characterization system for the whole mouse brain," *Nature neuroscience* 13.1 (2009), 133–40 (cited on p. 26).
- [287] LUO Liqun, Edward M CALLAWAY, and Karel SVOBODA. "Genetic dissection of neural circuits," *Neuron* 57.5 (2008), 634–60 (cited on p. 26).
- [288] ATANASIJEVIC Tatjana, Maxim SHUSTEFF, Peter FAM, and Alan JASANOFF.

  "Calcium-sensitive MRI contrast agents based on super paramagnetic iron oxide nanoparticles and calmodulin," 
  Proceedings of the National Academy of Sciences of the United States of America 103.40 (Oct. 3, 2006), 14707–12, 
  DOI: 10.1073/pnas.0606749103 (cited on p. 26).
- [289] LI Wen-hong, Scott E. FRASER, and Thomas J. MEADE. "A calcium-sensitive magnetic resonance imaging contrast agent," *Journal of the American Chemical Society* 121.6 (Feb. 17, 1999), 1413–4, DOI: 10.1021/ja9837021 (cited on p. 26).
- [290] Shapiro Mikhail G., Tatjana Atanasijevic, Henryk Faas, Gil G. Westmeyer, and Alan Jasanoff. "Dynamic imaging with MRI contrast agents: quantitative considerations,"

  \*\*Magnetic Resonance Imaging 24.4 (May 2006), 449–62, DOI: 10.1016/j.mri.2005.12.033 (cited on p. 26).
- [291] JASANOFF Alan and Phillip Z SUN. "In vivo magnetic resonance microscopy of brain structure in unanesthetized flies," *Journal of Magnetic Resonance* 158.1 (2002), 79–85 (cited on p. 26).
- [292] FRIEDLAND Ari E., Timothy K. Lu, Xiao WANG, David SHI, George M. CHURCH, and James J. COLLINS. "Synthetic gene networks that count," *Science* 324.5931 (May 29, 2009), 1199–202, DOI: 10.1126/science.1172005 (cited on p. 27).
- [293] BONNET Jerome, Peter YIN, Monica E. ORTIZ, Pakpoom SUBSOONTORN, and Drew ENDY. "Amplifying genetic logic gates," *Science* 340.6132 (May 3, 2013), 599–603, DOI: 10.1126/science.1232758 (cited on p. 27).
- [294] **KELMAN** Zvi and Mike O'DONNELL.

  "DNA polymerase III holoenzyme: Structure and function of a chromosomal replicating machine,"

  Annual Review of Biochemistry 64 (July 1995), 171–200, DOI: 10/cvmv32 (cited on p. 27).
- [295] FRANK Ekaterina G. and Roger WOODGATE.

  "Increased catalytic activity and altered fidelity of human DNA polymerase iota in the presence of manganese,"

  The Journal of Biological Chemistry 282.34 (Aug. 24, 2007), 24689–96, DOI: 10/cv5bmm (cited on p. 27).
- [296] OCHMAN Howard and Allan C. WILSON.

  "Evolution in bacteria: evidence for a universal substitution rate in cellular genomes," English,

  Journal of Molecular Evolution 26.1-2 (1987), 74–86, ISSN: 0022-2844, DOI: 10.1007/BF02111283,

  URL: http://dx.doi.org/10.1007/BF02111283 (cited on p. 27).
- [297] ELOWITZ Michael B. and Stanislas LEIBLER. "A synthetic oscillatory network of transcriptional regulators," *Nature* 6767 (2000), 335–338, DOI: 10.1038/35002125 (cited on p. 27).
- [298] BERNSTEIN Jacob G., Paul A GARRITY, and Edward S. BOYDEN.

  "Optogenetics and thermogenetics: technologies for controlling the activity of targeted cells within intact neural circuits,"

  Current Opinion in Neurobiology 22.1 (2012), 61 –71, ISSN: 0959-4388,

  DOI: http://dx.doi.org/10.1016/j.conb.2011.10.023 (cited on p. 27).
- [299] KONERMANN Silvana, Mark D BRIGHAM, Alexandro TREVINO, Patrick D HSU, Matthias HEIDENREICH, et al. "Optical control of mammalian endogenous transcription and epigenetic states," *Nature* (2013) (cited on p. 27).
- [300] ZADOR Anthony M., Joshua DUBNAU, Hassana K. OYIBO, Huiqing ZHAN, Gang CAO, and Ian D. PEIKON. "Sequencing the connectome," *PLoS Biology* 10.10 (Oct. 23, 2012), e1001411, DOI: 10.1371/journal.pbio.1001411 (cited on pp. 27–29).
- [301] LEE Je-Hyuk, Evan DAUGHARTHY, Jonathan SCHEIMAN, Reza KALHOR, Richard TERRY, et al. "Highly multiplexed three-dimensional subcellular transcriptome sequencing in situ," (Submitted) (cited on pp. 27–29).
- [302] CHUNG Kwanghun, Jenelle WALLACE, Sung-Yon KIM, Sandhiya KALYANASUNDARAM, Aaron S ANDALMAN, et al. "Structural and molecular interrogation of intact biological systems," *Nature* 497.7449 (2013), 332–7 (cited on p. 27).
- [303] JACKSON Dean A., Ana POMBO, and Francisco IBORRA.

  "The balance sheet for transcription: An analysis of nuclear RNA metabolism in mammalian cells,"

  The FASEB Journal 14.2 (Feb. 2000), 242–54, URL: http://www.fasebj.org/content/14/2/242.long (cited on p. 27).

- [304] MISHCHENKO Yuriy. "On optical detection of densely labeled synapses in neuropil and mapping connectivity with combinatorially multiplexed fluorescent synaptic markers," *PloS one* 5.1 (2010), e8853 (cited on p. 28).
- [305] SANJANA Neville E, Erez Y LEVANON, Emily A HUESKE, Jessica M AMBROSE, and Jin Billy LI. "Activity-dependent a-to-i rna editing in rat cortical neurons," *Genetics* 192.1 (2012), 281–7 (cited on p. 28).
- [306] DEAN Thomas, Biafra AHANONU, Mainak CHOWDHURY, Anjali DATTA, Andre ESTEVA, et al. "On the technology prospects and investment opportunities for scalable neuroscience," (2013), URL: http://arxiv.org/abs/1307.7302 (cited on p. 28).
- [307] WOOD Lowell, Personal Communication, July 30, 2013 (cited on pp. 28–30).
- [308] LAMY F. and J.J. FONTAINE. "Transparent media characterization using subpicosecond dye laser," (1981), URL: http://cds.cern.ch/record/870056/files/p35.pdf (cited on p. 29).
- [309] HYNYNEN Kullervo, Nathan MCDANNOLD, Nickolai A SHEIKOV, Ferenc A JOLESZ, and Natalia VYKHODTSEVA. "Local and reversible blood-brain barrier disruption by noninvasive focused ultrasound at frequencies suitable for trans-skull sonications," *Neuroimage* 24.1 (2005), 12–20 (cited on p. 29).
- [310] IWASA Kunihiko, Ichiji TASAKI, and Robert C GIBBONS. "Swelling of nerve fibers associated with action potentials," *Science* 210.4467 (1980), 338–9 (cited on p. 29).
- [311] OH Seungeun, Christopher FANG-YEN, Wonshik CHOI, Zahid YAQOOB, Dan FU, et al. "Label-free imaging of membrane potential using membrane electromotility," *Biophysical Journal* 103.1 (2012), 11–8 (cited on p. 29).
- [312] FANG-YEN Christopher, Mark C CHU, H Sebastian SEUNG, Ramachandra R DASARI, and Michael S FELD. "Noncontact measurement of nerve displacement during action potential with a dual-beam low-coherence interferometer," Optics letters 29.17 (2004), 2028–30 (cited on p. 29).
- [313] KIM GH, P KOSTERIN, AL OBAID, and BM SALZBERG.

  "A mechanical spike accompanies the action potential in mammalian nerve terminals,"

  Biophysical journal 92.9 (2007), 3122–9 (cited on p. 29).
- [314] STEPNOSKI RA, A LAPORTA, F RACCUIA-BEHLING, GE BLONDER, RE SLUSHER, and D KLEINFELD. "Noninvasive detection of changes in membrane potential in cultured neurons by light scattering.," *Proceedings of the National Academy of Sciences* 88.21 (1991), 9382–6 (cited on p. 29).
- [315] LAZEBNIK Mariya, Daniel L MARKS, Kurt POTGIETER, Rhanor GILLETTE, and Stephen A BOPPART. "Functional optical coherence tomography for detecting neural activity through scattering changes," *Optics letters* 28.14 (2003), 1218–20 (cited on p. 29).
- [316] HOSHI Yoko. "Functional near-infrared optical imaging: utility and limitations in human brain mapping," *Psychophysiology* 40.4 (2003), 511–20 (cited on p. 29).
- [317] PARPALEIX Alexandre, Yannick Goulam HOUSSEN, and Serge CHARPAK. "Imaging local neuronal activity by monitoring po2 transients in capillaries," *Nature medicine* (2013) (cited on p. 29).
- [318] LECOQ Jérôme, Alexandre PARPALEIX, Emmanuel ROUSSAKIS, Mathieu DUCROS, Yannick Goulam HOUSSEN, et al. "Simultaneous two-photon imaging of oxygen and blood flow in deep cerebral vessels," *Nature medicine* 17.7 (2011), 893–8 (cited on p. 29).
- [319] HILLMAN Elizabeth MC. "Optical brain imaging in vivo: techniques and applications from animal to man," *Journal of biomedical optics* 12.5 (2007), 51402–2 (cited on p. 29).
- [320] JOSEPH Danny K, Theodore J HUPPERT, Maria Angela FRANCESCHINI, and David A BOAS. "Diffuse optical tomography system to image brain activation with improved spatial resolution and validation with functional magnetic resonance imaging," *Applied optics* 45.31 (2006), 8142–51 (cited on p. 29).
- [321] HUPPERT Theodore J, Maria Angela FRANCESCHINI, and David A BOAS. "14 noninvasive imaging of cerebral activation with diffuse optical tomography," (2009) (cited on p. 29).
- [322] SHAPIRO MG, PW GOODWILL, A NEOGY, DV SCHAFFER, and SM CONOLLY. "Genetically encoded gas nanostructures as ultrasonic molecular reporters," (In revision) (cited on p. 29).
- [323] DEISSEROTH Karl. "Optogenetics and psychiatry: applications, challenges, and opportunities.," *Biological psychiatry* 12 (2012), 71 (cited on p. 29).
- [324] SCHRÖDEL Tina, Robert PREVEDEL, Karin AUMAYR, Manuel ZIMMER, and Alipasha VAZIRI. "Brain-wide 3d imaging of neuronal activity in caenorhabditis elegans with sculpted light," *Nature Methods* (2013) (cited on p. 29).
- [325] RASKAR Ramesh and Jack TUMBLIN.

  Computational Photography: Mastering New Techniques for Lenses, Lighting, and Sensors, AK Peters, Ltd., 2009 (cited on p. 30).

- [326] VELTEN Andreas, Thomas WILLWACHER, Otkrist GUPTA, Ashok VEERARAGHAVAN, Moungi G BAWENDI, and Ramesh RASKAR. "Recovering three-dimensional shape around a corner using ultrafast time-of-flight imaging," *Nature Communications* 3 (2012), 745 (cited on p. 30).
- [327] KIM Jaewon. "Next generation CAT system," PhD thesis, Massachusetts Institute of Technology, 2010, URL: http://dspace.mit.edu/bitstream/handle/1721.1/62117/709605980.pdf (cited on p. 30).
- [328] PNEVMATIKAKIS Eftychios A and Liam PANINSKI.

  "Sparse nonnegative deconvolution for compressive calcium imaging: algorithms and phase transitions,"

  Advances in Neural Information Processing Systems (NIPS) (2013),

  URL: http://stat.columbia.edu/~liam/research/pubs/eftychios-CS-calcium.pdf (cited on p. 30).
- [329] LAUXTERMANN Stefan, Georg ISRAEL, Peter SEITZ, Hans BLOSS, Jürgen ERNST, et al. "A mega-pixel high speed cmos imager with sustainable gigapixel/sec readout rate," at the 2001 IEEE Workshop on Charge-Coupled Devices and Advanced Image Sensors, 2001 (cited on p. 30).
- [330] P series linear photodiode array imager datasheet, Reticon, Inc., Sunnyvale, CA, USA, URL: http://www.jai.com/sitecollectionimages/sensor\_datasheet\_p-series-021web.pdf (visited on 08/24/2013) (cited on p. 30).
- [331] GODA Keisuke, Ali AYAZI, Daniel R GOSSETT, Jagannath SADASIVAM, Cejo K LONAPPAN, et al. "High-throughput single-microparticle imaging flow analyzer," Proceedings of the National Academy of Sciences 109.29 (2012), 11630–5 (cited on p. 30).
- [332] GODA K, KK TSIA, and B JALALI. "Serial time-encoded amplified imaging for real-time observation of fast dynamic phenomena," *Nature* 458.7242 (2009), 1145–9 (cited on p. 30).
- [333] GODA Keisuke, Kevin K TSIA, and Bahram JALALI.

  "Amplified dispersive fourier-transform imaging for ultrafast displacement sensing and barcode reading,"

  Applied Physics Letters 93.13 (2008), 131109–9 (cited on p. 30).
- [334] GODA Keisuke, Daniel R SOLLI, Kevin K TSIA, and Bahram JALALI. "Theory of amplified dispersive fourier transformation," *Physical Review A* 80.4 (2009), 043821 (cited on p. 30).
- [335] MAHJOUBFAR Ata, Keisuke GODA, Ali AYAZI, Ali FARD, Sang Hyup KIM, and Bahram JALALI. "High-speed nanometer-resolved imaging vibrometer and velocimeter," *Applied Physics Letters* 98.10 (2011), 101107–7 (cited on p. 30).
- [336] TSIA Kevin K, Keisuke GODA, Dale CAPEWELL, and Bahram JALALI.

  "Performance of serial time-encoded amplified microscopy,"

  Lasers and Electro-Optics (CLEO) and Quantum Electronics and Laser Science Conference (QELS), 2010 Conference on, IEEE, 2010, 1–2 (cited on p. 30).
- [337] GODA K and B JALALI. "Dispersive fourier transformation for fast continuous single-shot measurements," *Nature Photonics* 7.2 (2013), 102–12 (cited on p. 30).
- [338] MOOSMANN Julian, Alexey ERSHOV, Venera ALTAPOVA, Tilo BAUMBACH, Maneeshi S. PRASAD, et al. "X-ray phase-contrast *in vivo* microtomography probes new aspects of *Xenopus* gastrulation," *Nature* 497 (May 16, 2013), 374–7, DOI: 10.1038/nature12116 (cited on p. 30).
- [339] ISSLER Sandy L. and Charlie C. TORARDI. "Solid state chemistry and luminescence of X-ray phosphors," *Journal of Alloys and Compounds* 229.1 (Oct. 15, 1995), 54–65, DOI: 10/fpz8gt (cited on p. 30).
- [340] LARABELL Carolyn A. and Mark A. LE GROS.

  "X-ray tomography generates 3-D reconstructions of the yeast, *Saccharomyces cerevisiae*, at 60-nm resolution," *Molecular Biology of the Cell* 15.3 (Mar. 1, 2004), 957–62, DOI: 10/br3d8z (cited on p. 30).
- [341] HOROWITZ Viva R., Benjamín J. ALEMÁN, David J. CHRISTLE, Andrew N. CLELAND, and David D. AWSCHALOM. "Electron spin resonance of nitrogen-vacancy centers in optically trapped nanodiamonds," *Proceedings of the National Academy of Sciences of the United States of America* 109.34 (Aug. 21, 2012), 13493–7, DOI: 10.1073/pnas.1211311109 (cited on p. 30).
- [342] DOLDE Florian, Helmut FEDDER, Marcus W. DOHERTY, Tobias NÖBAUER, Florian REMPP, et al. "Electric-field sensing using single diamond spins," *Nature Physics* 7 (June 2011), 459–63, DOI: 10.1038/nphys1969 (cited on p. 30).
- [343] HALL LT, GCG BEART, EA THOMAS, DA SIMPSON, LP McGUINNESS, et al. "High spatial and temporal resolution wide-field imaging of neuron activity using quantum nv-diamond," *Scientific reports* 2 (2012), DOI: 10.1038/srep00401 (cited on p. 30).

- [344] WEE Tse-Luen, Yan-Kai TZENG, Chau-Chung HAN, Huan-Cheng CHANG, Wunshain FANN, et al. "Two-photon excited fluorescence of nitrogen-vacancy centers in proton-irradiated type ib diamond," *The Journal of Physical Chemistry A* 111.38 (2007), 9379–86, DOI: 10.1021/jp0739380 (cited on p. 30).
- [345] SADEK Akram S, Rassul B KARABALIN, Jiangang DU, Michael L ROUKES, Christof KOCH, and Sotiris C MASMANIDIS. "Wiring nanoscale biosensors with piezoelectric nanomechanical resonators," *Nano letters* 10.5 (2010), 1769–73 (cited on p. 30).
- [346] JASANOFF Alan. "Noninvasive imaging-based electrophysiology using microelectronic devices," (NIH Grant 1R01NS076462-01),
  URL: http://projectreporter.nih.gov/project\_info\_description.cfm?aid=8180844 (cited on p. 30).